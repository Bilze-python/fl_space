"""动态通信的物理量级、服务守恒、时间边界和轨道适配验证。"""

from dataclasses import replace
import json
import math

import pytest

from fl_space.communication import (
    DynamicCommunication,
    GeometrySample,
    Message,
    RFConfig,
    TransferResult,
    evaluate_link,
    line_of_sight,
    orbit_position_provider,
    sample_geometry,
)
from fl_space.environment import CelestialBody
from fl_space.orbit import KeplerOrbit, OrbitalElements


def fixed(**kwargs):
    return RFConfig(mode="fixed", fixed_rate_bps=8, acquisition_s=0, switching_s=0, **kwargs)


def test_budget_range_noise_gain_and_pointing():
    config = RFConfig(max_rate_bps=1e12, min_snr_db=-100)
    near = evaluate_link(GeometrySample(0, 1, 1e6), config)
    far = evaluate_link(GeometrySample(0, 1, 2e6), config)
    assert far.fspl_db - near.fspl_db == pytest.approx(6.0206, abs=1e-4)
    assert near.fspl_db == pytest.approx(178.468, abs=0.01)
    assert near.noise_power_dbw == pytest.approx(-130.975, abs=0.02)
    assert far.rate_bps < near.rate_bps
    gain = evaluate_link(GeometrySample(0, 1, 1e6), replace(config, tx_gain_dbi=23))
    loss = evaluate_link(GeometrySample(0, 1, 1e6, pointing_loss_db=3), config)
    assert gain.snr_db - near.snr_db == pytest.approx(3)
    assert near.snr_db - loss.snr_db == pytest.approx(3)
    assert evaluate_link(GeometrySample(0, 1, 35786e3), config).propagation_s == pytest.approx(
        0.11937, abs=1e-5
    )


def test_visibility_threshold_and_fixed_baseline():
    sample = GeometrySample(0, 1, 1e6)
    assert evaluate_link(sample, RFConfig(min_snr_db=100)).rate_bps == 0
    assert evaluate_link(sample, fixed()).rate_bps == 8
    assert evaluate_link(replace(sample, visible=False), fixed()).rate_bps == 0


def test_wgs84_surface_station_and_earth_occlusion():
    assert line_of_sight((6378137, 0, 0), (6928137, 0, 0))
    assert not line_of_sight((6378137, 0, 0), (-6928137, 0, 0))
    assert not line_of_sight((6928137, 0, 0), (-6928137, 0, 0))
    assert line_of_sight((6378137, 0, 0), (6378137, 1e6, 0))  # 地平线切线


def test_transmission_finish_is_not_delivery_and_deadline_uses_arrival():
    sample = GeometrySample(0, 1, 299792458)
    result = DynamicCommunication(fixed()).transfer(Message("m", 1), [sample])
    assert result.tx_finished_s == 1
    assert result.arrival_s == 2
    assert result.status == "in_flight"
    assert result.delivered_bytes == 0
    assert not any(event.kind == "delivered" for event in result.events)
    longer = [replace(sample, end_s=3)]
    expired = DynamicCommunication(fixed()).transfer(Message("m", 1, deadline_s=1.5), longer)
    assert expired.status == "expired"
    delivered = DynamicCommunication(fixed()).transfer(Message("m", 1, deadline_s=2), longer)
    assert delivered.status == "delivered"
    assert delivered.delivered_bytes == 1


def test_outage_resume_with_acquisition_and_no_double_counting():
    config = replace(fixed(), acquisition_s=0.5, switching_s=0.5)
    samples = [
        GeometrySample(0, 1, 1),
        GeometrySample(1, 2, 1, visible=False),
        GeometrySample(2, 3, 1),
        GeometrySample(3, 5, 1),
    ]
    result = DynamicCommunication(config).transfer(Message("m", 2), samples)
    assert result.status == "delivered"
    assert result.tx_finished_s == pytest.approx(4.5)
    assert result.transmitted_bits == 16
    assert result.transmit_energy_j == pytest.approx(2)
    assert sum(event.bits for event in result.events) == 16
    assert len([event for event in result.events if event.kind == "delivered"]) == 1


def test_acquisition_spans_samples_and_switching_cost():
    config = replace(fixed(), acquisition_s=2, switching_s=1)
    samples = [
        GeometrySample(0, 1, 1),
        GeometrySample(1, 3, 1),
        GeometrySample(3, 10, 1, link_id="other"),
    ]
    result = DynamicCommunication(config).transfer(Message("m", 2), samples)
    assert result.tx_finished_s == pytest.approx(7)
    assert result.transmit_energy_j == pytest.approx(2)
    acquisition = [event for event in result.events if event.kind == "acquiring"]
    assert sum(event.end_s - event.time_s for event in acquisition) == pytest.approx(5)


def test_delivery_waits_for_earlier_fragments_with_longer_propagation():
    samples = [
        GeometrySample(0, 1, 3 * 299792458),
        GeometrySample(1, 2, 1),
        GeometrySample(2, 5, 1, visible=False),
    ]
    result = DynamicCommunication(fixed()).transfer(Message("m", 2), samples)
    assert result.tx_finished_s == 2
    assert result.arrival_s == 4
    assert result.status == "delivered"


def test_no_contact_never_delivers_or_consumes_tx_energy():
    result = DynamicCommunication().transfer(
        Message("m", 1, deadline_s=5), [GeometrySample(0, 10, 1e6, visible=False)]
    )
    assert result.status == "expired"
    assert result.transmitted_bits == result.delivered_bytes == result.transmit_energy_j == 0
    assert result.arrival_s is None


def test_gaps_no_extrapolation_and_message_created_mid_interval():
    engine = DynamicCommunication(replace(fixed(), acquisition_s=0.5))
    samples = [GeometrySample(0, 1, 1), GeometrySample(3, 5, 1)]
    result = engine.transfer(Message("m", 2, created_s=0.25), samples)
    assert result.status == "pending"
    assert result.transmitted_bits == pytest.approx(14)
    assert result.observed_until_s == 5
    assert result.arrival_s is None


def test_dynamic_service_integrates_rate_and_matches_fixed_when_equal():
    config = RFConfig(acquisition_s=0, switching_s=0, max_rate_bps=1e12, min_snr_db=-100)
    samples = [GeometrySample(0, 1, 1e6), GeometrySample(1, 2, 2e6)]
    result = DynamicCommunication(config).transfer(Message("large", 10**9), samples)
    expected = sum(evaluate_link(sample, config).rate_bps for sample in samples)
    assert result.transmitted_bits == pytest.approx(expected)
    constant = replace(
        config, mode="fixed", fixed_rate_bps=evaluate_link(samples[0], config).rate_bps
    )
    message = Message("small", 100)
    assert DynamicCommunication(config).transfer(message, samples).arrival_s == pytest.approx(
        DynamicCommunication(constant).transfer(message, samples).arrival_s
    )


def test_roundtrip_json():
    config = RFConfig()
    assert RFConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    sample = GeometrySample(0, 3, 1)
    assert GeometrySample.from_dict(sample.to_dict()) == sample
    message = Message("m", 1)
    assert Message.from_dict(message.to_dict()) == message
    result = DynamicCommunication(fixed()).transfer(message, [sample])
    assert (
        TransferResult.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
        == result
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bandwidth_hz": 0},
        {"frequency_hz": math.nan},
        {"efficiency": 1.1},
        {"switching_s": -1},
        {"tx_power_w": math.inf},
        {"mode": "optical"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        RFConfig(**kwargs)


def test_invalid_geometry_message_and_order():
    with pytest.raises(ValueError):
        GeometrySample(1, 1, 1)
    with pytest.raises(ValueError):
        Message("m", 1.5)
    with pytest.raises(ValueError):
        Message("m", 1, created_s=2, deadline_s=1)
    with pytest.raises(ValueError):
        sample_geometry(lambda _: (1, 2, 3), lambda _: (4, 5, 6), [0, 0])
    with pytest.raises(ValueError):
        DynamicCommunication().transfer(
            Message("m", 1), [GeometrySample(0, 2, 1), GeometrySample(1, 3, 1)]
        )


def test_kepler_adapter_uses_seconds_and_si_units():
    orbit = KeplerOrbit(
        OrbitalElements(semi_major_axis_km=6928.137, inclination_deg=0), CelestialBody.earth()
    )
    position = orbit_position_provider(orbit)
    assert position(60) == pytest.approx(tuple(value * 1000 for value in orbit.ecef_at_time(1)))
    samples = sample_geometry(
        position, lambda _: (6378137, 0, 0), [0, 30, 60], visibility=lambda time: time < 30
    )
    assert samples[0].distance_m == pytest.approx(550000)
    assert samples[0].visible
    assert not samples[1].visible
    assert samples[1].distance_m != samples[0].distance_m
    result = DynamicCommunication().transfer(Message("model-update", 1024), samples)
    assert result.status == "delivered"
    assert 0 < result.arrival_s < 30


def test_cli_exports_reproducible_comparison(tmp_path, monkeypatch):
    from fl_space.communication.__main__ import main

    output = tmp_path / "baseline.json"
    monkeypatch.setattr("sys.argv", ["communication", "--output", str(output)])
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["fidelity"]["orbit"] == "kepler"
    assert report["source"] == "assumption"
    assert report["output_path"] == str(output.resolve())
    assert report["runs"]["fixed"]["result"]["status"] == "delivered"
    assert report["runs"]["dynamic"]["result"]["status"] == "delivered"
    assert (
        report["runs"]["dynamic"]["result"]["arrival_s"]
        > report["runs"]["fixed"]["result"]["arrival_s"]
    )
    assert len(report["runs"]["dynamic"]["links"]) == 240
