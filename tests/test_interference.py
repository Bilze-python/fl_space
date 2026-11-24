"""接收端干扰功率、因果观测、换频成本及分组交付集成。"""

from dataclasses import replace
import json

import pytest

from fl_space.communication import (
    AntiJammingConfig,
    FlowMessage,
    GeometrySample,
    InterferenceBurst,
    InterferenceEnvironment,
    Message,
    PacketNetwork,
    PacketNetworkConfig,
    RFConfig,
    evaluate_link,
    interference_from_link,
)


def burst(
    source="jam", receiver="ground", start=0, end=10, frequency=20e9, bandwidth=10e6, power=1e-8
):
    return InterferenceBurst(source, receiver, start, end, frequency, bandwidth, power)


def test_linear_power_sum_spectral_overlap_receiver_and_time():
    env = InterferenceEnvironment(
        (burst(), burst("second", power=2e-8), burst("wrong", receiver="other"))
    )
    assert env.power_w("ground", 1, 20e9, 10e6) == pytest.approx(3e-8)
    assert env.power_w("ground", 1, 20.005e9, 10e6) == pytest.approx(1.5e-8)
    assert env.power_w("ground", 1, 20.02e9, 10e6) == 0
    assert env.power_w("ground", 10, 20e9, 10e6) == 0


def test_more_interference_never_increases_sinr_or_rate():
    config = RFConfig(min_snr_db=-100)
    sample = GeometrySample(0, 1, 550000)
    budgets = [
        evaluate_link(replace(sample, interference_w=power), config)
        for power in (0, 1e-14, 1e-12, 1e-10)
    ]
    assert budgets[0].sinr_db == pytest.approx(budgets[0].snr_db)
    assert all(
        right.sinr_db < left.sinr_db and right.rate_bps < left.rate_bps
        for left, right in zip(budgets, budgets[1:])
    )
    jammed = evaluate_link(replace(sample, interference_w=1e-8), RFConfig())
    assert jammed.rate_bps == 0
    assert jammed.reason == "below_sinr_threshold"


def test_link_projection_obeys_visibility():
    sample = GeometrySample(0, 1, 550000)
    config = RFConfig()
    projected = interference_from_link("jam", "ground", sample, config)
    assert projected.received_power_w == pytest.approx(
        10 ** (evaluate_link(sample, config).received_power_dbw / 10)
    )
    assert (
        interference_from_link(
            "jam", "ground", replace(sample, visible=False), config
        ).received_power_w
        == 0
    )


def test_observation_has_no_future_knowledge():
    env = InterferenceEnvironment((burst(start=1), burst("new", start=2, frequency=20.02e9)))
    config = AntiJammingConfig(observation_delay_s=1)
    assert env.choose(config, "ground", 0, 0, 10e6, "key") == (0, None, [])
    selected, time, _ = env.choose(config, "ground", 1.5, 0, 10e6, "key")
    assert selected == 0 and time == 0.5
    selected, time, powers = env.choose(config, "ground", 2, 0, 10e6, "key")
    assert selected == 1 and time == 1 and powers[1] == 0
    assert env.choose(config, "ground", 3, 1, 10e6, "key")[0] == 2


def test_burst_boundaries_split_intervals_without_averaging():
    env = InterferenceEnvironment((burst(start=0.2, end=0.3),))
    samples = env.apply([GeometrySample(0, 1, 1)], "ground", 20e9, 10e6, 0.1)
    assert [(sample.start_s, sample.end_s) for sample in samples] == [
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 1),
    ]
    assert [sample.interference_w for sample in samples] == [0, 1e-8, 0]


def simulate(anti, bursts=None, until=10):
    item = FlowMessage("f", "sat", "ground", Message("m", 1200, deadline_s=until))
    rf = RFConfig(acquisition_s=0)
    env = InterferenceEnvironment((burst(end=until),) if bursts is None else bursts)
    return PacketNetwork(rf, PacketNetworkConfig(), anti, env).run(
        [item], {"f": [GeometrySample(0, until, 550000)]}, until
    )


def test_observed_defense_delivers_while_fixed_expires_and_costs_are_charged():
    config = AntiJammingConfig(observation_delay_s=1)
    assert simulate(replace(config, policy="fixed")).messages["m"]["status"] == "expired"
    result = simulate(config)
    assert result.messages["m"]["status"] == "delivered"
    assert result.metrics["channel_switches"] == 1
    assert result.metrics["retune_resync_energy_j"] == pytest.approx(0.04)
    tx = next(event for event in result.events if event["kind"] == "transmitting")
    assert tx["time_s"] >= 1.05
    assert result.metrics["sensing_energy_j"] > 0
    assert result.metrics["data_and_defense_energy_j"] > result.metrics["transmit_energy_j"]


def test_excessive_switch_overhead_cannot_send_and_charges_only_actual_time():
    config = AntiJammingConfig(observation_delay_s=0, retune_s=10, resync_s=0, sensing_s=0)
    result = simulate(config, until=2)
    assert result.metrics["unique_delivered_bytes"] == 0
    assert result.metrics["channel_switches"] == 0
    assert result.metrics["retune_resync_energy_j"] == pytest.approx(2)


def test_broadband_jam_has_no_magic_escape():
    result = simulate(AntiJammingConfig(), (burst(frequency=20.02e9, bandwidth=100e6, power=1e-6),))
    assert result.messages["m"]["status"] == "expired"


def test_defense_energy_stops_at_deadline():
    config = AntiJammingConfig(observation_delay_s=0, sensing_s=1, sensing_power_w=2)
    result = simulate(config, until=0.1)
    assert result.metrics["sensing_energy_j"] == pytest.approx(0.2)
    assert result.metrics["unique_delivered_bytes"] == 0


def test_no_interference_control_needs_no_switch_and_defense_can_cost_time():
    config = AntiJammingConfig(observation_delay_s=0)
    fixed = simulate(replace(config, policy="fixed"), ())
    observed = simulate(config, ())
    assert observed.metrics["channel_switches"] == 0
    assert observed.messages["m"]["arrival_s"] > fixed.messages["m"]["arrival_s"]


def test_roundtrip_random_reproducibility_and_invalid_values():
    config = AntiJammingConfig(policy="random")
    assert AntiJammingConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert InterferenceBurst.from_dict(burst().to_dict()) == burst()
    assert simulate(config) == simulate(config)
    for values in ({"initial_channel": -1}, {"sensing_s": -1}, {"frequencies_hz": [20e9, 20e9]}):
        with pytest.raises(ValueError):
            AntiJammingConfig(**values)
    with pytest.raises(ValueError):
        InterferenceEnvironment((burst(), burst()))
    with pytest.raises(ValueError):
        PacketNetwork(RFConfig(mode="fixed"), anti_jamming=config)
    with pytest.raises(ValueError):
        GeometrySample(0, 1, 1, interference_w=-1)


def test_antijamming_cli(tmp_path, monkeypatch):
    from fl_space.communication.__main__ import main

    output = tmp_path / "anti.json"
    monkeypatch.setattr(
        "sys.argv", ["communication", "--scenario", "anti-jamming", "--output", str(output)]
    )
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["runs"]["fixed"]["result"]["metrics"]["unique_delivered_bytes"] == 0
    assert report["runs"]["observed"]["result"]["metrics"]["unique_delivered_bytes"] == 48000
