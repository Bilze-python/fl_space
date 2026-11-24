"""分组网络资源守恒、ARQ 因果、去重、公平性与输入边界。"""

from dataclasses import replace
import json

import pytest

from fl_space.communication import (
    FlowMessage,
    GeometrySample,
    Message,
    PacketNetwork,
    PacketNetworkConfig,
    PacketNetworkResult,
    PacketScheduler,
    RFConfig,
)


def run(messages, until=10, **kwargs):
    config = PacketNetworkConfig(
        slot_s=1,
        packet_bytes=100,
        header_bytes=0,
        ack_bytes=1,
        ack_rate_bps=8000,
        ack_timeout_s=0.1,
        **kwargs,
    )
    rf = RFConfig(mode="fixed", fixed_rate_bps=8000, acquisition_s=0, switching_s=0)
    geometry = {item.flow_id: [GeometrySample(0, until, 1)] for item in messages}
    return PacketNetwork(rf, config).run(messages, geometry, until)


def owner(flow="a", size=300, created=0, deadline=None, identity=None):
    return FlowMessage(flow, flow, "ground", Message(identity or flow, size, created, deadline))


def test_shared_capacity_rr_and_unique_delivery():
    result = run([owner("a"), owner("b")])
    attempts = [event for event in result.events if event["kind"] == "packet_attempt"]
    assert [event["flow_id"] for event in attempts] == ["a", "b", "a", "b", "a", "b"]
    assert all(item["status"] == "delivered" for item in result.messages.values())
    assert result.metrics["unique_delivered_bytes"] == 600
    assert result.metrics["transmitted_wire_bits"] == 4800
    assert result.metrics["jain_fairness"] == 1
    tx = [event for event in result.events if event["kind"] == "transmitting"]
    assert all(right["time_s"] >= left["end_s"] for left, right in zip(tx, tx[1:]))
    assert sum(event["bits"] for event in tx) <= 8000 * 10
    assert result.metrics["queue_remaining_bytes"] == {"a": 0, "b": 0}


def test_ack_loss_retries_but_never_duplicate_delivery():
    result = run([owner(size=100)], ack_loss_probability=1, max_attempts=3)
    assert result.messages["a"]["status"] == "delivered"
    assert result.messages["a"]["sender_status"] == "retry_exhausted"
    assert result.metrics["packet_attempts"] == 3
    assert len([event for event in result.events if event["kind"] == "delivered"]) == 1
    assert len([event for event in result.events if event["kind"] == "duplicate_received"]) == 2
    assert result.metrics["unique_delivered_bytes"] == 100


def test_total_data_loss_exhausts_without_delivery():
    result = run([owner(size=100)], packet_loss_probability=1, max_attempts=2)
    assert result.messages["a"]["status"] == "retry_exhausted"
    assert result.metrics["unique_delivered_bytes"] == 0
    assert result.metrics["packet_attempts"] == 2


def test_queue_admission_overflow_deadline_and_future_arrivals():
    result = run(
        [
            owner(size=300),
            owner(size=100, identity="second"),
            owner("b", created=2, deadline=2.05),
            owner("c", created=20),
        ],
        queue_bytes_per_flow=300,
    )
    assert result.messages["second"]["status"] == "overflow"
    assert result.messages["b"]["status"] == "expired"
    assert result.messages["c"]["status"] == "not_created"
    assert max(result.metrics["queue_peak_bytes"].values()) <= 300
    assert all(event["time_s"] >= 2 for event in result.events if event["flow_id"] == "b")


def test_in_flight_until_horizon_is_not_delivered():
    item = owner(size=100)
    config = PacketNetworkConfig(slot_s=1, packet_bytes=100, header_bytes=0)
    rf = RFConfig(mode="fixed", fixed_rate_bps=8000, acquisition_s=0)
    result = PacketNetwork(rf, config).run([item], {"a": [GeometrySample(0, 0.2, 299792458)]}, 0.2)
    assert result.messages["a"]["status"] == "pending"
    assert result.messages["a"]["arrival_s"] is None
    assert not any(event["kind"] == "delivered" for event in result.events)


def test_packet_cannot_resume_across_different_grants():
    item = owner(size=100)
    config = PacketNetworkConfig(
        slot_s=1, packet_bytes=100, header_bytes=0, max_attempts=2, ack_timeout_s=0.1
    )
    rf = RFConfig(mode="fixed", fixed_rate_bps=400, acquisition_s=0)
    result = PacketNetwork(rf, config).run([item], {"a": [GeometrySample(0, 10, 1)]}, 10)
    assert result.messages["a"]["status"] == "retry_exhausted"
    assert result.metrics["transmitted_wire_bits"] == pytest.approx(800)
    assert result.metrics["unique_delivered_bytes"] == 0


def test_no_contact_waits_without_spending_attempt_budget():
    item = owner(size=100)
    result = PacketNetwork().run([item], {"a": [GeometrySample(0, 5, 1e6, False)]}, 5)
    assert result.metrics["packet_attempts"] == 0
    assert result.metrics["transmit_energy_j"] == 0
    assert result.messages["a"]["status"] == "pending"


def test_default_configuration_can_deliver():
    result = PacketNetwork().run([owner(size=100)], {"a": [GeometrySample(0, 5, 550000)]}, 5)
    assert result.messages["a"]["status"] == "delivered"


def test_drr_gives_byte_fair_service_for_unequal_packet_sizes():
    scheduler = PacketScheduler(["small", "large"], "drr", 100)
    served = {"small": 0, "large": 0}
    for _ in range(400):
        flow = scheduler.choose({"small": 100, "large": 300})
        served[flow] += 100 if flow == "small" else 300
    assert served["small"] == served["large"]
    rr = PacketScheduler(["small", "large"], "tdma_rr", 100)
    assert [rr.choose({"small": 100, "large": 300}) for _ in range(4)] == ["small", "large"] * 2


def test_timeout_before_ack_resends_and_late_ack_is_safe():
    config = PacketNetworkConfig(
        slot_s=0.2, packet_bytes=100, header_bytes=0, ack_timeout_s=0.1, max_attempts=10
    )
    rf = RFConfig(mode="fixed", fixed_rate_bps=8000, acquisition_s=0)
    result = PacketNetwork(rf, config).run(
        [owner(size=100)], {"a": [GeometrySample(0, 3, 299792458 * 0.3)]}, 3
    )
    assert result.messages["a"]["sender_status"] == "confirmed"
    assert result.metrics["packet_attempts"] > 1
    assert len([event for event in result.events if event["kind"] == "delivered"]) == 1
    acknowledgments = [event for event in result.events if event["kind"] == "ack_transmitting"]
    assert all(
        right["time_s"] >= left["end_s"]
        for left, right in zip(acknowledgments, acknowledgments[1:])
    )


def test_roundtrip_and_seed_reproducibility():
    item = owner(size=1000)
    assert FlowMessage.from_dict(json.loads(json.dumps(item.to_dict()))) == item
    config = PacketNetworkConfig()
    assert PacketNetworkConfig.from_dict(config.to_dict()) == config
    result = run([item], packet_loss_probability=0.3)
    assert result == run([item], packet_loss_probability=0.3)
    assert (
        PacketNetworkResult.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
        == result
    )


@pytest.mark.parametrize(
    "change",
    [
        {"slot_s": 0},
        {"packet_bytes": 0},
        {"ack_timeout_s": -1},
        {"seed": 1.5},
        {"packet_loss_probability": 2},
        {"queue_bytes_per_flow": True},
        {"header_bytes": -1},
    ],
)
def test_invalid_config(change):
    with pytest.raises(ValueError):
        PacketNetworkConfig(**change)


def test_invalid_network():
    with pytest.raises(ValueError):
        PacketNetwork().run([owner(), owner()], {"a": []}, 1)
    with pytest.raises(ValueError):
        PacketNetwork().run([owner()], {}, 1)
    with pytest.raises(ValueError):
        PacketNetwork().run(
            [owner(), replace(owner(identity="other"), sender_id="other")], {"a": []}, 1
        )


def test_outage_inside_grant_requires_whole_packet_retry():
    rf = RFConfig(mode="fixed", fixed_rate_bps=800, acquisition_s=0, switching_s=0)
    config = PacketNetworkConfig(slot_s=2, packet_bytes=100, header_bytes=0, ack_timeout_s=0.1)
    trace = [GeometrySample(0, 0.5, 1), GeometrySample(0.5, 1, 1, False), GeometrySample(1, 6, 1)]
    result = PacketNetwork(rf, config).run([owner(size=100)], {"a": trace}, 6)
    assert result.metrics["packet_attempts"] == 2
    assert result.messages["a"]["arrival_s"] > 3
    assert result.metrics["unique_delivered_bytes"] == 100


def test_partial_message_is_not_eligible_for_aggregation():
    result = run([owner(size=200)], until=0.5)
    assert result.messages["a"]["received_payload_bytes"] == 100
    assert result.messages["a"]["delivered_bytes"] == 0
    assert result.messages["a"]["status"] == "pending"


def test_contention_cli(tmp_path, monkeypatch):
    from fl_space.communication.__main__ import main

    output = tmp_path / "network.json"
    monkeypatch.setattr(
        "sys.argv", ["communication", "--scenario", "contention", "--output", str(output)]
    )
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["scenario"] == "shared_channel_contention"
    assert set(report["runs"]) == {"tdma_rr", "drr"}
    for run_result in report["runs"].values():
        metrics = run_result["result"]["metrics"]
        assert metrics["unique_delivered_bytes"] == 360000
        assert metrics["data_channels"] == 1
