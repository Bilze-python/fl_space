"""同轨迹、同丢包种子的共享单信道调度对照。"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from fl_space.environment import CelestialBody
from fl_space.orbit import KeplerOrbit, OrbitalElements

from .geometry import orbit_position_provider, sample_geometry
from .models import Message, RFConfig
from .packet_network import FlowMessage, PacketNetwork, PacketNetworkConfig


def run_contention_demo(output_path: Path) -> dict:
    """三条消息竞争唯一 RF 资源；可复现外生丢包，不声称抗干扰效果。"""
    orbit = KeplerOrbit(
        OrbitalElements(semi_major_axis_km=6928.137, inclination_deg=0), CelestialBody.earth()
    )
    samples = sample_geometry(
        orbit_position_provider(orbit),
        lambda _: (6378137, 0, 0),
        list(range(121)),
        link_id="shared-ground-link",
    )
    messages = [
        FlowMessage(
            f"flow-{index}",
            f"sat-{index}",
            "ground",
            Message(f"update-{index}", 120_000, created_s=index, deadline_s=120),
        )
        for index in range(3)
    ]
    geometry = {item.flow_id: samples for item in messages}
    rf = RFConfig(acquisition_s=0.002, switching_s=0.001)
    config = PacketNetworkConfig(
        slot_s=0.1,
        packet_loss_probability=0.15,
        ack_loss_probability=0.05,
        ack_timeout_s=0.3,
        max_attempts=5,
    )
    report = {
        "schema_version": 1,
        "scenario": "shared_channel_contention",
        "fidelity": {
            "orbit": "kepler",
            "channel": "first_order",
            "protocol": "slotted_stop_and_wait_arq",
        },
        "source": "assumption",
        "seed": config.seed,
        "output_path": str(output_path.resolve()),
        "rf_config": rf.to_dict(),
        "orbit_elements": orbit.elements.to_dict(),
        "geometry_assumption": "three logical flows use the same contact trace for scheduling ablation",
        "geometry": {
            flow: [sample.to_dict() for sample in values] for flow, values in geometry.items()
        },
        "messages": [item.to_dict() for item in messages],
        "runs": {},
    }
    for policy in ("tdma_rr", "drr"):
        current = replace(config, policy=policy)
        result = PacketNetwork(rf, current).run(messages, geometry, 120)
        report["runs"][policy] = {"config": current.to_dict(), "result": result.to_dict()}
        print(
            f"{policy}: delivered={result.metrics['unique_delivered_bytes']} bytes, "
            f"attempts={result.metrics['packet_attempts']}, "
            f"goodput={result.metrics['goodput_bps']:.0f} bit/s"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"结果已写入 {output_path.resolve()}")
    return report
