"""同一轨迹和外部干扰下，固定/随机/观测避扰对照。"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from fl_space.environment import CelestialBody
from fl_space.orbit import KeplerOrbit, OrbitalElements

from .geometry import orbit_position_provider, sample_geometry
from .interference import AntiJammingConfig, InterferenceBurst, InterferenceEnvironment
from .models import Message, RFConfig
from .packet_network import FlowMessage, PacketNetwork, PacketNetworkConfig


def run_antijamming_demo(path: Path) -> dict:
    """强窄带干扰压力场景；不预设算法收益。"""
    orbit = KeplerOrbit(
        OrbitalElements(semi_major_axis_km=6928.137, inclination_deg=0), CelestialBody.earth()
    )
    samples = sample_geometry(
        orbit_position_provider(orbit),
        lambda _: (6378137, 0, 0),
        list(range(101)),
        link_id="sat-ground",
    )
    messages = [FlowMessage("uplink", "sat", "ground", Message("update", 48000, deadline_s=100))]
    bursts = (InterferenceBurst("jammer-0", "ground", 0, 100, 20e9, 10e6, 1e-8),)
    environment = InterferenceEnvironment(bursts)
    rf = RFConfig(acquisition_s=0.002)
    protocol = PacketNetworkConfig(slot_s=1, ack_timeout_s=0.2)
    defense = AntiJammingConfig(observation_delay_s=1)
    report = {
        "schema_version": 1,
        "scenario": "anti_jamming",
        "source": "assumption",
        "fidelity": {
            "orbit": "kepler",
            "channel": "first_order",
            "interference": "receiver_power_flat_spectrum",
            "protocol": "slotted_stop_and_wait_arq",
        },
        "seed": defense.seed,
        "output_path": str(path.resolve()),
        "orbit_elements": orbit.elements.to_dict(),
        "ground_ecef_m": [6378137, 0, 0],
        "rf_config": rf.to_dict(),
        "protocol": protocol.to_dict(),
        "messages": [item.to_dict() for item in messages],
        "geometry": [sample.to_dict() for sample in samples],
        "interference": [burst.to_dict() for burst in bursts],
        "runs": {},
    }
    for policy in ("fixed", "random", "observed"):
        anti = replace(defense, policy=policy)
        result = PacketNetwork(rf, protocol, anti, environment).run(
            messages, {"uplink": samples}, 100
        )
        report["runs"][policy] = {"anti_jamming": anti.to_dict(), "result": result.to_dict()}
        print(
            f"{policy}: delivered={result.metrics['unique_delivered_bytes']} bytes, "
            f"arrival_s={result.messages['update']['arrival_s']}, "
            f"switches={result.metrics['channel_switches']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"结果已写入 {path.resolve()}")
    return report
