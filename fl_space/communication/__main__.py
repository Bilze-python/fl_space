"""本地运行：python -m fl_space.communication --output output/communication.json。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from fl_space.environment import CelestialBody
from fl_space.orbit import KeplerOrbit, OrbitalElements

from . import (
    DynamicCommunication,
    Message,
    RFConfig,
    evaluate_link,
    orbit_position_provider,
    sample_geometry,
)


def main() -> None:
    """运行同一条 Kepler 星地轨迹的固定/动态 RF 对照并导出完整 JSON。"""
    parser = argparse.ArgumentParser(description="SpaceFL 动态 RF 单链路对照；所有工程参数为假设")
    parser.add_argument("--output", type=Path, default=Path("output/communication.json"))
    parser.add_argument(
        "--scenario", choices=("dynamic", "contention", "anti-jamming"), default="dynamic"
    )
    args = parser.parse_args()
    if args.scenario == "anti-jamming":
        from .antijamming_demo import run_antijamming_demo

        run_antijamming_demo(args.output)
        return
    if args.scenario == "contention":
        from .contention_demo import run_contention_demo

        run_contention_demo(args.output)
        return
    orbit = KeplerOrbit(
        OrbitalElements(semi_major_axis_km=6928.137, inclination_deg=0), CelestialBody.earth()
    )
    samples = sample_geometry(
        orbit_position_provider(orbit),
        lambda _time: (6378137.0, 0.0, 0.0),
        list(range(0, 1201, 5)),
        link_id="satellite-ground",
    )
    message = Message("model-update-0", 20_000_000, deadline_s=1200)
    config = RFConfig()
    output = {
        "schema_version": 1,
        "fidelity": {
            "orbit": "kepler",
            "channel": "first_order",
            "transfer": "single_message_fluid",
        },
        "seed": None,
        "source": "assumption",
        "output_path": str(args.output.resolve()),
        "geometry_step_s": 5,
        "geometry_sampling": "left_endpoint_zero_order_hold",
        "orbit_elements": orbit.elements.to_dict(),
        "ground_ecef_m": [6378137.0, 0.0, 0.0],
        "message": message.to_dict(),
        "runs": {},
    }
    for mode in ("fixed", "dynamic"):
        current = replace(config, mode=mode)
        result = DynamicCommunication(current).transfer(message, samples)
        output["runs"][mode] = {
            "config": current.to_dict(),
            "links": [evaluate_link(sample, current).to_dict() for sample in samples],
            "result": result.to_dict(),
        }
        print(
            f"{mode}: {result.status}, sent={result.transmitted_bits / 8:.0f} bytes, "
            f"arrival_s={result.arrival_s}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"结果已写入 {args.output.resolve()}")


if __name__ == "__main__":
    main()
