"""
SpaceFL CesiumJS 3D 可视化服务器（FastAPI）。

将 OrbitSimulator 的轨道数据以 JSON 格式提供给 CesiumJS 前端，
支持 REST API 和静态文件服务。

用法::

    # 从命令行启动
    python web/server.py --sim-hours 24 --sats 10 --gs 5 --port 8700

    # 或以编程方式
    from web.server import create_app
    app = create_app(sim_kwargs={...})

启动后访问: http://127.0.0.1:8700
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    raise ImportError("Web 服务器需要 fastapi 和 uvicorn。请运行: pip install fastapi uvicorn")

# ── 目录 ──
WEB_DIR = Path(__file__).parent


def build_orbit_data(
    sim_hours: float = 24.0,
    sats: int = 5,
    gs: int = 3,
    altitude_km: float = 500.0,
    inclination_deg: float = 53.0,
    timeslot_min: float = 1.0,
    isl_enabled: bool = False,
    isl_buffer: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """从 OrbitSimulator 构建前端所需的 JSON 数据。

    Returns
    -------
    dict
        包含 satellites, ground_stations, timeslots 的字典。
    """
    if sim_hours <= 0:
        raise ValueError("sim_hours 必须大于 0")
    if timeslot_min <= 0:
        raise ValueError("timeslot_min 必须大于 0")
    if sats < 1 or gs < 1:
        raise ValueError("sats 和 gs 必须至少为 1")
    if altitude_km <= 0:
        raise ValueError("altitude_km 必须大于 0")
    if not 0 <= inclination_deg <= 180:
        raise ValueError("inclination_deg 必须在 0 到 180 之间")

    from fl_space.isl.base import ISLConfig
    from fl_space.simulator import OrbitSimulator

    n_slots = int(sim_hours * 60 / timeslot_min)

    isl_cfg = ISLConfig(
        enabled=isl_enabled,
        calculator="wgs84",
        atmosphere_buffer_km=isl_buffer,
        step_seconds=timeslot_min * 60,
        cluster_mode="plane",
    )

    sim = OrbitSimulator(
        num_satellites=sats,
        num_ground_stations=gs,
        orbit_altitude_km=altitude_km,
        orbit_inclination_deg=inclination_deg,
        timeslot_duration_min=timeslot_min,
        num_timeslots=n_slots,
        isl_config=isl_cfg,
        random_seed=seed,
        verbose=False,
    )

    # 地面站
    gs_list = []
    for i, station in enumerate(sim.ground_network):
        gs_list.append(
            {
                "id": i,
                "name": station.name,
                "lat": station.lat_deg,
                "lon": station.lon_deg,
            }
        )

    # 预计算 ISL（如果启用）
    if isl_enabled:
        sim.compute_isl()

    # 构建时隙数据
    base_dt = datetime(*sim.sim_start_date, tzinfo=timezone.utc)
    timeslots = []
    for ts in range(n_slots):
        # 卫星位置
        positions = []
        for sid in range(sats):
            lat, lon = sim.get_sat_position(sid, ts)
            positions.append(
                {
                    "sat_id": sid,
                    "lat": lat,
                    "lon": lon,
                    "alt_km": sim.orbit_altitude_km,
                }
            )

        # 接触链路
        contacts = []
        for sid in range(sats):
            gs_ids = sim.get_all_contacts(sid, ts)
            for gid in gs_ids:
                contacts.append({"sat_id": sid, "gs_id": gid})

        # ISL 链路
        isl_links = []
        if isl_enabled:
            active = sim.isl_active_at(ts)
            for w in active:
                a_id = int(w.satellite_a.split("-")[1]) if "-" in w.satellite_a else 0
                b_id = int(w.satellite_b.split("-")[1]) if "-" in w.satellite_b else 0
                isl_links.append({"a_id": a_id, "b_id": b_id})

        time_label = (base_dt + timedelta(minutes=ts * timeslot_min)).isoformat()
        timeslots.append(
            {
                "ts": ts,
                "time": time_label,
                "positions": positions,
                "contacts": contacts,
                "isl_links": isl_links,
            }
        )

    return {
        "satellites": sats,
        "ground_stations": gs_list,
        "isl_enabled": isl_enabled,
        "timeslot_duration_min": timeslot_min,
        "sim_hours": sim_hours,
        "timeslots": timeslots,
    }


def create_app(**sim_kwargs: Any) -> FastAPI:
    """创建 FastAPI 应用。

    Parameters
    ----------
    **sim_kwargs
        传递给 build_orbit_data() 的参数。
    """
    app = FastAPI(title="SpaceFL 3D Visualization", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 静态文件
    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = WEB_DIR / "index.html"
        return html_path.read_text(encoding="utf-8")

    @app.get("/api/orbit_data")
    async def orbit_data(
        sim_hours: float = Query(default=24.0),
        sats: int = Query(default=5),
        gs: int = Query(default=3),
        altitude_km: float = Query(default=500.0),
        inclination_deg: float = Query(default=53.0),
        timeslot_min: float = Query(default=1.0),
        isl_enabled: bool = Query(default=False),
        isl_buffer: float = Query(default=0.0),
        seed: int = Query(default=42),
    ):
        """获取轨道数据 JSON。"""
        kwargs = {
            "sim_hours": sim_hours,
            "sats": sats,
            "gs": gs,
            "altitude_km": altitude_km,
            "inclination_deg": inclination_deg,
            "timeslot_min": timeslot_min,
            "isl_enabled": isl_enabled,
            "isl_buffer": isl_buffer,
            "seed": seed,
        }
        # 合并用户传入的 sim_kwargs（命令行参数优先级更高）
        kwargs.update(sim_kwargs)
        data = build_orbit_data(**kwargs)
        return data

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "SpaceFL 3D Visualization"}

    return app


# ── CLI 入口 ──


def main():
    import argparse

    p = argparse.ArgumentParser(description="SpaceFL CesiumJS 3D 可视化服务器")
    p.add_argument("--host", default="0.0.0.0", help="监听地址")
    p.add_argument("--port", type=int, default=8700, help="端口")
    p.add_argument("--sats", type=int, default=5, help="卫星数量")
    p.add_argument("--gs", type=int, default=3, help="地面站数量")
    p.add_argument("--sim-hours", type=float, default=24.0, help="模拟时长(h)")
    p.add_argument("--altitude", type=float, default=500.0, help="轨道高度(km)")
    p.add_argument("--inclination", type=float, default=53.0, help="轨道倾角(°)")
    p.add_argument("--timeslot-min", type=float, default=1.0, help="时隙粒度(min)")
    p.add_argument("--isl", choices=["disabled", "wgs84"], default="disabled")
    p.add_argument("--isl-buffer", type=float, default=0.0, help="ISL 大气余量(km)")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    app = create_app(
        sim_hours=args.sim_hours,
        sats=args.sats,
        gs=args.gs,
        altitude_km=args.altitude,
        inclination_deg=args.inclination,
        timeslot_min=args.timeslot_min,
        isl_enabled=(args.isl == "wgs84"),
        isl_buffer=args.isl_buffer,
        seed=args.seed,
    )

    print("\n  SpaceFL 3D 可视化服务器")
    print(f"  {'=' * 40}")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  卫星: {args.sats} | GS: {args.gs} | ISL: {args.isl}")
    print(f"  模拟: {args.sim_hours}h @ {args.timeslot_min}min/时隙")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
