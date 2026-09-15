"""
配置加载器 — 从文件或预设加载配置

支持: JSON / YAML / TOML / Python 字典
"""

import json
from pathlib import Path
from typing import Any, Union

from fl_space.environment import CelestialBody
from fl_space.environment.ground_station import GroundStation, GroundStationNetwork
from fl_space.orbit.satellite_phases import ConstellationConfig


def load_celestial_body(config: dict[str, Any]) -> CelestialBody:
    """从字典加载天体配置。"""
    name = config.get("name", "Custom")
    if name.lower() in ("earth", "mars", "moon"):
        return {
            "earth": CelestialBody.earth,
            "mars": CelestialBody.mars,
            "moon": CelestialBody.moon,
        }[name.lower()]()
    return CelestialBody.from_dict(config)


def load_ground_stations(config: list) -> GroundStationNetwork:
    """从列表加载地面站网络。"""
    stations = []
    for item in config:
        if isinstance(item, dict):
            stations.append(GroundStation.from_dict(item))
        elif isinstance(item, (list, tuple)):
            name, lat, lon = item[0], item[1], item[2]
            extra = {}
            if len(item) > 3:
                extra = item[3] if isinstance(item[3], dict) else {}
            stations.append(GroundStation(name=name, lat_deg=lat, lon_deg=lon, **extra))
    return GroundStationNetwork(stations)


def load_constellation_config(config: dict[str, Any]) -> ConstellationConfig:
    """从字典加载星座配置。"""
    return ConstellationConfig(**{
        k: v for k, v in config.items()
        if k in ConstellationConfig.__dataclass_fields__
    })


def load_sim_config_from_dict(config: dict[str, Any]) -> dict[str, Any]:
    """
    从配置字典中提取模拟器参数。

    Returns
    -------
    dict
        可直接解包传给 OrbitSimulator 的参数。
    """
    params = {}

    if "body" in config:
        params["body"] = load_celestial_body(config["body"])

    if "constellation" in config:
        params["constellation_config"] = load_constellation_config(config["constellation"])
    else:
        for k in ("num_satellites", "orbit_altitude_km", "orbit_inclination_deg", "distribution"):
            if k in config:
                params[k] = config[k]

    if "ground_stations" in config:
        params["ground_station_network"] = load_ground_stations(config["ground_stations"])
    elif "num_ground_stations" in config:
        params["num_ground_stations"] = config["num_ground_stations"]

    for k in ("timeslot_duration_min", "num_timeslots", "contact_mode",
              "random_seed", "verbose", "use_extended_gs"):
        if k in config:
            params[k] = config[k]

    return params


def load_sim_config_from_json(filepath: Union[str, Path]) -> dict[str, Any]:
    """从 JSON 文件加载模拟器配置。"""
    with open(filepath, encoding='utf-8') as f:
        config = json.load(f)
    return load_sim_config_from_dict(config)
