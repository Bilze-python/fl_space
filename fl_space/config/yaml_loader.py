"""
YAML 配置加载器 — 可选的 YAML 配置格式支持。

对齐师兄项目 autoFly_Stk 的 config/loader.py，但适配 SpaceFL 数据结构。

用法::

    from fl_space.config.yaml_loader import load_yaml_config

    config = load_yaml_config("my_experiment.yaml")
    # config 包含 sim_kwargs, fl_config, gs_list, isl_config
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """从 YAML 文件加载 SpaceFL 实验配置。

    返回包含以下键的字典：
        - sim_kwargs: OrbitSimulator 构造参数
        - fl_config: FLConfig 兼容字典
        - gs_list: 地面站列表 [(name, lat, lon), ...]
        - isl_config: ISL 配置字典
        - raw: 原始 YAML 数据

    Parameters
    ----------
    path : str | Path
        YAML 配置文件路径。

    Returns
    -------
    dict
        解析后的配置字典。

    Raises
    ------
    ImportError
        PyYAML 未安装。
    FileNotFoundError
        配置文件不存在。
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML 未安装。请运行: pip install pyyaml")

    yaml_path = Path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"配置文件为空: {yaml_path}")

    # 解析星座
    constellation = data.get("constellation", {})
    total_sats = constellation.get("num_planes", 1) * constellation.get("sats_per_plane", 3)
    altitude = constellation.get("altitude_km", 500.0)
    inclination = constellation.get("inclination_deg", 53.0)

    # 解析地面站
    gs_data = data.get("ground_stations", [])
    gs_list = []
    for gs in gs_data:
        gs_list.append((gs.get("name", "GS"), gs.get("lat_deg", 0.0), gs.get("lon_deg", 0.0)))

    # 解析 ISL
    isl_data = data.get("intra_cluster", {})
    isl_config = {
        "enabled": isl_data.get("enabled", False),
        "calculator": isl_data.get("calculator", "wgs84"),
        "atmosphere_buffer_km": isl_data.get("atmosphere_buffer_km", 0.0),
        "step_seconds": isl_data.get("step_seconds", 60.0),
        "cluster_mode": isl_data.get("cluster_mode", "plane"),
    }

    # 解析 FL 实验
    fl_data = data.get("fl_experiment", {})
    fl_config = {
        "algorithm": fl_data.get("algorithm", "fedavg"),
        "num_rounds": fl_data.get("num_rounds", 300),
        "num_clients": total_sats,
        "local_epochs": fl_data.get("local_epochs", 2),
        "batch_size": fl_data.get("batch_size", 32),
        "learning_rate": fl_data.get("learning_rate", 0.01),
        "mu": fl_data.get("mu", 0.01),
        "early_stop_acc": fl_data.get("early_stop_acc", 0.90),
        "device": fl_data.get("device", "cpu"),
        "fraction": fl_data.get("fraction", 1.0),
        "isl_enabled": isl_config["enabled"],
        "isl_calculator": isl_config["calculator"],
        "isl_atmosphere_buffer_km": isl_config["atmosphere_buffer_km"],
        "isl_step_seconds": isl_config["step_seconds"],
    }

    # 模拟器参数
    sim_hours = data.get("sim_hours", 168.0)
    ts_min = data.get("timeslot_duration_min", 1.0)
    sim_kwargs = {
        "num_satellites": total_sats,
        "num_ground_stations": len(gs_list),
        "orbit_altitude_km": altitude,
        "orbit_inclination_deg": inclination,
        "distribution": constellation.get("topology", "uniform"),
        "timeslot_duration_min": ts_min,
        "num_timeslots": int(sim_hours * 60 / ts_min),
        "random_seed": data.get("seed", 42),
    }

    return {
        "sim_kwargs": sim_kwargs,
        "fl_config": fl_config,
        "gs_list": gs_list,
        "isl_config": isl_config,
        "raw": data,
    }
