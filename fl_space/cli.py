"""
fls CLI — SpaceFL 三层指令架构

    fls tune <参数> <值>     调参指令 — 管理超参数
    fls mount <组件> <值>    挂载指令 — 选择算法/模块
    fls run <实验类型>       运行指令 — 执行实验/模拟/导出

所有 tune/mount 的修改持久化到 .fls_session.json，run 命令消费当前 session。

设计原则：
    - 零额外依赖核心功能（仅 argparse 标准库）
    - tune/mount 无 -- 前缀，纯空格分隔
    - run 子命令支持 -- 可选覆盖 session 值
    - Tab 补全通过 argcomplete 或手动脚本
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# ── Session 持久化 ──────────────────────────────────────────────

SESSION_FILE = ".fls_session.json"

DEFAULT_SESSION: dict[str, dict[str, Any]] = {
    "tune": {
        "lr": 0.01,
        "rounds": 300,
        "epochs": 5,
        "batch_size": 32,
        "mu": 0.01,
        "seed": 42,
        "dataset": "mnist",
        "scale": "small",
        "early_stop": 0.90,
        "workers": 1,
        "data_workers": 0,
        "non_iid": True,
        "alpha": 0.5,
        "classes_per_client": 2,
        "max_samples": 1000,
        "partition_strategy": "probability",
        "class_probability": 0.8,
        "preference_mode": "class_balanced",
        "preferred_clients_per_class": 1,
        "sample_cap_strategy": "preserve",
        "data_dir": "./data",
        "device": "cpu",
        "buffer_size": 5,
    },
    "mount": {
        "algo": "fedavg",
        "isl": "disabled",
        "isl_buffer": 0.0,
        "isl_step": 60.0,
        "time_model": "slot",
        "time_model_args": None,
        "backend": "kepler",
        "body": "earth",
        "distribution": "uniform",
        "staleness": False,
        "sats": 5,
        "stations": 3,
        "sim_hours": 3.0,
        "timeslot_min": 1.0,
        "altitude": 500.0,
        "inclination": 53.0,
        "config": None,
    },
}


def load_session() -> dict[str, dict[str, Any]]:
    """加载 session 文件。"""
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, encoding="utf-8") as f:
                data = json.load(f)
            # 用默认值补全缺失字段
            for section in ("tune", "mount"):
                if section not in data:
                    data[section] = {}
                for k, v in DEFAULT_SESSION[section].items():
                    if k not in data[section]:
                        data[section][k] = v
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return _deep_copy_default()


def _deep_copy_default() -> dict[str, dict[str, Any]]:
    return {
        "tune": dict(DEFAULT_SESSION["tune"]),
        "mount": dict(DEFAULT_SESSION["mount"]),
    }


def save_session(session: dict) -> None:
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


def _get_session_or_default(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """读取 session，同时支持 --no-session 跳过。"""
    if getattr(args, "no_session", False):
        return _deep_copy_default()
    return load_session()


# ── 环境检查 ─────────────────────────────────────────────────────


def _check_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _check_skyfield() -> bool:
    try:
        import skyfield  # noqa: F401
        return True
    except ImportError:
        return False


# ── 通用辅助 ─────────────────────────────────────────────────────


def _args_to_dict(args: argparse.Namespace, *keys: str) -> dict:
    return {k: v for k in keys if (v := getattr(args, k, None)) is not None}


def _merge_config(json_path: str | None, cli_args: dict, defaults: dict) -> dict:
    """优先级：CLI 显式 > JSON > 默认值。"""
    config = dict(defaults)
    if json_path:
        try:
            with open(json_path, encoding="utf-8") as f:
                config.update(json.load(f))
            print(f"  [配置] 加载 {json_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  [警告] 配置加载失败: {e}，使用默认值")
    config.update({k: v for k, v in cli_args.items() if v is not None})
    return config


# ══════════════════════════════════════════════════════════════════
#  tune 指令 — 调参
# ══════════════════════════════════════════════════════════════════


def _tune_set(args: argparse.Namespace, key: str, coerce: type = float) -> int:
    s = load_session()
    try:
        val = coerce(args.value)
    except (ValueError, TypeError):
        print(f"错误: '{args.value}' 不是有效的 {coerce.__name__} 值")
        return 1
    s["tune"][key] = val
    save_session(s)
    print(f"  [tune] {key} = {val}")
    return 0


def _tune_bool_toggle(args: argparse.Namespace, key: str) -> int:
    s = load_session()
    val = args.value.lower()
    if val in ("on", "true", "1", "yes"):
        s["tune"][key] = True
    elif val in ("off", "false", "0", "no"):
        s["tune"][key] = False
    else:
        print(f"错误: '{args.value}' 不是有效的布尔值 (on/off)")
        return 1
    save_session(s)
    print(f"  [tune] {key} = {s['tune'][key]}")
    return 0


def cmd_tune_lr(args: argparse.Namespace) -> int:
    return _tune_set(args, "lr")


def cmd_tune_rounds(args: argparse.Namespace) -> int:
    return _tune_set(args, "rounds", int)


def cmd_tune_epochs(args: argparse.Namespace) -> int:
    return _tune_set(args, "epochs", int)


def cmd_tune_batch(args: argparse.Namespace) -> int:
    return _tune_set(args, "batch_size", int)


def cmd_tune_mu(args: argparse.Namespace) -> int:
    return _tune_set(args, "mu")


def cmd_tune_seed(args: argparse.Namespace) -> int:
    return _tune_set(args, "seed", int)


def cmd_tune_dataset(args: argparse.Namespace) -> int:
    val = args.value.lower()
    if val not in ("mnist", "fashion_mnist", "cifar10"):
        print(f"错误: 未知数据集 '{val}'，可选: mnist, fashion_mnist, cifar10")
        return 1
    s = load_session()
    s["tune"]["dataset"] = val
    save_session(s)
    print(f"  [tune] dataset = {val}")
    return 0


def cmd_tune_scale(args: argparse.Namespace) -> int:
    val = args.value.lower()
    if val not in ("small", "medium", "large"):
        print(f"错误: 未知规模 '{val}'，可选: small, medium, large")
        return 1
    s = load_session()
    s["tune"]["scale"] = val
    save_session(s)
    print(f"  [tune] scale = {val}")
    return 0


def cmd_tune_early_stop(args: argparse.Namespace) -> int:
    return _tune_set(args, "early_stop")


def cmd_tune_workers(args: argparse.Namespace) -> int:
    return _tune_set(args, "workers", int)


def cmd_tune_data_workers(args: argparse.Namespace) -> int:
    return _tune_set(args, "data_workers", int)


def cmd_tune_non_iid(args: argparse.Namespace) -> int:
    return _tune_bool_toggle(args, "non_iid")


def cmd_tune_alpha(args: argparse.Namespace) -> int:
    return _tune_set(args, "alpha")


def cmd_tune_device(args: argparse.Namespace) -> int:
    val = args.value.lower()
    if val not in ("cpu", "cuda"):
        print(f"错误: 未知设备 '{val}'，可选: cpu, cuda")
        return 1
    s = load_session()
    s["tune"]["device"] = val
    save_session(s)
    print(f"  [tune] device = {val}")
    return 0


def cmd_tune_buffer_size(args: argparse.Namespace) -> int:
    return _tune_set(args, "buffer_size", int)


def cmd_tune_classes_per_client(args: argparse.Namespace) -> int:
    return _tune_set(args, "classes_per_client", int)


def cmd_tune_max_samples(args: argparse.Namespace) -> int:
    return _tune_set(args, "max_samples", int)


def cmd_tune_partition_strategy(args: argparse.Namespace) -> int:
    val = args.value.lower()
    if val not in ("iid", "dirichlet", "shard", "probability"):
        print("??: partition_strategy ?? iid, dirichlet, shard, probability")
        return 1
    s = load_session()
    s["tune"]["partition_strategy"] = val
    save_session(s)
    print(f"  [tune] partition_strategy = {val}")
    return 0


def cmd_tune_class_probability(args: argparse.Namespace) -> int:
    return _tune_set(args, "class_probability")


def cmd_tune_data_dir(args: argparse.Namespace) -> int:
    s = load_session()
    s["tune"]["data_dir"] = args.value
    save_session(s)
    print(f"  [tune] data_dir = {args.value}")
    return 0


def cmd_tune_preference_mode(args: argparse.Namespace) -> int:
    val = args.value.lower()
    if val not in ("client_window", "class_balanced"):
        print("??: preference_mode ?? client_window, class_balanced")
        return 1
    s = load_session()
    s["tune"]["preference_mode"] = val
    save_session(s)
    print(f"  [tune] preference_mode = {val}")
    return 0


def cmd_tune_preferred_clients_per_class(args: argparse.Namespace) -> int:
    return _tune_set(args, "preferred_clients_per_class", int)


def cmd_tune_sample_cap_strategy(args: argparse.Namespace) -> int:
    val = args.value.lower()
    if val not in ("preserve", "balanced"):
        print("??: sample_cap_strategy ?? preserve, balanced")
        return 1
    s = load_session()
    s["tune"]["sample_cap_strategy"] = val
    save_session(s)
    print(f"  [tune] sample_cap_strategy = {val}")
    return 0


def cmd_tune_show(args: argparse.Namespace) -> int:
    """显示当前 tune 参数。"""
    s = load_session()
    t = s["tune"]
    print("\n  === 当前调参 (tune) ===")
    print(f"    {'学习率 (lr)':22s} = {t['lr']}")
    print(f"    {'训练轮次 (rounds)':22s} = {t['rounds']}")
    print(f"    {'本地epoch (epochs)':22s} = {t['epochs']}")
    print(f"    {'batch size':22s} = {t['batch_size']}")
    print(f"    {'FedProx μ (mu)':22s} = {t['mu']}")
    print(f"    {'FedBuff K (buffer_size)':22s} = {t['buffer_size']}")
    print(f"    {'随机种子 (seed)':22s} = {t['seed']}")
    print(f"    {'数据集 (dataset)':22s} = {t['dataset']}")
    print(f"    {'实验规模 (scale)':22s} = {t['scale']}")
    print(f"    {'早停阈值 (early_stop)':22s} = {t['early_stop']}")
    print(f"    {'训练线程 (workers)':22s} = {t['workers']}")
    print(f"    {'数据加载进程':22s} = {t['data_workers']}")
    print(f"    {'non-IID':22s} = {t['non_iid']}")
    print(f"    {'Dirichlet α (alpha)':22s} = {t['alpha']}")
    print(f"    {'每客户端类别数':22s} = {t['classes_per_client']}")
    print(f"    {'每客户端样本上限':22s} = {t['max_samples']}")
    print(f"    {'计算设备 (device)':22s} = {t['device']}")
    print()
    return 0


def cmd_tune_reset(args: argparse.Namespace) -> int:
    s = load_session()
    s["tune"] = dict(DEFAULT_SESSION["tune"])
    save_session(s)
    print("  [tune] 已重置为默认值")
    return cmd_tune_show(args)


# ══════════════════════════════════════════════════════════════════
#  mount 指令 — 挂载组件
# ══════════════════════════════════════════════════════════════════


def _mount_set(args: argparse.Namespace, key: str, choices: list[str] | None = None) -> int:
    s = load_session()
    val = args.value.lower()
    if choices and val not in choices:
        print(f"错误: '{val}' 不在可选值中: {', '.join(choices)}")
        return 1
    s["mount"][key] = val
    save_session(s)
    print(f"  [mount] {key} = {val}")
    return 0


def cmd_mount_algo(args: argparse.Namespace) -> int:
    return _mount_set(args, "algo", ["fedavg", "fedprox", "fedbuff"])


def cmd_mount_isl(args: argparse.Namespace) -> int:
    return _mount_set(args, "isl", ["disabled", "wgs84"])


def cmd_mount_isl_buffer(args: argparse.Namespace) -> int:
    s = load_session()
    s["mount"]["isl_buffer"] = float(args.value)
    save_session(s)
    print(f"  [mount] isl_buffer = {s['mount']['isl_buffer']} km")
    return 0


def cmd_mount_isl_step(args: argparse.Namespace) -> int:
    s = load_session()
    s["mount"]["isl_step"] = float(args.value)
    save_session(s)
    print(f"  [mount] isl_step = {s['mount']['isl_step']} s")
    return 0


def cmd_mount_time_model(args: argparse.Namespace) -> int:
    return _mount_set(args, "time_model", ["slot", "physics"])


def cmd_mount_time_model_args(args: argparse.Namespace) -> int:
    s = load_session()
    try:
        s["mount"]["time_model_args"] = json.loads(args.value)
    except json.JSONDecodeError:
        s["mount"]["time_model_args"] = args.value
    save_session(s)
    print(f"  [mount] time_model_args = {s['mount']['time_model_args']}")
    return 0


def cmd_mount_backend(args: argparse.Namespace) -> int:
    return _mount_set(args, "backend", ["kepler", "skyfield"])


def cmd_mount_body(args: argparse.Namespace) -> int:
    return _mount_set(args, "body", ["earth", "mars", "moon", "jupiter", "saturn", "venus"])


def cmd_mount_distribution(args: argparse.Namespace) -> int:
    return _mount_set(args, "distribution", ["uniform", "walker", "cluster"])


def cmd_mount_staleness(args: argparse.Namespace) -> int:
    s = load_session()
    val = args.value.lower()
    s["mount"]["staleness"] = val in ("on", "true", "1", "yes")
    save_session(s)
    print(f"  [mount] staleness = {s['mount']['staleness']}")
    return 0


def cmd_mount_sats(args: argparse.Namespace) -> int:
    s = load_session()
    s["mount"]["sats"] = int(args.value)
    save_session(s)
    print(f"  [mount] sats = {s['mount']['sats']}")
    return 0


def cmd_mount_stations(args: argparse.Namespace) -> int:
    s = load_session()
    s["mount"]["stations"] = int(args.value)
    save_session(s)
    print(f"  [mount] stations = {s['mount']['stations']}")
    return 0


def cmd_mount_sim_hours(args: argparse.Namespace) -> int:
    s = load_session()
    s["mount"]["sim_hours"] = float(args.value)
    save_session(s)
    print(f"  [mount] sim_hours = {s['mount']['sim_hours']} h")
    return 0


def cmd_mount_timeslot_min(args: argparse.Namespace) -> int:
    s = load_session()
    s["mount"]["timeslot_min"] = float(args.value)
    save_session(s)
    print(f"  [mount] timeslot_min = {s['mount']['timeslot_min']} min")
    return 0


def cmd_mount_altitude(args: argparse.Namespace) -> int:
    s = load_session()
    s["mount"]["altitude"] = float(args.value)
    save_session(s)
    print(f"  [mount] altitude = {s['mount']['altitude']} km")
    return 0


def cmd_mount_inclination(args: argparse.Namespace) -> int:
    s = load_session()
    s["mount"]["inclination"] = float(args.value)
    save_session(s)
    print(f"  [mount] inclination = {s['mount']['inclination']} deg")
    return 0


def cmd_mount_config(args: argparse.Namespace) -> int:
    s = load_session()
    path = args.value
    if not os.path.exists(path):
        print(f"错误: 配置文件不存在: {path}")
        return 1
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # 将 JSON 配置合并到 tune 和 mount
        _merge_json_into_session(s, data)
        s["mount"]["config"] = os.path.abspath(path)
        save_session(s)
        print(f"  [mount] config = {os.path.abspath(path)}")
        print("  [提示] JSON 配置已合并到当前 session")
    except json.JSONDecodeError as e:
        print(f"错误: JSON 解析失败: {e}")
        return 1
    return 0


def _merge_json_into_session(session: dict, data: dict) -> None:
    """将 JSON 数据键值映射到 session 的 tune/mount 字段。"""
    # 常见 JSON 字段 → session 映射
    key_map = {
        "learning_rate": ("tune", "lr"),
        "lr": ("tune", "lr"),
        "num_rounds": ("tune", "rounds"),
        "rounds": ("tune", "rounds"),
        "local_epochs": ("tune", "epochs"),
        "epochs": ("tune", "epochs"),
        "batch_size": ("tune", "batch_size"),
        "mu": ("tune", "mu"),
        "seed": ("tune", "seed"),
        "device": ("tune", "device"),
        "num_clients": ("tune", "scale"),
        "algorithm": ("mount", "algo"),
        "algo": ("mount", "algo"),
        "backend": ("mount", "backend"),
        "distribution": ("mount", "distribution"),
        "num_satellites": ("mount", "sats"),
        "num_ground_stations": ("mount", "stations"),
        "orbit_altitude_km": ("mount", "altitude"),
        "orbit_inclination_deg": ("mount", "inclination"),
        "timeslot_duration_min": ("mount", "timeslot_min"),
        "buffer_size": ("tune", "buffer_size"),
        "staleness_weight": ("mount", "staleness"),
        "non_iid": ("tune", "non_iid"),
        "alpha": ("tune", "alpha"),
        "early_stop_acc": ("tune", "early_stop"),
        "classes_per_client": ("tune", "classes_per_client"),
        "max_samples_per_client": ("tune", "max_samples"),
        "partition_strategy": ("tune", "partition_strategy"),
        "class_probability": ("tune", "class_probability"),
        "data_dir": ("tune", "data_dir"),
        "preference_mode": ("tune", "preference_mode"),
        "preferred_clients_per_class": ("tune", "preferred_clients_per_class"),
        "sample_cap_strategy": ("tune", "sample_cap_strategy"),
        "dataset": ("tune", "dataset"),
    }
    for json_key, (section, sess_key) in key_map.items():
        if json_key in data:
            session[section][sess_key] = data[json_key]


def cmd_mount_show(args: argparse.Namespace) -> int:
    s = load_session()
    m = s["mount"]
    print("\n  === 当前挂载 (mount) ===")
    print(f"    {'FL算法 (algo)':26s} = {m['algo']}")
    print(f"    {'ISL计算器 (isl)':26s} = {m['isl']}")
    print(f"    {'ISL大气余量':26s} = {m['isl_buffer']} km")
    print(f"    {'ISL采样步长':26s} = {m['isl_step']} s")
    print(f"    {'时间模型':26s} = {m['time_model']}")
    print(f"    {'时间模型参数':26s} = {m['time_model_args']}")
    print(f"    {'轨道后端 (backend)':26s} = {m['backend']}")
    print(f"    {'中心天体 (body)':26s} = {m['body']}")
    print(f"    {'星座分布':26s} = {m['distribution']}")
    print(f"    {'陈旧度降权':26s} = {m['staleness']}")
    print(f"    {'卫星数 (sats)':26s} = {m['sats']}")
    print(f"    {'地面站 (stations)':26s} = {m['stations']}")
    print(f"    {'模拟时长 (sim_hours)':26s} = {m['sim_hours']} h")
    print(f"    {'时隙粒度 (timeslot_min)':26s} = {m['timeslot_min']} min")
    print(f"    {'轨道高度 (altitude)':26s} = {m['altitude']} km")
    print(f"    {'轨道倾角 (inclination)':26s} = {m['inclination']} deg")
    if m["config"]:
        print(f"    {'外挂配置 (config)':26s} = {m['config']}")
    print()
    return 0


def cmd_mount_clear(args: argparse.Namespace) -> int:
    s = load_session()
    s["mount"] = dict(DEFAULT_SESSION["mount"])
    save_session(s)
    print("  [mount] 已重置为默认值")
    return cmd_mount_show(args)


# ══════════════════════════════════════════════════════════════════
#  run 指令 — 实验运行
# ══════════════════════════════════════════════════════════════════


def _build_sim_config(args: argparse.Namespace, session: dict) -> dict:
    """从 session + CLI 覆盖构建模拟器配置。"""
    m = session["mount"]
    return {
        "num_satellites": getattr(args, "sats", None) or m["sats"],
        "num_ground_stations": getattr(args, "stations", None) or m["stations"],
        "sim_hours": getattr(args, "hours", None) or m["sim_hours"],
        "orbit_altitude_km": getattr(args, "altitude", None) or m["altitude"],
        "orbit_inclination_deg": getattr(args, "inclination", None) or m["inclination"],
        "backend": getattr(args, "backend", None) or m["backend"],
        "distribution": getattr(args, "distribution", None) or m["distribution"],
        "timeslot_duration_min": getattr(args, "timeslot_min", None) or m["timeslot_min"],
        "body": getattr(args, "body", None) or m["body"],
        "isl_enabled": (getattr(args, "isl", None) or m["isl"]) == "wgs84",
        "isl_buffer": getattr(args, "isl_buffer", None) or m["isl_buffer"],
        "isl_step": m["isl_step"],
    }


def cmd_run_simulate(args: argparse.Namespace) -> int:
    """运行轨道接触模拟。"""
    from fl_space.environment import CelestialBody, create_default_network
    from fl_space.isl.base import ISLConfig
    from fl_space.orbit import create_circular_orbit
    from fl_space.simulator import OrbitSimulator

    session = _get_session_or_default(args)
    sc = _build_sim_config(args, session)

    n_sats = int(sc["num_satellites"])
    n_gs = int(sc["num_ground_stations"])
    num_slots = int(sc["sim_hours"] * 60 / sc["timeslot_duration_min"])

    # 天体
    body_map = {
        "earth": CelestialBody.earth,
        "mars": CelestialBody.mars,
        "moon": CelestialBody.moon,
        "jupiter": CelestialBody.jupiter,
        "saturn": CelestialBody.saturn,
        "venus": CelestialBody.venus,
    }
    body = body_map.get(sc["body"], CelestialBody.earth)()

    gs_network = create_default_network(n_gs)

    orbits = []
    for i in range(n_sats):
        raan = (360.0 / n_sats) * i if sc["distribution"] == "uniform" else i * 72.0
        orb = create_circular_orbit(
            altitude_km=sc["orbit_altitude_km"],
            inclination_deg=sc["orbit_inclination_deg"],
            raan_deg=raan,
            true_anomaly_deg=i * (360.0 / n_sats),
            body=body,
        )
        orbits.append(orb)

    isl_cfg = ISLConfig(
        enabled=sc["isl_enabled"],
        calculator="wgs84" if sc["isl_enabled"] else "disabled",
        atmosphere_buffer_km=sc["isl_buffer"],
        step_seconds=sc["isl_step"],
        cluster_mode="plane",
    )

    quiet = getattr(args, "quiet", False)
    if not quiet:
        print("=== 轨道接触模拟 ===")
        print(f"  卫星数: {n_sats}  地面站: {n_gs}  时长: {sc['sim_hours']} h")
        print(f"  后端: {sc['backend']}  高度: {sc['orbit_altitude_km']} km")
        print(f"  ISL: {'wgs84' if sc['isl_enabled'] else 'disabled'}")
        print()

    sim = OrbitSimulator(
        body=body,
        orbits=orbits,
        ground_station_network=gs_network,
        num_timeslots=num_slots,
        timeslot_duration_min=sc["timeslot_duration_min"],
        backend=sc["backend"],
        distribution=sc["distribution"],
        isl_config=isl_cfg,
        verbose=not quiet,
    )

    if not quiet:
        sim.summary()

    # 导出 JSON
    if args.output:
        output_data = {
            "config": sc,
            "stats": sim.stats,
            "contact_rate": sim.stats.get("contact_rate", 0),
        }
        if sc["isl_enabled"]:
            output_data["isl_stats"] = sim.isl_stats
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        if not quiet:
            print(f"  结果已导出至: {args.output}")

    return 0


def cmd_run_train(args: argparse.Namespace) -> int:
    """运行 FL 训练实验。"""
    if not _check_torch():
        print("错误: FL 训练需要 PyTorch。请运行: pip install fl-space[full]")
        return 1

    from fl_space.fl.runner import FLRunner

    session = _get_session_or_default(args)
    t = session["tune"]
    m = session["mount"]

    algo = m["algo"]
    device = t["device"]

    # CLI 覆盖
    overrides: dict[str, Any] = {}
    if getattr(args, "rounds", None) is not None:
        overrides["num_rounds"] = args.rounds
    if getattr(args, "epochs", None) is not None:
        overrides["local_epochs"] = args.epochs
    if getattr(args, "lr", None) is not None:
        overrides["learning_rate"] = args.lr
    if getattr(args, "batch_size", None) is not None:
        overrides["batch_size"] = args.batch_size
    if getattr(args, "mu", None) is not None:
        overrides["mu"] = args.mu
    if getattr(args, "buffer_size", None) is not None:
        overrides["buffer_size"] = args.buffer_size
    if getattr(args, "device", None) is not None:
        overrides["device"] = args.device
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    if getattr(args, "partition_strategy", None) is not None:
        overrides["partition_strategy"] = args.partition_strategy
        t["partition_strategy"] = args.partition_strategy
    if getattr(args, "class_probability", None) is not None:
        overrides["class_probability"] = args.class_probability
        t["class_probability"] = args.class_probability
    if getattr(args, "data_dir", None) is not None:
        overrides["data_dir"] = args.data_dir
        t["data_dir"] = args.data_dir
    if getattr(args, "preference_mode", None) is not None:
        overrides["preference_mode"] = args.preference_mode
        t["preference_mode"] = args.preference_mode
    if getattr(args, "preferred_clients_per_class", None) is not None:
        overrides["preferred_clients_per_class"] = args.preferred_clients_per_class
        t["preferred_clients_per_class"] = args.preferred_clients_per_class
    if getattr(args, "sample_cap_strategy", None) is not None:
        overrides["sample_cap_strategy"] = args.sample_cap_strategy
        t["sample_cap_strategy"] = args.sample_cap_strategy

    time_model = getattr(args, "time_model", None) or m["time_model"]
    if time_model:
        overrides["time_model"] = time_model

    import contextlib

    tma = getattr(args, "time_model_args", None) or m["time_model_args"]
    if tma:
        if isinstance(tma, str):
            with contextlib.suppress(json.JSONDecodeError):
                overrides["time_model_kwargs"] = json.loads(tma)
        else:
            overrides["time_model_kwargs"] = tma

    # 从 preset 创建
    runner = FLRunner.from_preset(
        algorithm=algo,
        scale=t["scale"],
        dataset=t["dataset"],
        device=device,
        num_rounds=t["rounds"],
        local_epochs=t["epochs"],
        batch_size=t["batch_size"],
        learning_rate=t["lr"],
        mu=t["mu"],
        buffer_size=t["buffer_size"],
        staleness_weight=m["staleness"],
        seed=t["seed"],
        partition_strategy=t["partition_strategy"],
        class_probability=t["class_probability"],
        preference_mode=t["preference_mode"],
        preferred_clients_per_class=t["preferred_clients_per_class"],
        sample_cap_strategy=t["sample_cap_strategy"],
        data_dir=t["data_dir"],
        **overrides,
    )

    quiet = getattr(args, "quiet", False)
    if not quiet:
        print("=== FL 训练实验 ===")
        c = runner.config
        print(f"  算法: {algo}  数据集: {t['dataset']}  规模: {t['scale']}")
        print(f"  轮次: {c.num_rounds}  本地epoch: {c.local_epochs}  lr: {c.learning_rate}")
        print(f"  时间模型: {c.time_model}  设备: {device}")
        if algo == "fedprox":
            print(f"  近端项 μ: {getattr(c, 'mu', t['mu'])}")
        if algo == "fedbuff":
            print(f"  缓冲区K: {getattr(c, 'buffer_size', t['buffer_size'])}  陈旧度: {m['staleness']}")
        print()

    _history = runner.run(
        dataset_name=t["dataset"],
        iid=not t["non_iid"],
        alpha=t["alpha"],
        classes_per_client=t.get("classes_per_client", None),
        max_samples_per_client=t.get("max_samples", 0),
        data_dir=t.get("data_dir", "./data"),
        partition_strategy=t.get("partition_strategy", "probability"),
        class_probability=t.get("class_probability", 0.8),
        preference_mode=t.get("preference_mode", "class_balanced"),
        preferred_clients_per_class=t.get("preferred_clients_per_class", 1),
        sample_cap_strategy=t.get("sample_cap_strategy", "preserve"),
        verbose=not quiet,
    )

    if args.output:
        output_data = {
            "config": {
                "algorithm": algo,
                "dataset": t["dataset"],
                "scale": t["scale"],
                "rounds": t["rounds"],
                "local_epochs": t["epochs"],
                "learning_rate": t["lr"],
                "batch_size": t["batch_size"],
                "iid": not t["non_iid"],
                "alpha": t["alpha"],
                "classes_per_client": t.get("classes_per_client"),
                "max_samples_per_client": t.get("max_samples"),
                "partition_strategy": t.get("partition_strategy"),
                "class_probability": t.get("class_probability"),
                "preference_mode": t.get("preference_mode"),
                "preferred_clients_per_class": t.get("preferred_clients_per_class"),
                "sample_cap_strategy": t.get("sample_cap_strategy"),
                "data_dir": t.get("data_dir"),
            },
            "history": runner.history_dict,
            "client_label_distribution": runner.client_label_distribution,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        if not quiet:
            print(f"  结果已导出至: {args.output}")

    return 0


def cmd_run_experiment(args: argparse.Namespace) -> int:
    """运行 SpaceFL 完整太空实验。"""
    if not _check_torch():
        print("错误: 实验需要 PyTorch。请运行: pip install fl-space[full]")
        return 1

    session = _get_session_or_default(args)
    t = session["tune"]
    m = session["mount"]

    algo = m["algo"]
    if algo == "fedbuff":
        print("警告: experiment 不支持 fedbuff，请用 fedavg 或 fedprox")
        return 1

    import importlib.util

    _os = os  # local alias inside function scope

    if algo == "fedavg":
        spec_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "examples",
            "standard_experiment.py",
        )
        if not _os.path.exists(spec_path):
            print(f"错误: 找不到标准化实验模块: {spec_path}")
            return 1
        spec = importlib.util.spec_from_file_location("standard_experiment", spec_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["standard_experiment"] = module
        spec.loader.exec_module(module)
        run_experiment_grid = module.run_experiment_grid

        import time as _time
        t_start = _time.time()

        gs_list = getattr(args, "gs", None) or [3, 5, 7, 10]
        sats_list = getattr(args, "sats_list", None) or [3, 5, 7, 10]

        run_experiment_grid(
            gs_counts=gs_list,
            sat_counts=sats_list,
            num_rounds=t["rounds"],
            local_epochs=t["epochs"],
            batch_size=t["batch_size"],
            learning_rate=t["lr"],
            early_stop_acc=t["early_stop"],
            altitude_km=m["altitude"],
            inclination_deg=m["inclination"],
            dataset=t["dataset"],
            device=t["device"],
            sim_hours=m["sim_hours"],
            timeslot_duration_min=m["timeslot_min"],
            seed=t["seed"],
            num_train_workers=t["workers"],
            num_workers=t["data_workers"],
            output_dir=args.output or "experiment_output",
            verbose=not getattr(args, "quiet", False),
            isl_enabled=(m["isl"] == "wgs84"),
            isl_calculator="wgs84" if m["isl"] == "wgs84" else "disabled",
            isl_atmosphere_buffer_km=m["isl_buffer"],
            isl_step_seconds=m["isl_step"],
            non_iid=t["non_iid"],
            classes_per_client=t.get("classes_per_client", None),
            max_samples_per_client=t.get("max_samples", 0),
            partition_strategy=t.get("partition_strategy", "probability"),
            class_probability=t.get("class_probability", 0.8),
            preference_mode=t.get("preference_mode", "class_balanced"),
            preferred_clients_per_class=t.get("preferred_clients_per_class", 1),
            sample_cap_strategy=t.get("sample_cap_strategy", "preserve"),
            data_dir=t.get("data_dir", "./data"),
        )
        total_elapsed = _time.time() - t_start
        if not getattr(args, "quiet", False):
            print(f"\n总耗时: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")

    else:  # fedprox
        spec_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "examples",
            "run_spacefl_experiment.py",
        )
        if not _os.path.exists(spec_path):
            print(f"错误: 找不到 FedProx 实验模块: {spec_path}")
            return 1
        spec = importlib.util.spec_from_file_location("spacefl_experiment", spec_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["spacefl_experiment"] = module
        spec.loader.exec_module(module)
        run_experiment_suite = module.run_experiment_suite

        import time as _time
        t_start = _time.time()

        gs_list = getattr(args, "gs", None) or [1, 3, 5]
        n_sats = getattr(args, "sats_single", None) or m["sats"]

        output_dir = args.output or "experiment_output"
        _os.makedirs(output_dir, exist_ok=True)

        altitudes = (
            [350 + i * (800 - 350) / max(n_sats - 1, 1) for i in range(n_sats)]
            if n_sats > 1
            else [500.0]
        )

        run_experiment_suite(
            gs_counts=list(gs_list),
            num_satellites=n_sats,
            num_rounds=t["rounds"],
            altitudes_km=altitudes,
            inclination_deg=m["inclination"],
            dataset=t["dataset"],
            device=t["device"],
            local_epochs=t["epochs"],
            batch_size=t["batch_size"],
            learning_rate=t["lr"],
            mu=t["mu"],
            early_stop_acc=t["early_stop"],
            num_train_workers=t["workers"],
            num_workers=t["data_workers"],
            sim_hours=m["sim_hours"],
            timeslot_duration_min=m["timeslot_min"],
            seed=t["seed"],
            output_dir=output_dir,
            verbose=not getattr(args, "quiet", False),
        )
        total_elapsed = _time.time() - t_start
        if not getattr(args, "quiet", False):
            print(f"\n总耗时: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")

    return 0


def cmd_run_quick_test(args: argparse.Namespace) -> int:
    """运行 FedProxSat 快速测试。"""
    if not _check_torch():
        print("错误: 实验需要 PyTorch。请运行: pip install fl-space[full]")
        return 1

    session = _get_session_or_default(args)
    t = session["tune"]
    m = session["mount"]

    import importlib.util

    _os = os

    spec_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "examples",
        "quick_test.py",
    )
    if not _os.path.exists(spec_path):
        print(f"错误: 找不到 quick_test.py: {spec_path}")
        return 1
    spec = importlib.util.spec_from_file_location("quick_test", spec_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["quick_test"] = module
    spec.loader.exec_module(module)
    run_quick_test = module.run_quick_test

    # 把 session 参数注入 args
    args.mu = getattr(args, "mu", None) or t["mu"]
    args.mu_min = getattr(args, "mu_min", None) or 0.001
    args.mu_max = getattr(args, "mu_max", None) or 1.0
    args.rounds = getattr(args, "rounds", None) or t["rounds"]
    args.epochs = getattr(args, "epochs", None) or t["epochs"]
    args.early_stop = getattr(args, "early_stop", None) or t["early_stop"]
    args.gs = getattr(args, "gs", None) or m["stations"]
    args.quiet = getattr(args, "quiet", False)
    args.output = args.output or "results"

    return run_quick_test(args)


def cmd_run_list(args: argparse.Namespace) -> int:
    """列出可用资源。"""
    resource = args.resource or "presets"

    if resource == "presets":
        from fl_space.fl.config import EXPERIMENT_PRESETS
        print("=== FL 实验预设 ===\n")
        for p in EXPERIMENT_PRESETS:
            print(f"  {p['name']:30s} — {p['description']}")

    elif resource == "models":
        try:
            from fl_space.fl.models import list_models
            print("=== 可用模型 ===\n")
            models = list_models()
            if models:
                for m in models:
                    print(f"  - {m}")
            else:
                print("  (PyTorch 未安装，无可用模型)")
        except ImportError:
            print("  (PyTorch 未安装，无可用模型)")

    elif resource == "satellites":
        from fl_space.orbit.satellite_registry import registry
        print("=== 已注册卫星类型 ===\n")
        types = registry.list_types()
        if types:
            for t in types:
                if isinstance(t, dict):
                    name = t.get("name", "?")
                    desc = t.get("description", "")
                    cat = t.get("category", "")
                    print(f"  {name:20s} [{cat}] — {desc}")
                else:
                    print(f"  - {t}")
        else:
            print("  (无已注册卫星类型)")

    elif resource == "experiments":
        from fl_space.config.defaults import EXPERIMENT_CONFIGS
        print("=== 模拟实验预设 ===\n")
        for cfg in EXPERIMENT_CONFIGS:
            name, sats, gss, desc = cfg
            print(f"  {name:15s} — {sats} 卫星, {gss} 地面站 — {desc}")

    else:
        print(f"未知资源类型: {resource}")
        print("可用类型: presets, models, satellites, experiments")
        return 1

    return 0


def cmd_run_export(args: argparse.Namespace) -> int:
    """导出模拟结果为 JSON。"""
    from fl_space.environment import CelestialBody, create_default_network
    from fl_space.orbit import create_circular_orbit
    from fl_space.simulator import OrbitSimulator

    session = _get_session_or_default(args)
    sc = _build_sim_config(args, session)

    n_sats = int(sc["num_satellites"])
    n_gs = int(sc["num_ground_stations"])
    num_slots = int(sc["sim_hours"] * 60 / sc["timeslot_duration_min"])

    body_map = {
        "earth": CelestialBody.earth, "mars": CelestialBody.mars,
        "moon": CelestialBody.moon, "jupiter": CelestialBody.jupiter,
        "saturn": CelestialBody.saturn, "venus": CelestialBody.venus,
    }
    body = body_map.get(sc["body"], CelestialBody.earth)()
    gs_network = create_default_network(n_gs)

    orbits = []
    for i in range(n_sats):
        raan = (360.0 / n_sats) * i if sc["distribution"] == "uniform" else i * 72.0
        orb = create_circular_orbit(
            altitude_km=sc["orbit_altitude_km"],
            inclination_deg=sc["orbit_inclination_deg"],
            raan_deg=raan,
            true_anomaly_deg=i * (360.0 / n_sats),
            body=body,
        )
        orbits.append(orb)

    sim = OrbitSimulator(
        body=body, orbits=orbits,
        ground_station_network=gs_network,
        num_timeslots=num_slots,
        timeslot_duration_min=sc["timeslot_duration_min"],
        backend=sc["backend"],
        distribution=sc["distribution"],
        verbose=False,
    )

    export_data = {
        "config": sc,
        "contact_rate": sim.stats.get("contact_rate", 0),
        "stats": sim.stats,
    }

    output_path = args.output or "sim_export.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"导出完成: {output_path}")
    print(f"  天体: {sc['body']}  卫星: {n_sats}  接触率: {sim.stats.get('contact_rate', 0):.2%}")

    return 0


def cmd_run_serve(args: argparse.Namespace) -> int:
    """启动 CesiumJS 3D 可视化服务器。"""
    try:
        import uvicorn

        from web.server import create_app
    except ImportError as e:
        print(f"依赖缺失: {e}")
        print("提示: pip install fastapi uvicorn")
        return 1

    session = _get_session_or_default(args)
    m = session["mount"]

    sats = getattr(args, "serve_sats", None) or m["sats"]
    gs = getattr(args, "serve_gs", None) or m["stations"]
    sim_hours = getattr(args, "serve_hours", None) or m["sim_hours"]
    altitude = getattr(args, "serve_altitude", None) or m["altitude"]
    inclination = getattr(args, "serve_inclination", None) or m["inclination"]
    timeslot_min = getattr(args, "serve_timeslot", None) or m["timeslot_min"]
    isl_enabled = (getattr(args, "serve_isl", None) or m["isl"]) == "wgs84"
    isl_buffer = getattr(args, "serve_isl_buffer", None) or m["isl_buffer"]
    seed = getattr(args, "serve_seed", None) or m.get("seed", 42)
    host = args.host or "0.0.0.0"
    port = args.port or 8700

    app = create_app(
        sim_hours=sim_hours, sats=sats, gs=gs,
        altitude_km=altitude, inclination_deg=inclination,
        timeslot_min=timeslot_min,
        isl_enabled=isl_enabled, isl_buffer=isl_buffer, seed=seed,
    )

    print("\n  SpaceFL 3D 可视化服务器")
    print(f"  {'=' * 38}")
    print(f"  地址: http://{host}:{port}")
    print(f"  卫星: {sats} | GS: {gs} | ISL: {'wgs84' if isl_enabled else 'disabled'}")
    print()

    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def cmd_run_show(args: argparse.Namespace) -> int:
    """显示当前 session 全貌。"""
    cmd_tune_show(args)
    cmd_mount_show(args)
    return 0


# ══════════════════════════════════════════════════════════════════
#  顶层指令：info / help / completion
# ══════════════════════════════════════════════════════════════════


def cmd_info(args: argparse.Namespace) -> int:
    """显示系统与环境信息。"""
    import platform as _platform

    from fl_space import __version__

    ok, no = "[OK]", "[--]"

    print("=== SpaceFL 环境信息 ===\n")
    print(f"  框架版本: {__version__}")
    print(f"  Python:   {_platform.python_version()} ({_platform.python_implementation()})")
    print(f"  操作系统: {_platform.system()} {_platform.release()}")
    print(f"  PyTorch:  {ok + ' 可用' if _check_torch() else no + ' 未安装'}")
    print(f"  Skyfield: {ok + ' 可用' if _check_skyfield() else no + ' 未安装'}")
    print("  NumPy:    ", end="")
    try:
        import numpy
        print(f"{ok} {numpy.__version__}")
    except ImportError:
        print(f"{no} 未安装")
    print("  Matplotlib: ", end="")
    try:
        import matplotlib
        print(f"{ok} {matplotlib.__version__}")
    except ImportError:
        print(f"{no} 未安装")
    print("  CUDA:     ", end="")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"{ok + ' ' + torch.version.cuda if cuda_ok else no + ' 不可用 (CPU only)'}")
    except ImportError:
        print(f"{no} PyTorch 未安装")

    if os.path.exists(SESSION_FILE):
        print(f"\n  Session:  {os.path.abspath(SESSION_FILE)} (已存在)")

    return 0


def _print_categorized_help():
    """自定义精美的分类帮助。"""
    print("""
SpaceFL — 太空联邦学习命令行工具
══════════════════════════════════════════════════════════════════

  用法:  fls <类别> <指令> [参数]

┌─────────────────────────────────────────────────────────────────┐
│  调参指令 (tune)  —  管理超参数                                  │
├─────────────────────────────────────────────────────────────────┤
  fls tune lr <float>            学习率               (默认 0.01)
  fls tune rounds <int>          训练轮次             (默认 300)
  fls tune epochs <int>          本地 epoch           (默认 5)
  fls tune batch <int>           batch size           (默认 32)
  fls tune mu <float>            FedProx μ            (默认 0.01)
  fls tune buffer-size <int>     FedBuff 缓冲区K      (默认 5)
  fls tune seed <int>            随机种子             (默认 42)
  fls tune dataset <name>        数据集               (默认 mnist)
  fls tune scale <name>          实验规模             (默认 small)
  fls tune early-stop <float>    早停准确率阈值       (默认 0.90)
  fls tune workers <int>         训练线程数           (默认 1)
  fls tune non-iid <on|off>      non-IID 开关         (默认 off)
  fls tune alpha <float>         Dirichlet α          (默认 0.5)
  fls tune device <cpu|cuda>     计算设备             (默认 cpu)
  fls tune show                  查看当前调参
  fls tune reset                 重置为默认值

┌─────────────────────────────────────────────────────────────────┐
│  挂载指令 (mount) —  选择算法 / 组件                              │
├─────────────────────────────────────────────────────────────────┤
  fls mount algo <name>          FL算法               (默认 fedavg)
                                 fedavg | fedprox | fedbuff
  fls mount isl <name>           ISL计算器             (默认 disabled)
                                 disabled | wgs84
  fls mount isl-buffer <float>   ISL大气余量 km        (默认 0.0)
  fls mount time-model <name>    时间模型              (默认 slot)
                                 slot | physics
  fls mount time-model-args <json> 时间模型参数        (默认 null)
  fls mount backend <name>       轨道后端              (默认 kepler)
                                 kepler | skyfield
  fls mount body <name>          中心天体              (默认 earth)
  fls mount distribution <name>  星座分布              (默认 uniform)
  fls mount staleness <on|off>   FedBuff陈旧度降权     (默认 off)
  fls mount sats <int>           卫星数量              (默认 5)
  fls mount stations <int>       地面站数量            (默认 3)
  fls mount sim-hours <float>    模拟时长 h            (默认 24)
  fls mount timeslot-min <float> 时隙粒度 min          (默认 1.0)
  fls mount altitude <float>     轨道高度 km           (默认 500)
  fls mount inclination <float>  轨道倾角 deg          (默认 53)
  fls mount config <path>        加载JSON配置文件
  fls mount show                 查看当前挂载
  fls mount clear                重置为默认值

┌─────────────────────────────────────────────────────────────────┐
│  运行指令 (run)  —  执行实验 / 模拟                               │
├─────────────────────────────────────────────────────────────────┤
  fls run simulate               运行轨道接触模拟
  fls run train                  运行 FL 训练实验
  fls run experiment             运行 SpaceFL 完整太空实验
  fls run quick-test             FedProxSat 快速测试
  fls run list [资源]            查看内置资源
  fls run export                 导出模拟结果 JSON
  fls run serve                  启动 CesiumJS 3D 可视化
  fls run show                   查看完整 session 状态

┌─────────────────────────────────────────────────────────────────┐
│  通用                                                           │
├─────────────────────────────────────────────────────────────────┤
  fls info                       系统与环境信息
  fls completion install         安装 Tab 补全 (需 argcomplete)

─────────────────────────────────────────────────────────────────
  工作流示例:
    fls tune lr 0.001
    fls tune rounds 500
    fls mount algo fedprox
    fls mount isl wgs84
    fls mount isl-buffer 80
    fls run train --quiet --output result.json

  Session 持久化: 所有 tune/mount 修改保存到 .fls_session.json
  run 子命令支持 --sats, --stations, --hours 等覆盖 session
─────────────────────────────────────────────────────────────────
""")


def cmd_help(args: argparse.Namespace) -> int:
    _print_categorized_help()
    return 0


def cmd_completion(args: argparse.Namespace) -> int:
    """安装 Tab 补全。"""
    action = args.action or "install"
    if action == "install":
        # 尝试 argcomplete
        import importlib.util as _iu
        if _iu.find_spec("argcomplete") is not None:
            print("argcomplete 已安装。")
            print()
            print("=== 启用 Tab 补全 ===")
            print()
            print("PowerShell (推荐):")
            print("  1. 以管理员身份运行 PowerShell")
            print("  2. 执行: Register-ArgumentCompleter -Command fls -ScriptBlock {")
            print('       param($wordToComplete, $commandAst, $cursorPosition)')
            print('       & python -m fl_space.cli _complete $commandAst')
            print("     }")
            print()
            print("  或在 PowerShell 配置文件中添加 "
                  "(~\\Documents\\PowerShell\\Microsoft.PowerShell_profile.ps1)")
            print()
            print("Bash / Zsh:")
            print('  eval "$(register-python-argcomplete fls)"')
            print()
            print("手动补全 (无需额外包):")
            print("  运行 fls completion ps1 生成 PowerShell 补全脚本")
            return 0

        # argcomplete 未安装 → 生成内置脚本
        print("argcomplete 未安装。")
        print("选项1: pip install argcomplete")
        print("选项2: 使用内置 PowerShell 补全脚本")
        print()
        print("=== 安装内置 PowerShell 补全 ===")
        ps1 = _generate_ps1_completion()
        script_path = os.path.join(os.path.dirname(__file__), "..", "fls_completion.ps1")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(ps1)
        print(f"补全脚本已生成: {os.path.abspath(script_path)}")
        print()
        print("在 PowerShell 配置文件中添加:")
        print(f"  . {os.path.abspath(script_path)}")
        print()
        print("或临时启用: . .\\fls_completion.ps1")
        return 0

    elif action == "ps1":
        print(_generate_ps1_completion())
        return 0

    print(f"未知补全操作: {action}")
    return 1


def _generate_ps1_completion() -> str:
    """生成 PowerShell Tab 补全脚本。"""
    return r'''# SpaceFL fls Tab 补全 (PowerShell)
# 用法: . .\fls_completion.ps1

$fls_commands = @(
    # 顶级
    "tune", "mount", "run", "info", "help", "completion",
    # tune 子命令
    "lr", "rounds", "epochs", "batch", "mu", "buffer-size", "seed",
    "dataset", "scale", "early-stop", "workers", "data-workers",
    "non-iid", "alpha", "classes-per-client", "max-samples", "partition-strategy",
    "class-probability", "data-dir", "device",
    "show", "reset",
    # mount 子命令
    "algo", "isl", "isl-buffer", "isl-step",
    "time-model", "time-model-args", "backend", "body",
    "distribution", "staleness", "sats", "stations",
    "sim-hours", "timeslot-min", "altitude", "inclination",
    "config", "clear",
    # run 子命令
    "simulate", "train", "experiment", "quick-test",
    "list", "export", "serve"
)

Register-ArgumentCompleter -CommandName fls -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)

    $tokens = $commandAst.ToString().Split(" ", [StringSplitOptions]::RemoveEmptyEntries)
    $prev = if ($tokens.Count -ge 2) { $tokens[-2] } else { "" }

    # 按上下文给出补全
    switch ($tokens[0]) {
        "fls" {
            if ($tokens.Count -eq 1) {
                @("tune", "mount", "run", "info", "help", "completion") | Where-Object { $_ -like "$wordToComplete*" }
            }
        }
        default { $fls_commands | Where-Object { $_ -like "$wordToComplete*" } }
    }
}
'''


# ══════════════════════════════════════════════════════════════════
#  Parser Builder
# ══════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fls",
        description="SpaceFL — 太空联邦学习研究框架 CLI (三层架构: tune | mount | run)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("--help", "-h", action="store_true", help="显示帮助")

    sub = parser.add_subparsers(dest="category", title="指令类别")

    # ══════════════════════════════════════════════════════════
    #  tune
    # ══════════════════════════════════════════════════════════

    p_tune = sub.add_parser("tune", help="调参指令 — 管理超参数", add_help=False)
    tune_sub = p_tune.add_subparsers(dest="tune_cmd", title="调参项")

    def _add_tune(name: str, hlp: str, fn, **kw):
        p = tune_sub.add_parser(name, help=hlp, add_help=False)
        p.add_argument("value", **kw)
        p.set_defaults(func=fn)
        return p

    _add_tune("lr", "学习率", cmd_tune_lr, type=float, help="学习率值")
    _add_tune("rounds", "训练轮次", cmd_tune_rounds, type=int, help="轮次数")
    _add_tune("epochs", "本地epoch数", cmd_tune_epochs, type=int, help="epoch数")
    _add_tune("batch", "batch size", cmd_tune_batch, type=int, help="batch size")
    _add_tune("mu", "FedProx μ", cmd_tune_mu, type=float, help="近端项系数")
    _add_tune("buffer-size", "FedBuff缓冲区K", cmd_tune_buffer_size, type=int, help="缓冲区大小")
    _add_tune("seed", "随机种子", cmd_tune_seed, type=int, help="种子值")
    _add_tune("dataset", "数据集", cmd_tune_dataset, type=str, help="mnist|fashion_mnist|cifar10")
    _add_tune("scale", "实验规模", cmd_tune_scale, type=str, help="small|medium|large")
    _add_tune("early-stop", "早停阈值", cmd_tune_early_stop, type=float, help="准确率阈值")
    _add_tune("workers", "训练线程数", cmd_tune_workers, type=int, help="线程数")
    _add_tune("data-workers", "数据加载进程数", cmd_tune_data_workers, type=int, help="进程数")
    _add_tune("non-iid", "non-IID开关", cmd_tune_non_iid, type=str, help="on|off")
    _add_tune("alpha", "Dirichlet α", cmd_tune_alpha, type=float, help="α值")
    _add_tune("classes-per-client", "每客户端类别数", cmd_tune_classes_per_client, type=int, help="2|3|...")
    _add_tune("max-samples", "每客户端样本上限", cmd_tune_max_samples, type=int, help="样本数, 0=不限")
    _add_tune("partition-strategy", "??????", cmd_tune_partition_strategy, type=str, help="iid|dirichlet|shard|probability")
    _add_tune("class-probability", "??????", cmd_tune_class_probability, type=float, help="0.0~1.0")
    _add_tune("data-dir", "????", cmd_tune_data_dir, type=str, help="?????ImageFolder???")
    _add_tune("preference-mode", "????", cmd_tune_preference_mode, type=str, help="client_window|class_balanced")
    _add_tune("preferred-clients-per-class", "????????", cmd_tune_preferred_clients_per_class, type=int, help="??")
    _add_tune("sample-cap-strategy", "??????", cmd_tune_sample_cap_strategy, type=str, help="preserve|balanced")
    _add_tune("device", "计算设备", cmd_tune_device, type=str, help="cpu|cuda")

    p = tune_sub.add_parser("show", help="查看当前调参", add_help=False)
    p.set_defaults(func=cmd_tune_show)
    p = tune_sub.add_parser("reset", help="重置为默认值", add_help=False)
    p.set_defaults(func=cmd_tune_reset)

    # ══════════════════════════════════════════════════════════
    #  mount
    # ══════════════════════════════════════════════════════════

    p_mount = sub.add_parser("mount", help="挂载指令 — 选择算法/组件", add_help=False)
    mount_sub = p_mount.add_subparsers(dest="mount_cmd", title="挂载项")

    def _add_mount(name: str, hlp: str, fn, **kw):
        p = mount_sub.add_parser(name, help=hlp, add_help=False)
        p.add_argument("value", **kw)
        p.set_defaults(func=fn)
        return p

    _add_mount("algo", "FL算法", cmd_mount_algo, type=str, help="fedavg|fedprox|fedbuff")
    _add_mount("isl", "ISL计算器", cmd_mount_isl, type=str, help="disabled|wgs84")
    _add_mount("isl-buffer", "ISL大气余量km", cmd_mount_isl_buffer, type=float, help="缓冲区km")
    _add_mount("isl-step", "ISL采样步长s", cmd_mount_isl_step, type=float, help="采样步长秒")
    _add_mount("time-model", "时间模型", cmd_mount_time_model, type=str, help="slot|physics")
    _add_mount("time-model-args", "时间模型参数JSON", cmd_mount_time_model_args, type=str, help="JSON字符串")
    _add_mount("backend", "轨道后端", cmd_mount_backend, type=str, help="kepler|skyfield")
    _add_mount("body", "中心天体", cmd_mount_body, type=str, help="earth|mars|moon|jupiter|saturn|venus")
    _add_mount("distribution", "星座分布", cmd_mount_distribution, type=str, help="uniform|walker|cluster")
    _add_mount("staleness", "陈旧度降权", cmd_mount_staleness, type=str, help="on|off")
    _add_mount("sats", "卫星数量", cmd_mount_sats, type=int, help="卫星数")
    _add_mount("stations", "地面站数量", cmd_mount_stations, type=int, help="地面站数")
    _add_mount("sim-hours", "模拟时长h", cmd_mount_sim_hours, type=float, help="小时")
    _add_mount("timeslot-min", "时隙粒度min", cmd_mount_timeslot_min, type=float, help="分钟")
    _add_mount("altitude", "轨道高度km", cmd_mount_altitude, type=float, help="高度km")
    _add_mount("inclination", "轨道倾角deg", cmd_mount_inclination, type=float, help="倾角deg")
    _add_mount("config", "加载JSON配置", cmd_mount_config, type=str, help="JSON文件路径")

    p = mount_sub.add_parser("show", help="查看当前挂载", add_help=False)
    p.set_defaults(func=cmd_mount_show)
    p = mount_sub.add_parser("clear", help="重置为默认值", add_help=False)
    p.set_defaults(func=cmd_mount_clear)

    # ══════════════════════════════════════════════════════════
    #  run
    # ══════════════════════════════════════════════════════════

    p_run = sub.add_parser(
        "run", help="运行指令 — 执行实验/模拟/导出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    run_sub = p_run.add_subparsers(dest="run_cmd", title="实验类型")

    # run simulate
    p_sim = run_sub.add_parser("simulate", help="轨道接触模拟", add_help=False)
    p_sim.add_argument("--sats", type=int, default=None, help="卫星数")
    p_sim.add_argument("--stations", type=int, default=None, help="地面站数")
    p_sim.add_argument("--hours", type=float, default=None, help="模拟时长h")
    p_sim.add_argument("--backend", choices=["kepler", "skyfield"], default=None, help="轨道后端")
    p_sim.add_argument("--altitude", type=float, default=None, help="轨道高度km")
    p_sim.add_argument("--inclination", type=float, default=None, help="轨道倾角deg")
    p_sim.add_argument("--distribution", choices=["uniform", "walker", "cluster"], default=None, help="分布")
    p_sim.add_argument("--timeslot-min", type=float, default=None, help="时隙min")
    p_sim.add_argument("--body", choices=["earth", "mars", "moon", "jupiter", "saturn", "venus"], default=None)
    p_sim.add_argument("--isl", choices=["disabled", "wgs84"], default=None, help="ISL")
    p_sim.add_argument("--isl-buffer", type=float, default=None, help="ISL余量km")
    p_sim.add_argument("--output", "-o", type=str, default=None, help="导出JSON")
    p_sim.add_argument("--quiet", "-q", action="store_true", help="安静模式")
    p_sim.add_argument("--no-session", action="store_true", help="忽略session使用默认值")
    p_sim.set_defaults(func=cmd_run_simulate)

    # run train
    p_tr = run_sub.add_parser("train", help="FL训练实验", add_help=False)
    p_tr.add_argument("--rounds", type=int, default=None, help="覆盖session轮次")
    p_tr.add_argument("--epochs", type=int, default=None, help="覆盖session epoch")
    p_tr.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    p_tr.add_argument("--batch-size", type=int, default=None, help="覆盖batch size")
    p_tr.add_argument("--mu", type=float, default=None, help="覆盖μ")
    p_tr.add_argument("--buffer-size", type=int, default=None, help="覆盖缓冲区K")
    p_tr.add_argument("--device", choices=["cpu", "cuda"], default=None, help="覆盖设备")
    p_tr.add_argument("--seed", type=int, default=None, help="覆盖种子")
    p_tr.add_argument("--time-model", type=str, default=None, help="覆盖时间模型")
    p_tr.add_argument("--time-model-args", type=str, default=None, help="时间模型参数JSON")
    p_tr.add_argument("--partition-strategy", choices=["iid", "dirichlet", "shard", "probability"], default=None, help="????????")
    p_tr.add_argument("--class-probability", type=float, default=None, help="????????")
    p_tr.add_argument("--data-dir", type=str, default=None, help="??????")
    p_tr.add_argument("--preference-mode", choices=["client_window", "class_balanced"], default=None, help="??????")
    p_tr.add_argument("--preferred-clients-per-class", type=int, default=None, help="??????????")
    p_tr.add_argument("--sample-cap-strategy", choices=["preserve", "balanced"], default=None, help="????????")
    p_tr.add_argument("--output", "-o", type=str, default=None, help="导出JSON")
    p_tr.add_argument("--quiet", "-q", action="store_true", help="安静模式")
    p_tr.add_argument("--no-session", action="store_true", help="忽略session使用默认值")
    p_tr.set_defaults(func=cmd_run_train)

    # run experiment
    p_exp = run_sub.add_parser("experiment", help="完整太空实验", add_help=False)
    p_exp.add_argument("--gs", type=int, nargs="+", default=None, help="地面站列表")
    p_exp.add_argument("--sats-list", type=int, nargs="+", default=None, help="卫星数列表(fedavg)")
    p_exp.add_argument("--sats-single", type=int, default=None, help="单卫星数(fedprox)")
    p_exp.add_argument("--output", "-o", type=str, default=None, help="输出目录")
    p_exp.add_argument("--quiet", "-q", action="store_true", help="安静模式")
    p_exp.add_argument("--no-session", action="store_true", help="忽略session")
    p_exp.set_defaults(func=cmd_run_experiment)

    # run quick-test
    p_qt = run_sub.add_parser("quick-test", help="FedProxSat快速测试", add_help=False)
    p_qt.add_argument("--mu", type=float, default=None, help="μ")
    p_qt.add_argument("--mu-min", type=float, default=None, help="μ下限")
    p_qt.add_argument("--mu-max", type=float, default=None, help="μ上限")
    p_qt.add_argument("--oscillation-threshold", type=float, default=0.1, help="震荡阈值")
    p_qt.add_argument("--stability-threshold", type=float, default=0.03, help="稳定阈值")
    p_qt.add_argument("--no-adaptive", action="store_true", help="禁用自适应μ")
    p_qt.add_argument("--rounds", type=int, default=None, help="轮次")
    p_qt.add_argument("--epochs", type=int, default=None, help="epoch")
    p_qt.add_argument("--early-stop", type=float, default=None, help="早停")
    p_qt.add_argument("--gs", type=int, default=None, help="地面站")
    p_qt.add_argument("--output", "-o", type=str, default=None, help="输出目录")
    p_qt.add_argument("--quiet", "-q", action="store_true", help="安静模式")
    p_qt.add_argument("--no-session", action="store_true", help="忽略session")
    p_qt.set_defaults(func=cmd_run_quick_test)

    # run list
    p_ls = run_sub.add_parser("list", help="查看内置资源", add_help=False)
    p_ls.add_argument("resource", nargs="?", default="presets",
                      choices=["presets", "models", "satellites", "experiments"],
                      help="presets|models|satellites|experiments")
    p_ls.set_defaults(func=cmd_run_list)

    # run export
    p_ex = run_sub.add_parser("export", help="导出模拟JSON", add_help=False)
    p_ex.add_argument("--output", "-o", type=str, default=None, help="输出JSON路径")
    p_ex.add_argument("--sats", type=int, default=None, help="卫星数")
    p_ex.add_argument("--stations", type=int, default=None, help="地面站数")
    p_ex.add_argument("--hours", type=float, default=None, help="模拟时长h")
    p_ex.add_argument("--backend", choices=["kepler", "skyfield"], default=None)
    p_ex.add_argument("--altitude", type=float, default=None)
    p_ex.add_argument("--inclination", type=float, default=None)
    p_ex.add_argument("--distribution", choices=["uniform", "walker", "cluster"], default=None)
    p_ex.add_argument("--timeslot-min", type=float, default=None)
    p_ex.add_argument("--body", choices=["earth", "mars", "moon", "jupiter", "saturn", "venus"], default=None)
    p_ex.add_argument("--no-session", action="store_true", help="忽略session")
    p_ex.set_defaults(func=cmd_run_export)

    # run serve
    p_sv = run_sub.add_parser("serve", help="CesiumJS 3D可视化", add_help=False)
    p_sv.add_argument("--host", default="0.0.0.0", help="监听地址")
    p_sv.add_argument("--port", type=int, default=8700, help="端口")
    p_sv.add_argument("--serve-sats", type=int, default=None, help="卫星数")
    p_sv.add_argument("--serve-gs", type=int, default=None, help="地面站数")
    p_sv.add_argument("--serve-hours", type=float, default=None, help="模拟时长h")
    p_sv.add_argument("--serve-altitude", type=float, default=None, help="轨道高度km")
    p_sv.add_argument("--serve-inclination", type=float, default=None, help="倾角deg")
    p_sv.add_argument("--serve-timeslot", type=float, default=None, help="时隙min")
    p_sv.add_argument("--serve-isl", choices=["disabled", "wgs84"], default=None, help="ISL")
    p_sv.add_argument("--serve-isl-buffer", type=float, default=None, help="ISL余量km")
    p_sv.add_argument("--serve-seed", type=int, default=None, help="种子")
    p_sv.add_argument("--no-session", action="store_true", help="忽略session")
    p_sv.set_defaults(func=cmd_run_serve)

    # run show
    p_sh = run_sub.add_parser("show", help="查看完整session状态", add_help=False)
    p_sh.set_defaults(func=cmd_run_show)

    # ══════════════════════════════════════════════════════════
    #  顶层：info / help / completion
    # ══════════════════════════════════════════════════════════

    p_info = sub.add_parser("info", help="系统与环境信息", add_help=False)
    p_info.set_defaults(func=cmd_info)

    p_help = sub.add_parser("help", help="显示分类帮助", add_help=False)
    p_help.set_defaults(func=cmd_help)

    p_comp = sub.add_parser("completion", help="安装Tab补全", add_help=False)
    p_comp_sub = p_comp.add_subparsers(dest="action", title="操作")
    p_ci = p_comp_sub.add_parser("install", help="安装补全", add_help=False)
    p_ci.set_defaults(func=cmd_completion)
    p_cp = p_comp_sub.add_parser("ps1", help="生成PowerShell脚本", add_help=False)
    p_cp.set_defaults(func=cmd_completion)

    return parser


# ══════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()

    if argv is None:
        argv = sys.argv[1:]

    # 无参数或 --help / -h 时显示分类帮助
    if not argv or set(argv) & {"--help", "-h"}:
        _print_categorized_help()
        return 0

    # 处理 argcomplete 内部调用
    if os.environ.get("_ARGCOMPLETE") == "1":
        try:
            import argcomplete
            argcomplete.autocomplete(parser)
        except ImportError:
            pass
        return 0

    # 快速路由：顶级 help 命令
    if argv[0] == "help":
        _print_categorized_help()
        return 0

    # 快速路由：顶级 info 命令（不需要完整子解析）
    if argv[0] == "info":
        return cmd_info(argparse.Namespace())

    args = parser.parse_args(argv)

    # 如果只有 category 没有子命令，显示相应帮助
    if not hasattr(args, "func"):
        if args.category == "tune":
            print("请指定 tune 子命令。可用: lr, rounds, epochs, batch, mu, ...")
            print("运行 'fls help' 查看完整列表。")
        elif args.category == "mount":
            print("请指定 mount 子命令。可用: algo, isl, backend, body, ...")
            print("运行 'fls help' 查看完整列表。")
        elif args.category == "run":
            print("请指定 run 子命令。可用: simulate, train, experiment, quick-test, list, export, serve")
            print("运行 'fls help' 查看完整列表。")
        elif args.category == "completion":
            print("请指定 completion 操作。可用: install, ps1")
        else:
            _print_categorized_help()
        return 0

    try:
        return args.func(args)
    except ImportError as e:
        print(f"依赖缺失: {e}")
        print("提示: 请安装所需依赖。例如: pip install fl-space[full]")
        return 1
    except Exception as e:
        print(f"错误: {e}")
        if "-q" not in argv and "--quiet" not in argv:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
