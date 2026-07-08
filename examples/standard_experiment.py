"""
SpaceFL 标准化实验 — 论文地面站 + FedAvg + 网格搜索 + 全套标准化输出

用法:
    # 完整网格搜索 (GS=[3,5,7,10] × SAT=[3,5,7,10] = 16组)
    python examples/standard_experiment.py

    # 单组实验
    python examples/standard_experiment.py --gs 5 --sats 7

    # CLI 快捷方式
    fls experiment --gs 3 5 7 10 --sats 3 5 7 10 --algo fedavg

标准化输出（每组实验）:
    {output_dir}/gs{GS}_sat{SAT}/
    ├── config.json              — 实验配置
    ├── history.json             — 每轮JSON (timeslot, acc, loss)
    ├── accuracy_trend.png       — 准确率趋势图
    ├── gs_positions.png         — 地面站位置图（含经纬度标注）
    ├── contact_heatmap.png      — 接触热力图
    ├── satellite_training_time.png  — 每卫星训练时长统计
    ├── orbit_cross_section.png  — 轨道剖面图（卫星相对位置 + 地球距离）
    ├── gs_sat_contacts.png      — 地面站-卫星接触次数条形图
    └── summary.json             — 汇总统计
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
import sys
import time as _time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_space.environment import CelestialBody, GroundStation, GroundStationNetwork
from fl_space.viz.i18n import setup_cjk_font, t, tf
from fl_space.fl.fedavg import (
    CappedSelector,
    FixedEpochTrainer,
    StandardEvaluator,
    SyncWeightedAggregator,
)
from fl_space.fl.runner import FLRunner
from fl_space.fl.scheduler import CommunicationScheduler
from fl_space.fl.server import FLConfig
from fl_space.orbit import KeplerOrbit, create_circular_orbit
from fl_space.simulator import OrbitSimulator

# matplotlib
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── 论文 Table 3 地面站 (13站, 按论文顺序) ─────────────────────

PAPER_GS = [
    ("Sioux Falls", 43.55, -96.72),
    ("Sanya", 18.25, 109.5),
    ("Johannesburg", -26.2, 28.03),
    ("Cordoba", -31.4, -64.18),
    ("Tromso", 69.65, 18.95),
    ("Kashi", 39.1, 77.2),
    ("Beijing", 39.9, 116.4),
    ("Neustrelitz", 53.1, 13.1),
    ("Parepare", -2.99, 119.8),
    ("Alice Springs", -25.1, 133.9),
    ("Fairbanks", 64.8, -147.7),
    ("Prince Albert", 53.2, -105.7),
    ("Shadnagar", 17.4, 78.5),
]


def get_paper_gs_network(n: int) -> GroundStationNetwork:
    """从论文 Table 3 取前 n 个地面站。"""
    n = min(n, len(PAPER_GS))
    return GroundStationNetwork(
        [GroundStation(name, lat, lon, 0.05) for name, lat, lon in PAPER_GS[:n]]
    )


# ── 轨道创建 ───────────────────────────────────────────────────


def create_uniform_orbits(
    body: CelestialBody | None = None,
    n_sats: int = 10,
    altitude_km: float = 500.0,
    inclination_deg: float = 53.0,
    raan_deg: float = 0.0,
) -> list[KeplerOrbit]:
    """创建均高卫星（同高度，不同真近点角均匀分布）。"""
    if body is None:
        body = CelestialBody.earth()
    orbits = []
    for i in range(n_sats):
        true_anomaly = i * (360.0 / n_sats)
        orb = create_circular_orbit(
            altitude_km=altitude_km,
            inclination_deg=inclination_deg,
            raan_deg=raan_deg,
            true_anomaly_deg=true_anomaly,
            body=body,
        )
        orbits.append(orb)
    return orbits


# ── 单组实验运行器 ─────────────────────────────────────────────


@dataclass
class SingleExperiment:
    """单组 (GS, SAT) 实验结果。"""

    gs_count: int
    sat_count: int
    gs_names: list[str] = field(default_factory=list)
    gs_coords: list[tuple[float, float]] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    sim: Any = None
    contact_stats: dict = field(default_factory=dict)
    elapsed_sec: float = 0.0
    config: dict = field(default_factory=dict)
    label_distribution: dict = field(default_factory=dict)


def run_single_experiment(
    gs_count: int,
    sat_count: int,
    *,
    num_rounds: int = 300,
    local_epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    early_stop_acc: float = 0.90,
    altitude_km: float = 500.0,
    inclination_deg: float = 53.0,
    dataset: str = "mnist",
    device: str = "cpu",
    sim_hours: float = 3.0,
    timeslot_duration_min: float = 1.0,
    seed: int = 42,
    num_train_workers: int = 1,
    num_workers: int = 0,
    verbose: bool = True,
    # ISL
    isl_enabled: bool = False,
    isl_calculator: str = "wgs84",
    isl_atmosphere_buffer_km: float = 0.0,
    isl_step_seconds: float = 60.0,
    # 数据划分
    non_iid: bool = True,
    classes_per_client: int = 2,
    max_samples_per_client: int = 1000,
    partition_strategy: str = "probability",
    class_probability: float = 0.8,
    preference_mode: str = "class_balanced",
    preferred_clients_per_class: int = 1,
    sample_cap_strategy: str = "preserve",
    data_dir: str = "./data",
    limit_to_sim_window: bool = True,
) -> SingleExperiment:
    """运行单组 SpaceFL 实验。"""
    t0 = _time.time()

    # 轨道：均高卫星
    body = CelestialBody.earth()
    orbits = create_uniform_orbits(body, sat_count, altitude_km, inclination_deg)

    # 地面站
    gs_network = get_paper_gs_network(gs_count)
    gs_names = [gs.name for gs in gs_network]
    gs_coords = [(float(gs.lat_deg), float(gs.lon_deg)) for gs in gs_network]

    num_timeslots = int(sim_hours * 60 / timeslot_duration_min)

    # 模拟器
    from fl_space.isl.base import ISLConfig

    isl_cfg = ISLConfig(
        enabled=isl_enabled,
        calculator=isl_calculator,
        atmosphere_buffer_km=isl_atmosphere_buffer_km,
        step_seconds=isl_step_seconds,
        cluster_mode="plane",
    )

    sim = OrbitSimulator(
        body=body,
        orbits=orbits,
        ground_station_network=gs_network,
        num_timeslots=num_timeslots,
        timeslot_duration_min=timeslot_duration_min,
        backend="kepler",
        contact_mode="simple",
        isl_config=isl_cfg,
        verbose=False,
    )

    if verbose:
        print(sim.summary())

    # FL 配置 — FedAvg
    max_selected = min(gs_count, sat_count)
    fl_config = FLConfig(
        algorithm="fedavg",
        num_rounds=num_rounds,
        num_clients=sat_count,
        fraction=1.0,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mu=0.0,
        device=device,
        seed=seed,
        time_model="slot",
        time_model_kwargs={"slots_per_epoch": 1},
        early_stop_acc=early_stop_acc,
        num_train_workers=num_train_workers,
        num_workers=num_workers,
        partition_strategy=partition_strategy,
        class_probability=class_probability,
        preference_mode=preference_mode,
        preferred_clients_per_class=preferred_clients_per_class,
        sample_cap_strategy=sample_cap_strategy,
        data_dir=data_dir,
        limit_to_sim_window=limit_to_sim_window,
    )

    scheduler = CommunicationScheduler(sim)
    selector = CappedSelector(max_count=max_selected, min_clients=1, seed=seed)
    trainer = FixedEpochTrainer(
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    aggregator = SyncWeightedAggregator(min_updates=1)
    evaluator = StandardEvaluator(device=device)

    runner = FLRunner(fl_config, selector, trainer, aggregator, evaluator, scheduler=scheduler)

    if verbose:
        print(f"  GS={gs_count}, SAT={sat_count}, 每轮最多选择 {max_selected} 客户端")

    history = runner.run(
        dataset_name=dataset,
        iid=not non_iid,
        alpha=0.5,
        classes_per_client=classes_per_client,
        max_samples_per_client=max_samples_per_client,
        data_dir=data_dir,
        partition_strategy=partition_strategy,
        class_probability=class_probability,
        preference_mode=preference_mode,
        preferred_clients_per_class=preferred_clients_per_class,
        sample_cap_strategy=sample_cap_strategy,
        verbose=verbose,
    )

    # 转换为字典
    history_dict = []
    for h in history:
        entry = {
            "round": h.round_num,
            "timeslot_start": getattr(h, "timeslot_start", None),
            "timeslot_end": getattr(h, "timeslot_end", None),
            "accuracy": h.eval_metrics.get("accuracy", 0),
            "loss": h.eval_metrics.get("loss", 0),
            "online_clients": getattr(h, "num_online", None),
            "selected_clients": getattr(h, "num_selected", None),
            "train_loss": getattr(h, "train_loss", None),
        }
        history_dict.append(entry)

    elapsed = _time.time() - t0

    # 接触统计
    contact_stats = _compute_contact_stats(sim)

    return SingleExperiment(
        gs_count=gs_count,
        sat_count=sat_count,
        gs_names=gs_names,
        gs_coords=gs_coords,
        history=history_dict,
        sim=sim,
        contact_stats=contact_stats,
        elapsed_sec=elapsed,
        label_distribution=runner.client_label_distribution,
        config={
            "gs_count": gs_count,
            "sat_count": sat_count,
            "gs_names": gs_names,
            "altitude_km": altitude_km,
            "inclination_deg": inclination_deg,
            "max_selected": max_selected,
            "algorithm": "fedavg",
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "early_stop_acc": early_stop_acc,
            "dataset": dataset,
            "device": device,
            "sim_hours": sim_hours,
            "timeslot_duration_min": timeslot_duration_min,
            "seed": seed,
            "completed_rounds": len(history_dict),
            "elapsed_sec": elapsed,
            "non_iid": non_iid,
            "classes_per_client": classes_per_client,
            "max_samples_per_client": max_samples_per_client,
            "partition_strategy": partition_strategy,
            "class_probability": class_probability,
            "preference_mode": preference_mode,
            "preferred_clients_per_class": preferred_clients_per_class,
            "sample_cap_strategy": sample_cap_strategy,
            "data_dir": data_dir,
            "limit_to_sim_window": limit_to_sim_window,
        },
    )


# ── 接触统计 ───────────────────────────────────────────────────


def _compute_contact_stats(sim: OrbitSimulator) -> dict:
    """计算每个地面站与每个卫星的接触次数统计。"""
    cm = sim.contact_matrix
    gs_network = sim.ground_network
    n_sats = sim.num_satellites
    n_slots = sim.num_timeslots
    n_gs = sim.num_ground_stations

    mat = cm.simple_matrix  # (N_sats, N_slots), values = gs_id or -1

    per_gs = {}
    for gs_id in range(n_gs):
        gs_contacts = []
        for sat_id in range(n_sats):
            row = mat[sat_id]
            in_contact = row == gs_id
            padded = np.concatenate([[False], in_contact, [False]])
            changes = np.diff(padded.astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            for s, e in zip(starts, ends):
                gs_contacts.append(
                    {
                        "sat_id": int(sat_id),
                        "start_ts": int(s),
                        "end_ts": int(e),
                        "duration_slots": int(e - s),
                    }
                )

        durations = [c["duration_slots"] for c in gs_contacts]
        contacted_sats = sorted({c["sat_id"] for c in gs_contacts})
        per_gs[gs_id] = {
            "gs_name": gs_network.names[gs_id] if gs_id < len(gs_network.names) else f"GS-{gs_id}",
            "lat": float(gs_network[gs_id].lat_deg),
            "lon": float(gs_network[gs_id].lon_deg),
            "total_contacts": len(gs_contacts),
            "contacted_satellites": contacted_sats,
            "num_contacted_sats": len(contacted_sats),
            "avg_duration_slots": float(np.mean(durations)) if durations else 0.0,
            "max_duration_slots": int(max(durations)) if durations else 0,
            "total_duration_slots": int(sum(durations)) if durations else 0,
            "per_sat_contacts": {
                sat_id: sum(1 for c in gs_contacts if c["sat_id"] == sat_id)
                for sat_id in range(n_sats)
            },
        }

    per_satellite = {}
    for sat_id in range(n_sats):
        row = mat[sat_id]
        contact_slots = int(np.sum(row >= 0))
        contacted_gs = sorted({int(row[ts]) for ts in range(n_slots) if row[ts] >= 0})
        in_contact = row >= 0
        padded = np.concatenate([[False], in_contact, [False]])
        changes = np.diff(padded.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        durations = [int(e - s) for s, e in zip(starts, ends)]
        per_satellite[sat_id] = {
            "total_contact_slots": contact_slots,
            "contact_rate": float(contact_slots / n_slots),
            "contacted_gs": contacted_gs,
            "num_windows": len(durations),
            "avg_window_slots": float(np.mean(durations)) if durations else 0.0,
            "max_window_slots": int(max(durations)) if durations else 0,
        }

    return {
        "per_gs": per_gs,
        "per_satellite": per_satellite,
        "global": {
            "num_satellites": n_sats,
            "num_ground_stations": n_gs,
            "num_timeslots": n_slots,
            "timeslot_duration_min": sim.timeslot_duration_min,
            "contact_rate": sim.stats.get("contact_rate", 0),
            "total_contacts": sim.stats.get("total_contacts", 0),
        },
    }


# ── 标准化输出生成 ─────────────────────────────────────────────


def generate_standard_outputs(exp: SingleExperiment, output_dir: str, quiet: bool = False, lang: str = "en") -> str:
    """为一个实验生成全套标准化输出文件。"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. config.json
    _save_json(os.path.join(output_dir, "config.json"), exp.config)

    # 2. history.json
    _save_json(os.path.join(output_dir, "history.json"), exp.history)

    # 2b. client_label_distribution.json
    _save_json(
        os.path.join(output_dir, "client_label_distribution.json"),
        exp.label_distribution,
    )

    # 3. summary.json
    if exp.history:
        accs = [h["accuracy"] for h in exp.history]
        summary = {
            "gs_count": exp.gs_count,
            "sat_count": exp.sat_count,
            "gs_names": exp.gs_names,
            "total_rounds": len(exp.history),
            "final_accuracy": round(accs[-1], 4),
            "max_accuracy": round(max(accs), 4),
            "min_accuracy": round(min(accs), 4),
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std_accuracy": round(float(np.std(accs)), 4),
            "total_timeslots": exp.history[-1].get(
                "timeslot_end", exp.history[-1].get("timeslot_start", 0)
            ),
            "elapsed_sec": exp.elapsed_sec,
            "contact_rate": exp.contact_stats.get("global", {}).get("contact_rate", 0),
            "contact_stats": exp.contact_stats["global"],
            "avg_label_entropy": exp.label_distribution.get("avg_entropy", 0),
        }
    else:
        summary = {"error": "no training history"}
    _save_json(os.path.join(output_dir, "summary.json"), summary)

    if not HAS_MPL:
        if not quiet:
            print("  [警告] matplotlib 未安装，跳过图表生成")
        return output_dir

    if lang == "zh":
        setup_cjk_font()

    # 4. accuracy_trend.png
    _plot_accuracy_trend(exp, os.path.join(output_dir, "accuracy_trend.png"), lang=lang)

    # 5. gs_positions.png
    _plot_gs_positions(exp, os.path.join(output_dir, "gs_positions.png"), lang=lang)

    # 6. contact_heatmap.png
    _plot_contact_heatmap_std(exp, os.path.join(output_dir, "contact_heatmap.png"), lang=lang)

    # 7. satellite_training_time.png
    _plot_satellite_training_time(exp, os.path.join(output_dir, "satellite_training_time.png"), lang=lang)

    # 8. orbit_cross_section.png
    _plot_orbit_cross_section(exp, os.path.join(output_dir, "orbit_cross_section.png"), lang=lang)

    # 9. gs_sat_contacts.png
    _plot_gs_sat_contacts(exp, os.path.join(output_dir, "gs_sat_contacts.png"), lang=lang)

    if not quiet:
        print(f"  [输出] {output_dir}/")

    return output_dir


def _save_json(path: str, data: Any) -> None:
    """保存 JSON 文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


# ── 图表生成函数 ───────────────────────────────────────────────

COLORS = [
    "#3498db",
    "#e74c3c",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#34495e",
    "#d35400",
    "#7f8c8d",
]


def _plot_accuracy_trend(exp: SingleExperiment, path: str, lang: str = "en") -> None:
    """准确率趋势图（按轮次 + 时间槽双面板）。"""
    history = exp.history
    if not history:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    rounds = [h["round"] for h in history]
    accs = [h["accuracy"] for h in history]

    ax1.plot(rounds, accs, "b-o", markersize=2, linewidth=0.8)
    ax1.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5, label=t("90%", lang))
    ax1.set_xlabel(t("Round", lang))
    ax1.set_ylabel(t("Accuracy", lang))
    ax1.set_title(tf("Accuracy vs Rounds (GS={gs}, SAT={sat})", lang,
                     gs=exp.gs_count, sat=exp.sat_count))
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    max_acc = max(accs)
    max_r = rounds[accs.index(max_acc)]
    ax1.annotate(
        f"{t('max=', lang)}{max_acc:.3f} @ R{max_r}",
        xy=(max_r, max_acc),
        fontsize=8,
        color="red",
        fontweight="bold",
    )

    ts = [h.get("timeslot_start", h.get("round", i) * 10) for i, h in enumerate(history)]
    ax2.plot(ts, accs, "g-o", markersize=2, linewidth=0.8)
    ax2.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel(t("Timeslot", lang))
    ax2.set_ylabel(t("Accuracy", lang))
    ax2.set_title(t("Accuracy vs Virtual Time", lang))
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle(
        tf("SpaceFL FedAvg — GS={gs}, SAT={sat}, Final Acc={acc:.3f}", lang,
           gs=exp.gs_count, sat=exp.sat_count, acc=accs[-1]),
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_gs_positions(exp: SingleExperiment, path: str, lang: str = "en") -> None:
    """地面站位置图（全球地图 + 经纬度标注）。"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # 绘制简化的海岸线参考
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.2)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.2)

    for i, (name, (lat, lon)) in enumerate(zip(exp.gs_names, exp.gs_coords)):
        color = COLORS[i % len(COLORS)]
        ax.scatter(lon, lat, c=color, s=150, zorder=5, edgecolors="black", linewidths=0.8)
        ax.annotate(
            f"  {name}\n  ({lat:.1f}°, {lon:.1f}°)",
            (lon, lat),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=8,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel(t("Longitude (°)", lang))
    ax.set_ylabel(t("Latitude (°)", lang))
    ax.set_title(tf("Ground Station Positions (GS={gs}) — Paper Table 3", lang,
                    gs=exp.gs_count))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(30))
    ax.grid(True, alpha=0.3)

    # 图例：地面站列表
    legend_text = "\n".join(
        f"  {name}: ({lat:.1f}°, {lon:.1f}°)"
        for name, (lat, lon) in zip(exp.gs_names, exp.gs_coords)
    )
    ax.text(
        0.02,
        0.02,
        tf("Ground Stations:\n{gs_text}", lang, gs_text=legend_text),
        transform=ax.transAxes,
        fontsize=7,
        family="monospace",
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_contact_heatmap_std(exp: SingleExperiment, path: str, lang: str = "en") -> None:
    """接触矩阵热力图。"""
    sim = exp.sim
    n_sats = sim.num_satellites
    n_slots = sim.num_timeslots
    gs_network = sim.ground_network

    heatmap = np.zeros((n_sats, n_slots), dtype=np.float32)
    for sat_id in range(n_sats):
        for ts in range(n_slots):
            contacts = sim.contact_matrix.get_all_contacts(sat_id, ts)
            heatmap[sat_id, ts] = len(contacts)

    fig, ax = plt.subplots(figsize=(16, max(5, n_sats * 0.4)))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", origin="lower", interpolation="nearest")

    hours = n_slots * sim.timeslot_duration_min / 60.0
    n_ticks = min(12, n_slots // 100 + 1)
    tick_positions = np.linspace(0, n_slots - 1, n_ticks, dtype=int)
    tick_labels = [f"{ts * sim.timeslot_duration_min / 60:.1f}h" for ts in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.set_yticks(range(n_sats))
    ax.set_yticklabels([f"SAT-{i}" for i in range(n_sats)], fontsize=8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(t("Connected GS", lang), fontsize=9)

    ax.set_xlabel(tf("Time ({hours:.0f}h, {slot_min}min/slot)", lang,
                     hours=hours, slot_min=sim.timeslot_duration_min))
    ax.set_ylabel(t("Satellite ID", lang))
    ax.set_title(
        tf("Contact Matrix — GS={gs}, SAT={sat}, Rate={rate:.1%}", lang,
           gs=exp.gs_count, sat=exp.sat_count,
           rate=sim.stats.get('contact_rate', 0))
    )

    # 右侧标注 GS 名称
    gs_text = "\n".join(f"GS-{i}: {gs_network.names[i]}" for i in range(len(gs_network.names)))
    ax.text(
        1.02,
        0.5,
        gs_text,
        transform=ax.transAxes,
        fontsize=7,
        family="monospace",
        verticalalignment="center",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_satellite_training_time(exp: SingleExperiment, path: str, lang: str = "en") -> None:
    """每卫星训练时长统计条形图。"""
    contact_stats = exp.contact_stats
    per_sat = contact_stats.get("per_satellite", {})
    n_sats = exp.sat_count
    slot_min = exp.config.get("timeslot_duration_min", 1.0)

    sats = list(range(n_sats))
    contact_slots = [per_sat.get(i, {}).get("total_contact_slots", 0) for i in sats]
    contact_minutes = [s * slot_min for s in contact_slots]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左：接触时长
    colors_sat = [COLORS[i % len(COLORS)] for i in range(n_sats)]
    bars1 = ax1.bar(
        [f"SAT-{i}" for i in sats], contact_minutes, color=colors_sat, edgecolor="white"
    )
    for bar, v in zip(bars1, contact_minutes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{v:.0f}",
            ha="center",
            fontsize=7,
        )
    ax1.set_ylabel(t("Total Contact Time (minutes)", lang))
    ax1.set_title(t("Per-Satellite Contact Duration", lang) + f" ({slot_min}min/slot)")
    ax1.grid(axis="y", alpha=0.3)

    # 右：接触率
    contact_rates = [per_sat.get(i, {}).get("contact_rate", 0) * 100 for i in sats]
    bars2 = ax2.bar([f"SAT-{i}" for i in sats], contact_rates, color=colors_sat, edgecolor="white")
    for bar, v in zip(bars2, contact_rates):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{v:.1f}%",
            ha="center",
            fontsize=7,
        )
    ax2.set_ylabel(t("Contact Rate (%)", lang))
    ax2.set_title(t("Per-Satellite Contact Rate", lang) + f" (GS={exp.gs_count})")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        tf("Satellite Connectivity — GS={gs}, {sats} Sats @ {alt}km", lang,
           gs=exp.gs_count, sats=exp.sat_count,
           alt=exp.config.get('altitude_km', 500)),
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_orbit_cross_section(exp: SingleExperiment, path: str, lang: str = "en") -> None:
    """轨道剖面图：卫星相对位置 + 地球距离。"""
    sat_count = exp.sat_count
    altitude_km = exp.config.get("altitude_km", 500.0)
    inclination = exp.config.get("inclination_deg", 53.0)

    earth_radius = 6371.0  # km
    orbit_radius = earth_radius + altitude_km

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": None})

    # 不使用 polar projection, 直接绘制圆形
    ax.set_aspect("equal")

    # 地球
    earth = plt.Circle(
        (0, 0), earth_radius, color="#2e86c1", alpha=0.3, label=f"Earth (R={earth_radius}km)"
    )
    ax.add_patch(earth)

    # 地球边界
    earth_edge = plt.Circle((0, 0), earth_radius, fill=False, color="#1a5276", linewidth=1.5)
    ax.add_patch(earth_edge)

    # 轨道圆
    orbit_circle = plt.Circle(
        (0, 0), orbit_radius, fill=False, color="#e74c3c", linestyle="--", linewidth=1.2
    )
    ax.add_patch(orbit_circle)

    # 卫星位置（均匀分布在轨道上）
    for i in range(sat_count):
        angle = np.deg2rad(i * (360.0 / sat_count))
        sat_x = orbit_radius * np.cos(angle)
        sat_y = orbit_radius * np.sin(angle)
        ax.plot(
            sat_x,
            sat_y,
            "o",
            color=COLORS[i % len(COLORS)],
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            f"SAT-{i}",
            (sat_x, sat_y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            fontweight="bold",
        )
        # 距离地球表面标注
        ax.plot([0, sat_x], [0, sat_y], color=COLORS[i % len(COLORS)], linewidth=0.5, alpha=0.3)

    # 地面站标注（极地方向）
    gs_text = "\n".join(
        f"  GS-{i}: {name} ({lat:.1f}°, {lon:.1f}°)"
        for i, (name, (lat, lon)) in enumerate(zip(exp.gs_names, exp.gs_coords))
    )
    ax.text(
        0.02,
        0.02,
        tf("Ground Stations:\n{gs_text}", lang, gs_text=gs_text),
        transform=ax.transAxes,
        fontsize=7,
        family="monospace",
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    # 标注信息
    ax.text(
        0.98,
        0.98,
        tf("Orbit: {alt}km alt, {incl}° incl\n{sats} satellites evenly spaced", lang,
           alt=altitude_km, incl=inclination, sats=sat_count),
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    limit = orbit_radius * 1.3
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel(t("X (km)", lang))
    ax.set_ylabel(t("Y (km)", lang))
    ax.set_title(
        tf("Orbit Cross-Section — GS={gs}, SAT={sat} @ {alt}km", lang,
           gs=exp.gs_count, sat=sat_count, alt=altitude_km),
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_gs_sat_contacts(exp: SingleExperiment, path: str, lang: str = "en") -> None:
    """地面站-卫星接触次数条形图（分组柱状图）。"""
    contact_stats = exp.contact_stats
    per_gs = contact_stats.get("per_gs", {})
    n_gs = exp.gs_count
    n_sats = exp.sat_count

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # 左：每个地面站与每颗卫星的接触次数（分组柱状图）
    x = np.arange(n_sats)
    width = 0.8 / n_gs

    for gs_id in range(n_gs):
        gs_data = per_gs.get(gs_id, {})
        per_sat = gs_data.get("per_sat_contacts", {})
        counts = [per_sat.get(sat_id, 0) for sat_id in range(n_sats)]
        offset = (gs_id - n_gs / 2 + 0.5) * width
        bars = ax1.bar(
            x + offset,
            counts,
            width,
            label=gs_data.get("gs_name", f"GS-{gs_id}"),
            color=COLORS[gs_id % len(COLORS)],
            edgecolor="white",
            alpha=0.85,
        )
        for bar, v in zip(bars, counts):
            if v > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(v),
                    ha="center",
                    fontsize=6,
                    rotation=90,
                )

    ax1.set_xlabel(t("Satellite ID", lang))
    ax1.set_ylabel(t("Contact Window Count", lang))
    ax1.set_title(tf("GS-Satellite Contact Counts (GS={gs}, SAT={sat})", lang,
                     gs=n_gs, sat=n_sats))
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"SAT-{i}" for i in range(n_sats)])
    ax1.legend(fontsize=7, ncol=min(n_gs, 5))
    ax1.grid(axis="y", alpha=0.3)

    # 右：每个地面站总接触次数
    gs_names = [per_gs.get(i, {}).get("gs_name", f"GS-{i}") for i in range(n_gs)]
    total_contacts = [per_gs.get(i, {}).get("total_contacts", 0) for i in range(n_gs)]
    colors_gs = [COLORS[i % len(COLORS)] for i in range(n_gs)]
    bars2 = ax2.barh(gs_names, total_contacts, color=colors_gs, edgecolor="white")
    for bar, v in zip(bars2, total_contacts):
        ax2.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(v),
            va="center",
            fontsize=8,
        )
    ax2.set_xlabel(t("Total Contact Windows", lang))
    ax2.set_title(t("Total Contacts per Ground Station", lang))
    ax2.grid(axis="x", alpha=0.3)
    ax2.invert_yaxis()

    fig.suptitle(
        tf("GS-Satellite Contact Analysis — GS={gs}, SAT={sat}", lang,
           gs=n_gs, sat=n_sats),
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 实验套件运行器 ─────────────────────────────────────────────


def run_experiment_grid(
    gs_counts: list[int] = (3, 5, 7, 10),
    sat_counts: list[int] = (3, 5, 7, 10),
    *,
    num_rounds: int = 300,
    local_epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    early_stop_acc: float = 0.90,
    altitude_km: float = 500.0,
    inclination_deg: float = 53.0,
    dataset: str = "mnist",
    device: str = "cpu",
    sim_hours: float = 3.0,
    timeslot_duration_min: float = 1.0,
    seed: int = 42,
    num_train_workers: int = 1,
    num_workers: int = 0,
    output_dir: str = "experiment_output",
    verbose: bool = True,
    lang: str = "en",
    # ISL 星间链路
    isl_enabled: bool = False,
    isl_calculator: str = "wgs84",
    isl_atmosphere_buffer_km: float = 0.0,
    isl_step_seconds: float = 60.0,
    # 数据划分
    non_iid: bool = True,
    classes_per_client: int = 2,
    max_samples_per_client: int = 1000,
    partition_strategy: str = "probability",
    class_probability: float = 0.8,
    preference_mode: str = "class_balanced",
    preferred_clients_per_class: int = 1,
    sample_cap_strategy: str = "preserve",
    data_dir: str = "./data",
    limit_to_sim_window: bool = True,
) -> list[SingleExperiment]:
    """运行完整的网格搜索实验套件。"""
    os.makedirs(output_dir, exist_ok=True)
    results: list[SingleExperiment] = []

    total = len(gs_counts) * len(sat_counts)
    idx = 0

    if verbose:
        print("=" * 70)
        print("  SpaceFL 标准化实验 — 网格搜索")
        print("=" * 70)
        print(f"  GS: {gs_counts}")
        print(f"  SAT: {sat_counts}")
        print("  算法: FedAvg (同步)")
        print("  选择器: min(GS, SAT)")
        print(f"  最大轮次: {num_rounds}, 早停阈值: {early_stop_acc}")
        print(f"  数据集: {dataset}, 设备: {device}")
        print(f"  输出: {output_dir}/")
        print("=" * 70)

    for gs in gs_counts:
        for sat in sat_counts:
            idx += 1
            if verbose:
                print(f"\n{'─' * 50}")
                print(f"  [{idx}/{total}] GS={gs}, SAT={sat}")
                print(f"{'─' * 50}")

            exp = run_single_experiment(
                gs_count=gs,
                sat_count=sat,
                num_rounds=num_rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                early_stop_acc=early_stop_acc,
                altitude_km=altitude_km,
                inclination_deg=inclination_deg,
                dataset=dataset,
                device=device,
                sim_hours=sim_hours,
                timeslot_duration_min=timeslot_duration_min,
                seed=seed,
                num_train_workers=num_train_workers,
                num_workers=num_workers,
                verbose=verbose,
                isl_enabled=isl_enabled,
                isl_calculator=isl_calculator,
                isl_atmosphere_buffer_km=isl_atmosphere_buffer_km,
                isl_step_seconds=isl_step_seconds,
                non_iid=non_iid,
                classes_per_client=classes_per_client,
                max_samples_per_client=max_samples_per_client,
                partition_strategy=partition_strategy,
                class_probability=class_probability,
                preference_mode=preference_mode,
                preferred_clients_per_class=preferred_clients_per_class,
                sample_cap_strategy=sample_cap_strategy,
                data_dir=data_dir,
                limit_to_sim_window=limit_to_sim_window,
            )

            exp_dir = os.path.join(output_dir, f"gs{gs}_sat{sat}")
            generate_standard_outputs(exp, exp_dir, quiet=not verbose, lang=lang)
            results.append(exp)

            if verbose and exp.history:
                acc = exp.history[-1]["accuracy"]
                print(f"  → 完成: {len(exp.history)}轮, Acc={acc:.4f}, 耗时={exp.elapsed_sec:.1f}s")

    # ── 生成网格汇总 ──
    _generate_grid_summary(results, output_dir, verbose, lang=lang)

    return results


def _generate_grid_summary(
    results: list[SingleExperiment],
    output_dir: str,
    verbose: bool,
    lang: str = "en",
) -> None:
    """生成网格搜索汇总图表。"""
    if not results or not HAS_MPL:
        return

    # 汇总 JSON
    grid_data = []
    for exp in results:
        if exp.history:
            accs = [h["accuracy"] for h in exp.history]
            grid_data.append(
                {
                    "gs_count": exp.gs_count,
                    "sat_count": exp.sat_count,
                    "rounds": len(exp.history),
                    "final_acc": round(accs[-1], 4),
                    "max_acc": round(max(accs), 4),
                    "elapsed_sec": exp.elapsed_sec,
                    "contact_rate": exp.contact_stats.get("global", {}).get("contact_rate", 0),
                }
            )
    _save_json(os.path.join(output_dir, "grid_summary.json"), grid_data)

    # 热力图：准确率矩阵
    _plot_grid_heatmap(results, output_dir, lang=lang)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  实验完成! 输出: {os.path.abspath(output_dir)}/")
        print("  汇总: grid_summary.json")
        print(f"{'=' * 70}")


def _plot_grid_heatmap(results: list[SingleExperiment], output_dir: str, lang: str = "en") -> None:
    """绘制网格搜索准确率热力图。"""
    gs_values = sorted({r.gs_count for r in results})
    sat_values = sorted({r.sat_count for r in results})

    acc_matrix = np.zeros((len(gs_values), len(sat_values)))
    rounds_matrix = np.zeros((len(gs_values), len(sat_values)), dtype=int)

    for exp in results:
        gi = gs_values.index(exp.gs_count)
        si = sat_values.index(exp.sat_count)
        if exp.history:
            accs = [h["accuracy"] for h in exp.history]
            acc_matrix[gi, si] = max(accs)
            rounds_matrix[gi, si] = len(exp.history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # 准确率热力图
    im1 = ax1.imshow(acc_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax1.set_xticks(range(len(sat_values)))
    ax1.set_xticklabels(sat_values)
    ax1.set_yticks(range(len(gs_values)))
    ax1.set_yticklabels(gs_values)
    ax1.set_xlabel(t("Satellites", lang))
    ax1.set_ylabel(t("Ground Stations", lang))
    ax1.set_title(t("Max Accuracy (FedAvg)", lang))
    for i in range(len(gs_values)):
        for j in range(len(sat_values)):
            ax1.text(
                j,
                i,
                f"{acc_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if acc_matrix[i, j] < 0.5 else "black",
            )
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 轮次热力图
    im2 = ax2.imshow(rounds_matrix, cmap="Blues", aspect="auto")
    ax2.set_xticks(range(len(sat_values)))
    ax2.set_xticklabels(sat_values)
    ax2.set_yticks(range(len(gs_values)))
    ax2.set_yticklabels(gs_values)
    ax2.set_xlabel(t("Satellites", lang))
    ax2.set_ylabel(t("Ground Stations", lang))
    ax2.set_title(t("Completed Rounds", lang))
    for i in range(len(gs_values)):
        for j in range(len(sat_values)):
            ax2.text(
                j,
                i,
                str(rounds_matrix[i, j]),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if rounds_matrix[i, j] > 150 else "black",
            )
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle(tf("SpaceFL Grid Search Summary — FedAvg, GS×SAT", lang),
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "grid_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CLI 入口 ──────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    p = argparse.ArgumentParser(
        description="SpaceFL 标准化实验 — 网格搜索 + 全套标准化输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  fls experiment --gs 3 5 7 10 --sats 3 5 7 10
  fls experiment --gs 5 --sats 7
  fls experiment --gs 3 5 --sats 3 5 --dataset cifar10 --device cuda
  python examples/standard_experiment.py --gs 5 --sats 7 -o my_results
        """,
    )
    p.add_argument(
        "--gs", type=int, nargs="+", default=[3, 5, 7, 10], help="地面站数量列表 (默认: 3 5 7 10)"
    )
    p.add_argument(
        "--sats", type=int, nargs="+", default=[3, 5, 7, 10], help="卫星数量列表 (默认: 3 5 7 10)"
    )
    p.add_argument("--rounds", type=int, default=300, help="最大训练轮次 (默认: 300)")
    p.add_argument("--epochs", type=int, default=2, help="本地训练 epoch (默认: 2)")
    p.add_argument("--batch-size", type=int, default=32, help="batch size (默认: 32)")
    p.add_argument("--lr", type=float, default=0.01, help="学习率 (默认: 0.01)")
    p.add_argument("--early-stop", type=float, default=0.90, help="早停准确率阈值 (默认: 0.90)")
    p.add_argument("--altitude", type=float, default=500.0, help="卫星轨道高度 km (默认: 500)")
    p.add_argument("--inclination", type=float, default=53.0, help="轨道倾角° (默认: 53)")
    p.add_argument(
        "--dataset",
        choices=["mnist", "fashion_mnist", "cifar10", "imagefolder", "custom"],
        default="mnist",
        help="数据集 (默认: mnist)",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--sim-hours", type=float, default=3.0, help="模拟时长/小时 (默认: 168 = 7天)")
    p.add_argument("--timeslot-min", type=float, default=1.0, help="每 timeslot 分钟 (默认: 1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-workers", type=int, default=1)
    p.add_argument("--data-workers", type=int, default=0)
    p.add_argument("--output", "-o", type=str, default="experiment_output")
    p.add_argument("--partition-strategy", choices=["iid", "dirichlet", "shard", "probability"], default="probability")
    p.add_argument("--class-probability", type=float, default=0.8)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--lang", choices=["en", "zh"], default="en",
                   help="输出图表语言 (默认: en)")
    p.add_argument("--quiet", "-q", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    t_start = _time.time()

    run_experiment_grid(
        gs_counts=args.gs,
        sat_counts=args.sats,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stop_acc=args.early_stop,
        altitude_km=args.altitude,
        inclination_deg=args.inclination,
        dataset=args.dataset,
        device=args.device,
        sim_hours=args.sim_hours,
        timeslot_duration_min=args.timeslot_min,
        seed=args.seed,
        num_train_workers=args.train_workers,
        num_workers=args.data_workers,
        output_dir=args.output,
        verbose=not args.quiet,
        lang=args.lang,
        partition_strategy=args.partition_strategy,
        class_probability=args.class_probability,
        preference_mode=args.preference_mode,
        preferred_clients_per_class=args.preferred_clients_per_class,
        sample_cap_strategy=args.sample_cap_strategy,
        data_dir=args.data_dir,
        limit_to_sim_window=not args.allow_sim_extension,
    )

    total_elapsed = _time.time() - t_start
    print(f"\n总耗时: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
