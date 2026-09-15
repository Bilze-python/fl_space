"""
SpaceFL 可视化工具 — 实验输出图表和统计

提供：
    - 接触矩阵热力图
    - 准确率对比曲线（SpaceFL vs 标准FL）
    - 时间分解堆叠图
    - 地面站分布地图
    - 接触统计汇总
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from fl_space.viz.i18n import setup_cjk_font, t, tf

try:
    import matplotlib
    matplotlib.use("Agg")  # 无头服务器兼容
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── 辅助函数 ──────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    """确保目录存在。"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ── 接触统计 ──────────────────────────────────────────────────

def get_contact_statistics(sim) -> dict[str, Any]:
    """
    从 OrbitSimulator 提取详细接触统计。

    Parameters
    ----------
    sim : OrbitSimulator
        已初始化的模拟器实例。

    Returns
    -------
    dict
        包含：
        - per_gs: 每个地面站的统计 {gs_id: {total_contacts, contacted_sats, avg_duration_slots, ...}}
        - per_satellite: 每个卫星的统计
        - global: 全局汇总
    """
    cm = sim.contact_matrix
    gs_network = sim.ground_network
    n_sats = sim.num_satellites
    n_slots = sim.num_timeslots
    n_gs = sim.num_ground_stations

    # 从 simple_matrix 提取每个 (gs, sat) 的接触片段
    mat = cm.simple_matrix  # shape (N_sats, N_slots), values = gs_id or -1

    per_gs: dict[int, dict] = {}
    for gs_id in range(n_gs):
        gs_contacts = []
        for sat_id in range(n_sats):
            row = mat[sat_id]
            in_contact = (row == gs_id)
            # 找连续接触段（窗口）
            padded = np.concatenate([[False], in_contact, [False]])
            changes = np.diff(padded.astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            for s, e in zip(starts, ends):
                gs_contacts.append({
                    "sat_id": int(sat_id),
                    "start_ts": int(s),
                    "end_ts": int(e),
                    "duration_slots": int(e - s),
                })

        durations = [c["duration_slots"] for c in gs_contacts]
        contacted_sats = sorted(set(c["sat_id"] for c in gs_contacts))
        per_gs[gs_id] = {
            "gs_name": gs_network.names[gs_id] if gs_id < len(gs_network.names) else f"GS-{gs_id}",
            "lat": float(gs_network[gs_id].lat_deg) if hasattr(gs_network, '__getitem__') else 0.0,
            "lon": float(gs_network[gs_id].lon_deg) if hasattr(gs_network, '__getitem__') else 0.0,
            "total_contacts": len(gs_contacts),
            "contacted_satellites": contacted_sats,
            "num_contacted_sats": len(contacted_sats),
            "avg_duration_slots": float(np.mean(durations)) if durations else 0.0,
            "max_duration_slots": int(max(durations)) if durations else 0,
            "total_duration_slots": int(sum(durations)) if durations else 0,
        }

    # 每卫星统计
    per_satellite = {}
    for sat_id in range(n_sats):
        row = mat[sat_id]
        contact_slots = np.sum(row >= 0)
        contacted_gs = sorted(set(int(row[ts]) for ts in range(n_slots) if row[ts] >= 0))
        # 卫星平均接触窗口时长
        in_contact = (row >= 0)
        padded = np.concatenate([[False], in_contact, [False]])
        changes = np.diff(padded.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        durations = [int(e - s) for s, e in zip(starts, ends)]
        per_satellite[sat_id] = {
            "total_contact_slots": int(contact_slots),
            "contact_rate": float(contact_slots / n_slots),
            "contacted_gs": contacted_gs,
            "num_windows": len(durations),
            "avg_window_slots": float(np.mean(durations)) if durations else 0.0,
            "max_window_slots": int(max(durations)) if durations else 0,
        }

    avg_contact_min = sim.timeslot_duration_min * (
        np.mean([g["avg_duration_slots"] for g in per_gs.values()])
        if per_gs else 0
    )

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
            "avg_contact_duration_slots": float(np.mean(
                [g["avg_duration_slots"] for g in per_gs.values()]
            )) if per_gs else 0,
            "avg_contact_duration_min": float(avg_contact_min),
        },
    }


# ── 触矩阵热力图 ──────────────────────────────────────────────

def plot_contact_heatmap(
    sim,
    output_path: str = "contact_heatmap.png",
    title: str = "Contact Matrix Heatmap",
    lang: str = "en",
) -> str:
    """
    生成接触矩阵热力图。

    横轴 = timeslot, 纵轴 = 卫星ID, 颜色 = 接触地面站数。

    Parameters
    ----------
    sim : OrbitSimulator
        模拟器实例。
    output_path : str
        输出 PNG 路径。
    title : str
        图表标题。
    lang : str
        语言，'en' 或 'zh'。

    Returns
    -------
    str
        输出文件路径。
    """
    if not HAS_MPL:
        print("[警告] matplotlib 未安装，跳过热力图生成")
        return output_path

    if lang == "zh":
        setup_cjk_font()

    cm = sim.contact_matrix
    n_sats = sim.num_satellites
    n_slots = sim.num_timeslots

    # 构建热力图数据：每个 (sat, ts) 的接触地面站数
    heatmap_data = np.zeros((n_sats, n_slots), dtype=np.float32)
    for sat_id in range(n_sats):
        for ts in range(n_slots):
            contacts = cm.get_all_contacts(sat_id, ts)
            heatmap_data[sat_id, ts] = len(contacts)

    fig, ax = plt.subplots(figsize=(16, max(5, n_sats * 0.4)))
    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        cmap="YlOrRd",
        origin="lower",
        interpolation="nearest",
    )

    # 时间轴标注
    hours = n_slots * sim.timeslot_duration_min / 60.0
    n_ticks = min(12, n_slots // 100 + 1)
    tick_positions = np.linspace(0, n_slots - 1, n_ticks, dtype=int)
    tick_labels = [f"{ts * sim.timeslot_duration_min / 60:.1f}h" for ts in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)

    ax.set_yticks(range(n_sats))
    ax.set_yticklabels([f"SAT-{i}" for i in range(n_sats)], fontsize=8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(t("Connected Ground Stations", lang), fontsize=9)

    ax.set_xlabel(tf("Time ({hours:.0f}h total, {slot_min}min/slot)", lang,
                     hours=hours, slot_min=sim.timeslot_duration_min))
    ax.set_ylabel(t("Satellite ID", lang))
    ax.set_title(f"{tf(title, lang) if lang == 'zh' else title}\n"
                 f"{t('Contact Rate:', lang)} {sim.stats.get('contact_rate', 0):.1%}")

    plt.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ── 准确率对比曲线 ────────────────────────────────────────────

def plot_accuracy_comparison(
    spacefl_history: list[dict],
    baseline_history: list[dict] | None = None,
    output_path: str = "accuracy_comparison.png",
    title: str = "Accuracy: SpaceFL vs Standard FL",
    lang: str = "en",
) -> str:
    """
    绘制 SpaceFL 与标准 FL 准确率对比曲线。

    Parameters
    ----------
    spacefl_history : list[dict]
        SpaceFL 训练历史（每轮含 accuracy, round, timeslot）。
    baseline_history : list[dict] | None
        标准 FL 基线历史，None 则只显示 SpaceFL。
    output_path : str
        输出 PNG 路径。
    title : str
        图表标题。
    lang : str
        语言，'en' 或 'zh'。

    Returns
    -------
    str
        输出文件路径。
    """
    if not HAS_MPL:
        print("[警告] matplotlib 未安装，跳过准确率图")
        return output_path

    if lang == "zh":
        setup_cjk_font()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：按轮次
    rounds_space = [h.get("round", i) for i, h in enumerate(spacefl_history)]
    acc_space = [h.get("accuracy", 0) for h in spacefl_history]
    ax1.plot(rounds_space, acc_space, "b-o", markersize=3, linewidth=1.2, label=t("SpaceFL", lang))

    if baseline_history:
        rounds_base = [h.get("round", i) for i, h in enumerate(baseline_history)]
        acc_base = [h.get("accuracy", 0) for h in baseline_history]
        ax1.plot(rounds_base, acc_base, "r--s", markersize=3, linewidth=1.2, label=t("Standard FL", lang))

    ax1.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5, label=t("90% threshold", lang))
    ax1.set_xlabel(t("Round", lang))
    ax1.set_ylabel(t("Accuracy", lang))
    ax1.set_title(t("Accuracy vs Rounds", lang))
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # 右图：按时间槽
    ts_space = [h.get("timeslot", h.get("round", i) * 10) for i, h in enumerate(spacefl_history)]
    ax2.plot(ts_space, acc_space, "b-o", markersize=3, linewidth=1.2, label=t("SpaceFL", lang))

    if baseline_history:
        ts_base = [h.get("timeslot", h.get("round", i) * 10) for i, h in enumerate(baseline_history)]
        ax2.plot(ts_base, acc_base, "r--s", markersize=3, linewidth=1.2, label=t("Standard FL", lang))

    ax2.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel(t("Timeslot", lang))
    ax2.set_ylabel(t("Accuracy", lang))
    ax2.set_title(t("Accuracy vs Virtual Time", lang))
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle(tf(title, lang) if lang == "zh" else title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ── 时间分解图 ─────────────────────────────────────────────────

def plot_time_breakdown(
    history: list[dict],
    output_path: str = "time_breakdown.png",
    title: str = "Per-Round Time Breakdown",
    lang: str = "en",
) -> str:
    """
    绘制每轮时间分解堆叠柱状图。

    Parameters
    ----------
    history : list[dict]
        训练历史（每轮含 time_breakdown 字段）。
    output_path : str
        输出 PNG 路径。
    title : str
        图表标题。
    lang : str
        语言，'en' 或 'zh'。

    Returns
    -------
    str
        输出文件路径。
    """
    if not HAS_MPL:
        print("[警告] matplotlib 未安装，跳过时间分解图")
        return output_path

    if lang == "zh":
        setup_cjk_font()

    # 提取时间分解数据
    rounds = []
    wait_dist = []
    download = []
    train = []
    wait_return = []
    upload = []

    for h in history:
        tb = h.get("time_breakdown", {})
        if not tb:
            continue
        rounds.append(h.get("round", len(rounds)))
        wait_dist.append(tb.get("wait_distribution", 0))
        download.append(tb.get("download", 0))
        train.append(tb.get("train", 0))
        wait_return.append(tb.get("wait_return", 0))
        upload.append(tb.get("upload", 0))

    if not rounds:
        return output_path

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(rounds))
    width = 0.7

    bottom = np.zeros(len(rounds))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    labels_raw = ["Wait Dist", "Download", "Train", "Wait Return", "Upload"]
    labels = [t(l, lang) for l in labels_raw]
    data = [wait_dist, download, train, wait_return, upload]

    for d, color, label in zip(data, colors, labels):
        d_arr = np.array(d)
        ax.bar(x, d_arr, width, bottom=bottom, color=color, label=label, alpha=0.85)
        bottom += d_arr

    ax.set_xlabel(t("Round", lang))
    ax.set_ylabel(t("Timeslot", lang))
    ax.set_title(tf(title, lang) if lang == "zh" else title)
    ax.legend(fontsize=8, ncol=5, loc="upper right")
    ax.set_xticks(x[::max(1, len(rounds) // 20)])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ── 地面站分布地图 ────────────────────────────────────────────

def plot_ground_station_map(
    sim,
    output_path: str = "ground_station_map.png",
    title: str = "Ground Station Distribution",
    show_tracks: bool = False,
    lang: str = "en",
) -> str:
    """
    绘制世界地图上的地面站分布和卫星轨道。

    Parameters
    ----------
    sim : OrbitSimulator
        模拟器实例。
    output_path : str
        输出 PNG 路径。
    title : str
        图表标题。
    show_tracks : bool
        是否显示卫星轨迹。
    lang : str
        语言，'en' 或 'zh'。

    Returns
    -------
    str
        输出文件路径。
    """
    if not HAS_MPL:
        print("[警告] matplotlib 未安装，跳过地图生成")
        return output_path

    if lang == "zh":
        setup_cjk_font()

    fig, ax = plt.subplots(figsize=(14, 7))

    # 绘制地面站
    gs_network = sim.ground_network
    for i, gs in enumerate(gs_network):
        ax.scatter(gs.lon_deg, gs.lat_deg, c="red", s=100, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(
            f"GS-{i}\n{gs.name}",
            (gs.lon_deg, gs.lat_deg),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=7,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # 显示卫星轨迹（可选）
    if show_tracks:
        for sat_id in range(min(sim.num_satellites, 5)):  # 最多显示5颗星
            lats, lons = [], []
            step = max(1, sim.num_timeslots // 200)
            for ts in range(0, sim.num_timeslots, step):
                lat, lon = sim.get_sat_position(sat_id, ts)
                lats.append(lat)
                lons.append(lon)
            ax.plot(lons, lats, linewidth=0.5, alpha=0.5, label=f"SAT-{sat_id}")

    # 地图范围
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel(t("Longitude", lang))
    ax.set_ylabel(t("Latitude", lang))
    ax.set_title(tf(title, lang) if lang == "zh" else title)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    # 经纬度刻度
    ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(30))

    if show_tracks:
        ax.legend(fontsize=6, loc="lower left")

    plt.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ── 实验报告器 ─────────────────────────────────────────────────

def save_experiment_report(
    results: dict[str, Any],
    output_dir: str = "experiment_output",
    lang: str = "en",
) -> str:
    """
    保存完整实验报告（JSON + 所有图表）。

    Parameters
    ----------
    results : dict
        实验结果字典，应包含：
        - config: 实验配置
        - experiments: list of {name, gs_count, spacefl_history, baseline_history, contact_stats}
    output_dir : str
        输出目录。
    lang : str
        语言，'en' 或 'zh'。

    Returns
    -------
    str
        报告 JSON 文件路径。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存 JSON 报告
    report_path = os.path.join(output_dir, "experiment_report.json")

    # 清理不可序列化的字段
    clean_results = _make_serializable(results)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2, default=str)

    # 生成图表
    if HAS_MPL:
        for exp in results.get("experiments", []):
            exp_name = exp.get("name", "experiment")
            exp_dir = os.path.join(output_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            spacefl_hist = exp.get("spacefl_history", [])
            baseline_hist = exp.get("baseline_history")

            # 准确率对比
            if spacefl_hist:
                plot_accuracy_comparison(
                    spacefl_hist, baseline_hist,
                    output_path=os.path.join(exp_dir, "accuracy.png"),
                    title=f"Accuracy: {exp_name}",
                    lang=lang,
                )

            # 时间分解
            if spacefl_hist:
                plot_time_breakdown(
                    spacefl_hist,
                    output_path=os.path.join(exp_dir, "time_breakdown.png"),
                    title=f"Time Breakdown: {exp_name}",
                    lang=lang,
                )

            # 时空信息图
            exp_config = exp.get("config", {})
            time_info = [
                f"GS={exp.get('gs_count', '?')}, "
                f"Sats={exp_config.get('num_satellites', '?')}, "
                f"Rounds={len(spacefl_hist)}, "
                f"FinalAcc={spacefl_hist[-1].get('accuracy', 0):.3f}" if spacefl_hist else ""
            ]
            info_path = os.path.join(exp_dir, "info.txt")
            with open(info_path, "w", encoding="utf-8") as f:
                f.write(f"Experiment: {exp_name}\n")
                f.write(f"Ground Stations: {exp.get('gs_count', '?')}\n")
                f.write(f"Rounds completed: {len(spacefl_hist)}\n")
                if spacefl_hist:
                    f.write(f"Final accuracy: {spacefl_hist[-1].get('accuracy', 0):.4f}\n")
                    f.write(f"Total timeslots: {spacefl_hist[-1].get('timeslot', 0)}\n")

    return report_path


def _make_serializable(obj: Any) -> Any:
    """递归转换 numpy 类型为 Python 原生类型。"""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
