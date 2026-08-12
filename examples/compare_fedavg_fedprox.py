"""
SpaceFL — FedAvg vs FedProx 对比实验
=====================================

统一对比脚本：相同轨道条件、相同数据划分、相同地面站，
在 FedAvg 和 FedProx (μ=0.01, 0.1) 之间公平对比准确率。

实验配置:
    地面站: 5 (全球分布, Paper Table 3 前 5 站)
    卫星:   12 (500km 均高圆形轨道, 倾角 53°)
    轮次:   100
    数据集: MNIST non-IID (probability + class_balanced)
    设备:   CPU

用法:
    python examples/compare_fedavg_fedprox.py
    python examples/compare_fedavg_fedprox.py --rounds 150 --local-epochs 3
    python examples/compare_fedavg_fedprox.py --mu-values 0.01 0.05 0.1 0.5

输出:
    compare_output/
    ├── summary.json               — 对比汇总指标
    ├── accuracy_comparison.png    — 准确率对比曲线
    ├── loss_comparison.png        — 损失对比曲线
    ├── fedavg_history.json
    ├── fedprox_mu0.01_history.json
    ├── fedprox_mu0.1_history.json
    └── contact_stats.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
import sys
import time as _time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_space.environment import CelestialBody, GroundStation, GroundStationNetwork
from fl_space.fl.fedavg import (
    CappedSelector,
    FixedEpochTrainer,
    StandardEvaluator,
    SyncWeightedAggregator,
)
from fl_space.fl.fedprox import ProximalTrainer
from fl_space.fl.runner import FLRunner
from fl_space.fl.scheduler import CommunicationScheduler
from fl_space.fl.server import FLConfig
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

# ── 地面站: Paper Table 3 前 5 站 ──────────────────────────────

PAPER_GS = [
    ("Sioux Falls", 43.55, -96.72),
    ("Sanya", 18.25, 109.5),
    ("Johannesburg", -26.2, 28.03),
    ("Cordoba", -31.4, -64.18),
    ("Tromso", 69.65, 18.95),
]


def create_gs_network(n: int) -> GroundStationNetwork:
    n = min(n, len(PAPER_GS))
    return GroundStationNetwork(
        [GroundStation(name, lat, lon, 0.05) for name, lat, lon in PAPER_GS[:n]]
    )


# ── 轨道创建 ───────────────────────────────────────────────────


def create_uniform_orbits(
    body: CelestialBody,
    n_sats: int,
    altitude_km: float = 500.0,
    inclination_deg: float = 53.0,
    raan_deg: float = 0.0,
):
    from fl_space.orbit import create_circular_orbit

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


# ── 实验结果 ───────────────────────────────────────────────────


@dataclass
class RunResult:
    name: str
    algorithm: str
    mu: float
    history: list[dict] = field(default_factory=list)
    round_timeslots: list[int] = field(default_factory=list)
    elapsed_sec: float = 0.0
    final_acc: float = 0.0
    max_acc: float = 0.0
    min_acc: float = 0.0
    mean_acc: float = 0.0
    std_acc: float = 0.0


# ── 单次运行 ───────────────────────────────────────────────────


def run_one(
    *,
    name: str,
    algorithm: str,
    mu: float,
    sim: OrbitSimulator,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    seed: int,
    dataset: str,
    non_iid: bool,
    classes_per_client: int,
    max_samples_per_client: int,
    partition_strategy: str,
    class_probability: float,
    preference_mode: str,
    preferred_clients_per_class: int,
    sample_cap_strategy: str,
    data_dir: str,
    limit_to_sim_window: bool,
    verbose: bool,
) -> RunResult:
    """运行单次 FL 实验（FedAvg 或 FedProx）。"""

    n_sats = sim.num_satellites
    n_gs = sim.num_ground_stations
    max_selected = min(n_gs, n_sats)

    t0 = _time.time()

    # FL 配置
    fl_config = FLConfig(
        algorithm=algorithm,
        num_rounds=num_rounds,
        num_clients=n_sats,
        fraction=1.0,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mu=mu,
        device=device,
        seed=seed,
        time_model="slot",
        time_model_kwargs={"slots_per_epoch": 1},
        early_stop_acc=0.99,  # 不早停，跑满 100 轮
        num_train_workers=1,
        num_workers=0,
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

    if algorithm == "fedprox":
        trainer = ProximalTrainer(
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            mu=mu,
            device=device,
        )
    else:
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
        print(f"  [{name}] 启动: algo={algorithm}, mu={mu}, max_selected={max_selected}")

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
        verbose=False,
    )

    # 转换为字典
    history_dict = []
    round_ts = []
    for h in history:
        history_dict.append({
            "round": h.round_num,
            "timeslot_start": getattr(h, "timeslot_start", None),
            "timeslot_end": getattr(h, "timeslot_end", None),
            "accuracy": h.eval_metrics.get("accuracy", 0),
            "loss": h.eval_metrics.get("loss", 0),
            "online_clients": getattr(h, "num_online", None),
            "selected_clients": getattr(h, "num_selected", None),
            "train_loss": getattr(h, "train_loss", None),
        })
        round_ts.append(getattr(h, "timeslot_start", h.round_num))

    elapsed = _time.time() - t0

    accs = [h["accuracy"] for h in history_dict]
    result = RunResult(
        name=name,
        algorithm=algorithm,
        mu=mu,
        history=history_dict,
        round_timeslots=round_ts,
        elapsed_sec=elapsed,
        final_acc=accs[-1] if accs else 0,
        max_acc=max(accs) if accs else 0,
        min_acc=min(accs) if accs else 0,
        mean_acc=float(np.mean(accs)) if accs else 0,
        std_acc=float(np.std(accs)) if accs else 0,
    )

    if verbose:
        print(f"  [{name}] 完成: {len(history_dict)} 轮, "
              f"Final Acc={result.final_acc:.4f}, Max={result.max_acc:.4f}, "
              f"耗时={elapsed:.1f}s")

    return result


# ── 接触统计 ───────────────────────────────────────────────────


def compute_contact_stats(sim: OrbitSimulator) -> dict:
    cm = sim.contact_matrix
    gs_network = sim.ground_network
    n_sats = sim.num_satellites
    n_slots = sim.num_timeslots
    n_gs = sim.num_ground_stations

    mat = cm.simple_matrix

    per_satellite = {}
    for sat_id in range(n_sats):
        row = mat[sat_id]
        contact_slots = int(np.sum(row >= 0))
        per_satellite[sat_id] = {
            "total_contact_slots": contact_slots,
            "contact_rate": float(contact_slots / n_slots),
        }

    return {
        "num_satellites": n_sats,
        "num_ground_stations": n_gs,
        "num_timeslots": n_slots,
        "timeslot_duration_min": sim.timeslot_duration_min,
        "contact_rate": sim.stats.get("contact_rate", 0),
        "per_satellite": per_satellite,
    }


# ── 可视化 ─────────────────────────────────────────────────────

COLORS = {
    "fedavg": "#3498db",
    "fedprox_mu0.01": "#e74c3c",
    "fedprox_mu0.1": "#2ecc71",
    "fedprox_mu0.5": "#f39c12",
}


def plot_comparison(results: list[RunResult], output_dir: str) -> None:
    """生成对比图表。"""
    if not HAS_MPL or not results:
        return

    os.makedirs(output_dir, exist_ok=True)
    n_algos = len(results)

    # ── 1. 准确率对比 (主图) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for r in results:
        label = f"{r.name} (max={r.max_acc:.3f})"
        rounds = [h["round"] for h in r.history]
        accs = [h["accuracy"] for h in r.history]
        color_key = f"{r.algorithm}_mu{r.mu}" if r.algorithm == "fedprox" else r.algorithm
        color = COLORS.get(color_key, "#95a5a6")
        ax1.plot(rounds, accs, color=color, linewidth=1.5, label=label, alpha=0.9)

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("FedAvg vs FedProx — Accuracy per Round\n(GS=5, SAT=12, MNIST non-IID)")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="lower right")

    # ── 2. 损失对比 ──
    for r in results:
        label = r.name
        rounds = [h["round"] for h in r.history]
        losses = [h.get("loss", 0) for h in r.history]
        color_key = f"{r.algorithm}_mu{r.mu}" if r.algorithm == "fedprox" else r.algorithm
        color = COLORS.get(color_key, "#95a5a6")
        ax2.plot(rounds, losses, color=color, linewidth=1.5, label=label, alpha=0.9)

    ax2.set_xlabel("Round")
    ax2.set_ylabel("Test Loss")
    ax2.set_title("FedAvg vs FedProx — Loss per Round")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="upper right")

    fig.suptitle("SpaceFL — FedAvg vs FedProx Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [图表] accuracy_comparison.png")

    # ── 3. 柱状图对比 ──
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

    names = [r.name for r in results]
    final_accs = [r.final_acc for r in results]
    max_accs = [r.max_acc for r in results]
    elapsed_min = [r.elapsed_sec / 60 for r in results]

    x = np.arange(len(names))
    width = 0.35

    colors_list = []
    for r in results:
        ck = f"{r.algorithm}_mu{r.mu}" if r.algorithm == "fedprox" else r.algorithm
        colors_list.append(COLORS.get(ck, "#95a5a6"))

    bars1 = ax3.bar(x - width / 2, final_accs, width, label="Final Acc", color=colors_list, edgecolor="white")
    bars2 = ax3.bar(x + width / 2, max_accs, width, label="Max Acc", color=colors_list, alpha=0.5, edgecolor="white")

    for bar, v in zip(bars1, final_accs):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.3f}",
                 ha="center", fontsize=8, fontweight="bold")
    for bar, v in zip(bars2, max_accs):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.3f}",
                 ha="center", fontsize=8)

    ax3.set_xticks(x)
    ax3.set_xticklabels(names, fontsize=9)
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Final vs Max Accuracy")
    ax3.set_ylim(0, 1.1)
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    ax4.bar(x, elapsed_min, color=colors_list, edgecolor="white")
    for i, (bar, v) in enumerate(zip(ax4.patches, elapsed_min)):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"{v:.1f}min",
                 ha="center", fontsize=8, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, fontsize=9)
    ax4.set_ylabel("Time (minutes)")
    ax4.set_title("Wall-clock Time")
    ax4.grid(axis="y", alpha=0.3)

    fig.suptitle("SpaceFL — FedAvg vs FedProx Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "summary_bars.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [图表] summary_bars.png")


# ── 主流程 ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="SpaceFL — FedAvg vs FedProx 对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gs", type=int, default=5, help="地面站数量 (默认: 5)")
    parser.add_argument("--sats", type=int, default=12, help="卫星数量 (默认: 12)")
    parser.add_argument("--rounds", type=int, default=100, help="训练轮次 (默认: 100)")
    parser.add_argument("--local-epochs", type=int, default=2, help="本地 epoch (默认: 2)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--mu-values", type=float, nargs="+", default=[0.01, 0.1],
                        help="FedProx μ 值列表 (默认: 0.01 0.1)")
    parser.add_argument("--altitude", type=float, default=500.0, help="轨道高度 km")
    parser.add_argument("--inclination", type=float, default=53.0, help="轨道倾角 °")
    parser.add_argument("--sim-hours", type=float, default=24.0,
                        help="模拟时长/小时 (默认: 24)")
    parser.add_argument("--timeslot-min", type=float, default=1.0)
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes-per-client", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output", "-o", type=str, default="compare_output")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    quiet = args.quiet

    # ── 1. 创建共享的 OrbitSimulator ──
    if not quiet:
        print("=" * 70)
        print("  SpaceFL — FedAvg vs FedProx 对比实验")
        print("=" * 70)
        print(f"  地面站: {args.gs} (全球分布)")
        print(f"  卫星:   {args.sats} ({args.altitude}km, {args.inclination}° 倾角)")
        print(f"  轮次:   {args.rounds}")
        print(f"  数据集: {args.dataset} (non-IID, probability+class_balanced)")
        print(f"  FedProx μ: {args.mu_values}")
        print(f"  模拟时长: {args.sim_hours}h")
        print(f"  设备:   {args.device}")
        print("=" * 70)

    body = CelestialBody.earth()
    orbits = create_uniform_orbits(body, args.sats, args.altitude, args.inclination)
    gs_network = create_gs_network(args.gs)
    num_timeslots = int(args.sim_hours * 60 / args.timeslot_min)

    sim = OrbitSimulator(
        body=body,
        orbits=orbits,
        ground_station_network=gs_network,
        num_timeslots=num_timeslots,
        timeslot_duration_min=args.timeslot_min,
        backend="kepler",
        contact_mode="simple",
        verbose=False,
    )

    if not quiet:
        print(f"\n  {sim.summary()}")

    # 接触统计
    contact_stats = compute_contact_stats(sim)

    # ── 2. 运行所有实验 ──
    results: list[RunResult] = []

    # FedAvg
    if not quiet:
        print(f"\n{'─' * 50}")
        print("  [1/3] FedAvg")
        print(f"{'─' * 50}")

    r_avg = run_one(
        name="FedAvg",
        algorithm="fedavg",
        mu=0.0,
        sim=sim,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
        dataset=args.dataset,
        non_iid=True,
        classes_per_client=args.classes_per_client,
        max_samples_per_client=args.max_samples,
        partition_strategy="probability",
        class_probability=0.8,
        preference_mode="class_balanced",
        preferred_clients_per_class=1,
        sample_cap_strategy="preserve",
        data_dir=args.data_dir,
        limit_to_sim_window=True,
        verbose=not quiet,
    )
    results.append(r_avg)

    # FedProx (多个 μ)
    for idx, mu_val in enumerate(args.mu_values, 1):
        name = f"FedProx (μ={mu_val})"
        if not quiet:
            print(f"\n{'─' * 50}")
            print(f"  [{idx+1}/{len(args.mu_values)+1}] {name}")
            print(f"{'─' * 50}")

        r_prox = run_one(
            name=name,
            algorithm="fedprox",
            mu=mu_val,
            sim=sim,
            num_rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            seed=args.seed,
            dataset=args.dataset,
            non_iid=True,
            classes_per_client=args.classes_per_client,
            max_samples_per_client=args.max_samples,
            partition_strategy="probability",
            class_probability=0.8,
            preference_mode="class_balanced",
            preferred_clients_per_class=1,
            sample_cap_strategy="preserve",
            data_dir=args.data_dir,
            limit_to_sim_window=True,
            verbose=not quiet,
        )
        results.append(r_prox)

    # ── 3. 输出 ──
    os.makedirs(args.output, exist_ok=True)

    # 保存各算法历史
    for r in results:
        fname = r.name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        out_path = os.path.join(args.output, f"{fname}_history.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(r.history, f, ensure_ascii=False, indent=2)

    # 接触统计
    with open(os.path.join(args.output, "contact_stats.json"), "w", encoding="utf-8") as f:
        json.dump(contact_stats, f, ensure_ascii=False, indent=2)

    # ── 4. 对比汇总 ──
    summary = {
        "config": {
            "gs_count": args.gs,
            "sat_count": args.sats,
            "num_rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "mu_values": args.mu_values,
            "altitude_km": args.altitude,
            "inclination_deg": args.inclination,
            "sim_hours": args.sim_hours,
            "timeslot_duration_min": args.timeslot_min,
            "dataset": args.dataset,
            "device": args.device,
            "seed": args.seed,
        },
        "contact_stats": contact_stats,
        "results": [],
    }

    for r in results:
        summary["results"].append({
            "name": r.name,
            "algorithm": r.algorithm,
            "mu": r.mu,
            "rounds": len(r.history),
            "final_acc": round(r.final_acc, 4),
            "max_acc": round(r.max_acc, 4),
            "min_acc": round(r.min_acc, 4),
            "mean_acc": round(r.mean_acc, 4),
            "std_acc": round(r.std_acc, 4),
            "elapsed_sec": round(r.elapsed_sec, 1),
            "elapsed_min": round(r.elapsed_sec / 60, 1),
        })

    # 计算增益
    if len(results) >= 2:
        baseline = results[0].max_acc
        for i, r in enumerate(results[1:], 1):
            gain = r.max_acc - baseline
            summary["results"][i]["gain_vs_fedavg"] = round(gain, 4)

    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── 5. 可视化 ──
    if HAS_MPL:
        plot_comparison(results, args.output)

    # ── 6. 终端报告 ──
    if not quiet:
        print(f"\n{'=' * 70}")
        print("  对比结果汇总")
        print(f"{'=' * 70}")
        print(f"  {'算法':<20} {'最终Acc':>8} {'最高Acc':>8} {'平均Acc':>8} {'标准差':>8} {'耗时':>8}")
        print(f"  {'─' * 60}")
        for r in results:
            print(f"  {r.name:<20} {r.final_acc:>8.4f} {r.max_acc:>8.4f} "
                  f"{r.mean_acc:>8.4f} {r.std_acc:>8.4f} {r.elapsed_sec:>7.1f}s")

        if len(results) >= 2:
            baseline = results[0]
            print(f"\n  --- vs FedAvg 基准 (max={baseline.max_acc:.4f}) ---")
            for r in results[1:]:
                gain = r.max_acc - baseline.max_acc
                sign = "+" if gain >= 0 else ""
                print(f"  {r.name}: Δmax_acc = {sign}{gain:.4f} "
                      f"({'提升' if gain >= 0 else '下降'} {abs(gain)*100:.2f}%)")

        print(f"\n  输出目录: {os.path.abspath(args.output)}/")
        print("    summary.json              — 完整对比汇总")
        print("    contact_stats.json        — 接触统计")
        for r in results:
            fname = r.name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
            print(f"    {fname}_history.json  — {r.name} 逐轮记录")
        if HAS_MPL:
            print("    accuracy_comparison.png   — 准确率对比图")
            print("    summary_bars.png          — 汇总柱状图")
        print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
