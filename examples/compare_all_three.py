"""
SpaceFL 三算法对比实验 — FedAvg / FedProx / FedBuff
=====================================================

在相同轨道条件下，对比三种联邦学习算法在太空场景中的表现：
- FedAvg:   同步加权平均（论文：McMahan et al., AISTATS 2017）
- FedProx:  同步近端优化（论文：Li et al., MLSys 2020）
- FedBuff:  异步缓冲聚合（论文：Nguyen et al., MLSys 2022）

实验配置:
    地面站: 5 (全球分布)
    卫星:   6 (500km 均高)
    轮次:   50 次服务端更新
    数据集: MNIST non-IID (probability + class_balanced)

关键对比维度:
    - 准确率 vs 时隙 (timeslot): FedBuff 异步优势
    - 准确率 vs 服务端更新: 收敛质量
    - FedBuff 陈旧度 (staleness) 分布

用法:
    python examples/compare_all_three.py
    python examples/compare_all_three.py --sats 12 --rounds 80
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
from fl_space.fl.fedbuff import AsyncSelector, AsyncTrainer, BufferAggregator
from fl_space.fl.fedprox import ProximalTrainer
from fl_space.fl.runner import FLRunner
from fl_space.fl.scheduler import CommunicationScheduler
from fl_space.fl.server import FLConfig
from fl_space.simulator import OrbitSimulator

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── 地面站 ────────────────────────────────────────────────────

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


# ── 轨道 ──────────────────────────────────────────────────────

def create_uniform_orbits(body, n_sats, altitude_km=500.0, inclination_deg=53.0):
    from fl_space.orbit import create_circular_orbit
    orbits = []
    for i in range(n_sats):
        ta = i * (360.0 / n_sats)
        orbits.append(create_circular_orbit(
            altitude_km=altitude_km, inclination_deg=inclination_deg,
            raan_deg=0.0, true_anomaly_deg=ta, body=body,
        ))
    return orbits


# ── 结果 ──────────────────────────────────────────────────────

@dataclass
class RunResult:
    name: str
    algorithm: str
    history: list[dict] = field(default_factory=list)
    elapsed_sec: float = 0.0
    final_acc: float = 0.0
    max_acc: float = 0.0
    final_timeslot: int = 0
    total_server_updates: int = 0
    total_client_updates: int = 0
    mean_staleness: float = 0.0


# ── FedAvg / FedProx 运行 ─────────────────────────────────────

def run_sync_algo(
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
    class_probability: float,
    data_dir: str,
    verbose: bool,
) -> RunResult:
    """通过 FLRunner + OrbitSimulator 运行同步算法。"""
    t0 = _time.time()

    n_sats = sim.num_satellites
    n_gs = sim.num_ground_stations
    max_selected = min(n_gs, n_sats)

    scheduler = CommunicationScheduler(sim)

    if algorithm == "fedprox":
        trainer = ProximalTrainer(
            local_epochs=local_epochs, batch_size=batch_size,
            learning_rate=learning_rate, mu=mu, device=device,
        )
    else:
        trainer = FixedEpochTrainer(
            local_epochs=local_epochs, batch_size=batch_size,
            learning_rate=learning_rate, device=device,
        )

    selector = CappedSelector(max_count=max_selected, min_clients=1, seed=seed)
    aggregator = SyncWeightedAggregator(min_updates=1)
    evaluator = StandardEvaluator(device=device)

    config = FLConfig(
        algorithm=algorithm, num_rounds=num_rounds, num_clients=n_sats,
        fraction=1.0, local_epochs=local_epochs, batch_size=batch_size,
        learning_rate=learning_rate, mu=mu, device=device, seed=seed,
        time_model="slot", time_model_kwargs={"slots_per_epoch": 1},
        early_stop_acc=0.99, limit_to_sim_window=True,
        partition_strategy="probability", class_probability=class_probability,
        preference_mode="class_balanced", preferred_clients_per_class=1,
        sample_cap_strategy="preserve", data_dir=data_dir,
    )

    runner = FLRunner(config, selector, trainer, aggregator, evaluator, scheduler=scheduler)

    if verbose:
        print(f"  [{name}] max_selected={max_selected}, mu={mu}")

    history = runner.run(
        dataset_name=dataset, iid=False, alpha=0.5,
        classes_per_client=2, max_samples_per_client=1000,
        partition_strategy="probability", class_probability=class_probability,
        preference_mode="class_balanced", preferred_clients_per_class=1,
        sample_cap_strategy="preserve", data_dir=data_dir,
        verbose=False,
    )

    history_dict = []
    for h in history:
        history_dict.append({
            "round": h.round_num,
            "timeslot": getattr(h, "timeslot", h.round_num),
            "timeslot_start": getattr(h, "timeslot_start", None),
            "accuracy": h.eval_metrics.get("accuracy", 0),
            "loss": h.eval_metrics.get("loss", 0),
            "num_clients": getattr(h, "num_clients", None),
        })

    elapsed = _time.time() - t0
    trained = [r for r in history_dict if r["round"] >= 0]
    accs = [r["accuracy"] for r in trained]

    result = RunResult(
        name=name, algorithm=algorithm,
        history=history_dict, elapsed_sec=elapsed,
        final_acc=accs[-1] if accs else 0,
        max_acc=max(accs) if accs else 0,
        final_timeslot=trained[-1]["timeslot"] if trained else 0,
        total_server_updates=len(trained),
        total_client_updates=sum(r.get("num_clients", 0) or 0 for r in trained),
    )

    if verbose:
        print(f"  [{name}] {len(trained)} updates, max_acc={result.max_acc:.4f}, "
              f"final_ts={result.final_timeslot}, {elapsed:.1f}s")

    return result


# ── FedBuff 运行 ──────────────────────────────────────────────

def run_fedbuff(
    *,
    sim: OrbitSimulator,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    buffer_size: int,
    staleness_weight: bool,
    server_learning_rate: float,
    device: str,
    seed: int,
    dataset: str,
    class_probability: float,
    data_dir: str,
    async_eval_every: int,
    verbose: bool,
) -> RunResult:
    """通过 FLRunner + OrbitSimulator 运行 FedBuff。"""
    t0 = _time.time()

    n_sats = sim.num_satellites
    n_gs = sim.num_ground_stations

    scheduler = CommunicationScheduler(sim)
    trainer = AsyncTrainer(
        local_epochs=local_epochs, batch_size=batch_size,
        learning_rate=learning_rate, device=device,
    )
    selector = AsyncSelector(min_clients=1)
    aggregator = BufferAggregator(
        buffer_size=buffer_size, staleness_weight=staleness_weight,
        server_learning_rate=server_learning_rate,
    )
    evaluator = StandardEvaluator(device=device)

    config = FLConfig(
        algorithm="fedbuff", num_rounds=num_rounds, num_clients=n_sats,
        timeslots_per_round=10,  # sufficient for async window
        fraction=1.0, local_epochs=local_epochs, batch_size=batch_size,
        learning_rate=learning_rate, buffer_size=buffer_size,
        staleness_weight=staleness_weight,
        server_learning_rate=server_learning_rate,
        async_eval_every=async_eval_every,
        device=device, seed=seed,
        time_model="slot", time_model_kwargs={"slots_per_epoch": 1},
        early_stop_acc=0.99, limit_to_sim_window=True,
        partition_strategy="probability", class_probability=class_probability,
        preference_mode="class_balanced", preferred_clients_per_class=1,
        sample_cap_strategy="preserve", data_dir=data_dir,
    )

    runner = FLRunner(config, selector, trainer, aggregator, evaluator, scheduler=scheduler)

    if verbose:
        print(f"  [FedBuff] buffer_size={buffer_size}, staleness_weight={staleness_weight}, "
              f"server_lr={server_learning_rate}")

    history = runner.run(
        dataset_name=dataset, iid=False, alpha=0.5,
        classes_per_client=2, max_samples_per_client=1000,
        partition_strategy="probability", class_probability=class_probability,
        preference_mode="class_balanced", preferred_clients_per_class=1,
        sample_cap_strategy="preserve", data_dir=data_dir,
        verbose=False,
    )

    history_dict = []
    total_arrivals = 0
    stalenesses = []

    for h in history:
        extra = getattr(h, "extra", {}) or {}
        history_dict.append({
            "round": h.round_num,
            "timeslot": getattr(h, "timeslot", h.round_num),
            "accuracy": h.eval_metrics.get("accuracy", 0) if h.eval_metrics else None,
            "loss": h.eval_metrics.get("loss", 0) if h.eval_metrics else None,
            "num_clients": getattr(h, "num_clients", None),
            "staleness": extra.get("staleness", []),
            "mean_staleness": extra.get("mean_staleness", 0.0),
            "total_arrivals": extra.get("total_arrivals", 0),
        })
        if extra.get("staleness"):
            stalenesses.extend(extra["staleness"])
        if extra.get("total_arrivals"):
            total_arrivals = extra["total_arrivals"]

    elapsed = _time.time() - t0
    trained = [r for r in history_dict if r["round"] >= 0 and r["accuracy"] is not None]
    valid_accs = [r["accuracy"] for r in trained if r["accuracy"] is not None]

    result = RunResult(
        name="FedBuff", algorithm="fedbuff",
        history=history_dict, elapsed_sec=elapsed,
        final_acc=valid_accs[-1] if valid_accs else 0,
        max_acc=max(valid_accs) if valid_accs else 0,
        final_timeslot=trained[-1]["timeslot"] if trained else 0,
        total_server_updates=len(trained),
        total_client_updates=total_arrivals,
        mean_staleness=float(np.mean(stalenesses)) if stalenesses else 0.0,
    )

    if verbose:
        print(f"  [FedBuff] {len(trained)} updates, max_acc={result.max_acc:.4f}, "
              f"final_ts={result.final_timeslot}, staleness_mean={result.mean_staleness:.2f}, "
              f"{elapsed:.1f}s")

    return result


# ── 可视化 ─────────────────────────────────────────────────────

def plot_comparison(results: list[RunResult], output_dir: str) -> None:
    if not HAS_MPL:
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Accuracy vs Server Update ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = {"FedAvg": "#3498db", "FedProx": "#e74c3c", "FedBuff": "#2ecc71"}

    for r in results:
        trained = [d for d in r.history if d["round"] >= 0 and d["accuracy"] is not None]
        if not trained:
            continue
        x = list(range(1, len(trained) + 1))
        y = [d["accuracy"] for d in trained]
        ax1.plot(x, y, color=colors.get(r.name, "#95a5a6"), linewidth=1.5,
                 label=f"{r.name} (max={r.max_acc:.3f})", alpha=0.9)

    ax1.set_xlabel("Server Update")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy vs Server Updates")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9, loc="lower right")

    # ── 2. Accuracy vs Timeslot ──
    for r in results:
        trained = [d for d in r.history if d["round"] >= 0 and d["accuracy"] is not None]
        if not trained:
            continue
        x = [d["timeslot"] for d in trained]
        y = [d["accuracy"] for d in trained]
        ax2.plot(x, y, color=colors.get(r.name, "#95a5a6"), linewidth=1.5,
                 label=f"{r.name} (ts={r.final_timeslot})", alpha=0.9)

    ax2.set_xlabel("Simulated Timeslot")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs Simulated Timeslots\n(FedBuff completes in fewer timeslots)")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc="lower right")

    fig.suptitle("SpaceFL — FedAvg vs FedProx vs FedBuff", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150)
    plt.close(fig)

    # ── 3. Summary bars ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = [r.name for r in results]

    bar_colors = [colors.get(n, "#95a5a6") for n in names]

    # Left: Max accuracy
    axes[0].bar(names, [r.max_acc for r in results], color=bar_colors, edgecolor="white")
    for i, v in enumerate([r.max_acc for r in results]):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Max Accuracy")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis="y", alpha=0.3)

    # Middle: Final timeslot
    axes[1].bar(names, [r.final_timeslot for r in results], color=bar_colors, edgecolor="white")
    for i, v in enumerate([r.final_timeslot for r in results]):
        axes[1].text(i, v + 2, str(v), ha="center", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Final Timeslot")
    axes[1].set_title("Lower is faster (FedBuff advantage)")
    axes[1].grid(axis="y", alpha=0.3)

    # Right: Client updates
    axes[2].bar(names, [r.total_client_updates for r in results], color=bar_colors, edgecolor="white")
    for i, v in enumerate([r.total_client_updates for r in results]):
        axes[2].text(i, v + 1, str(v), ha="center", fontsize=9, fontweight="bold")
    axes[2].set_ylabel("Total Client Updates")
    axes[2].set_title("Client training events")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("SpaceFL — Three-Algorithm Comparison Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "summary_bars.png"), dpi=150)
    plt.close(fig)

    print("  [图表] accuracy_comparison.png + summary_bars.png")


# ── 主流程 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SpaceFL 三算法对比实验")
    parser.add_argument("--gs", type=int, default=5, help="地面站数量")
    parser.add_argument("--sats", type=int, default=6, help="卫星数量")
    parser.add_argument("--rounds", type=int, default=50, help="服务端更新次数")
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--mu", type=float, default=0.1, help="FedProx μ")
    parser.add_argument("--buffer-size", type=int, default=3, help="FedBuff buffer size K")
    parser.add_argument("--server-lr", type=float, default=1.0, help="FedBuff server learning rate")
    parser.add_argument("--staleness-weight", action="store_true", default=True)
    parser.add_argument("--no-staleness-weight", dest="staleness_weight", action="store_false")
    parser.add_argument("--sim-hours", type=float, default=48.0, help="模拟时长/小时")
    parser.add_argument("--altitude", type=float, default=500.0)
    parser.add_argument("--inclination", type=float, default=53.0)
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", default="compare_three_output")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--skip-fedavg", action="store_true", help="跳过 FedAvg")
    parser.add_argument("--skip-fedprox", action="store_true", help="跳过 FedProx")
    parser.add_argument("--skip-fedbuff", action="store_true", help="跳过 FedBuff")
    parser.add_argument("--async-eval-every", type=int, default=2,
                        help="FedBuff 每 N 次聚合评估一次")
    args = parser.parse_args()

    quiet = args.quiet

    if not quiet:
        print("=" * 70)
        print("  SpaceFL — 三算法对比实验")
        print("=" * 70)
        print(f"  地面站: {args.gs}  |  卫星: {args.sats}  |  轮次: {args.rounds}")
        print(f"  数据集: {args.dataset} (non-IID, probability+class_balanced)")
        print(f"  模拟:   {args.sim_hours}h  |  设备: {args.device}")
        print(f"  FedProx μ={args.mu}  |  FedBuff K={args.buffer_size}, "
              f"server_lr={args.server_lr}, staleness={args.staleness_weight}")
        print("=" * 70)

    # ── 创建共享 OrbitSimulator ──
    body = CelestialBody.earth()
    orbits = create_uniform_orbits(body, args.sats, args.altitude, args.inclination)
    gs_network = create_gs_network(args.gs)
    num_timeslots = int(args.sim_hours * 60 / 1.0)

    sim = OrbitSimulator(
        body=body, orbits=orbits, ground_station_network=gs_network,
        num_timeslots=num_timeslots, timeslot_duration_min=1.0,
        backend="kepler", contact_mode="simple", verbose=False,
    )

    if not quiet:
        print(f"\n  {sim.summary()}")

    results: list[RunResult] = []

    common_kwargs = dict(
        sim=sim, num_rounds=args.rounds, local_epochs=args.local_epochs,
        batch_size=args.batch_size, learning_rate=args.lr,
        device=args.device, seed=args.seed, dataset=args.dataset,
        class_probability=0.8, data_dir=args.data_dir,
        verbose=not quiet,
    )

    # ── FedAvg ──
    if not args.skip_fedavg:
        if not quiet:
            print(f"\n{'─' * 50}\n  [1] FedAvg\n{'─' * 50}")
        r_avg = run_sync_algo(name="FedAvg", algorithm="fedavg", mu=0.0, **common_kwargs)
        results.append(r_avg)

    # ── FedProx ──
    if not args.skip_fedprox:
        if not quiet:
            print(f"\n{'─' * 50}\n  [2] FedProx (μ={args.mu})\n{'─' * 50}")
        r_prox = run_sync_algo(name="FedProx", algorithm="fedprox", mu=args.mu, **common_kwargs)
        results.append(r_prox)

    # ── FedBuff ──
    if not args.skip_fedbuff:
        if not quiet:
            print(f"\n{'─' * 50}\n  [3] FedBuff (K={args.buffer_size})\n{'─' * 50}")
        r_buff = run_fedbuff(
            sim=sim, num_rounds=args.rounds, local_epochs=args.local_epochs,
            batch_size=args.batch_size, learning_rate=args.lr,
            buffer_size=args.buffer_size, staleness_weight=args.staleness_weight,
            server_learning_rate=args.server_lr,
            device=args.device, seed=args.seed, dataset=args.dataset,
            class_probability=0.8, data_dir=args.data_dir,
            async_eval_every=args.async_eval_every,
            verbose=not quiet,
        )
        results.append(r_buff)

    # ── 输出 ──
    os.makedirs(args.output, exist_ok=True)

    for r in results:
        fname = r.name.lower().replace(" ", "_")
        with open(os.path.join(args.output, f"{fname}_history.json"), "w", encoding="utf-8") as f:
            json.dump(r.history, f, ensure_ascii=False, indent=2)

    summary = {
        "config": {
            "gs_count": args.gs, "sat_count": args.sats,
            "num_rounds": args.rounds, "local_epochs": args.local_epochs,
            "batch_size": args.batch_size, "learning_rate": args.lr,
            "fedprox_mu": args.mu, "fedbuff_K": args.buffer_size,
            "server_lr": args.server_lr, "staleness_weight": args.staleness_weight,
            "sim_hours": args.sim_hours, "device": args.device, "seed": args.seed,
        },
        "contact_rate": sim.stats.get("contact_rate", 0),
        "results": [],
    }

    for r in results:
        summary["results"].append({
            "name": r.name, "algorithm": r.algorithm,
            "max_acc": round(r.max_acc, 4),
            "final_acc": round(r.final_acc, 4),
            "final_timeslot": r.final_timeslot,
            "server_updates": r.total_server_updates,
            "client_updates": r.total_client_updates,
            "mean_staleness": round(r.mean_staleness, 4),
            "elapsed_sec": round(r.elapsed_sec, 1),
        })

    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if HAS_MPL:
        plot_comparison(results, args.output)

    # ── 终端报告 ──
    if not quiet:
        print(f"\n{'=' * 70}")
        print("  三算法对比结果")
        print(f"{'=' * 70}")
        header = f"  {'算法':<12} {'最高Acc':>8} {'最终Acc':>8} {'最终时隙':>8} {'客户端更新':>10} {'陈旧度':>8}"
        print(header)
        print(f"  {'─' * 62}")
        for r in results:
            staleness_str = f"{r.mean_staleness:.2f}" if r.algorithm == "fedbuff" else "—"
            print(f"  {r.name:<12} {r.max_acc:>8.4f} {r.final_acc:>8.4f} "
                  f"{r.final_timeslot:>8} {r.total_client_updates:>10} {staleness_str:>8}")

        # 时隙效率对比
        if len(results) >= 2:
            sync_results = [r for r in results if r.algorithm in ("fedavg", "fedprox")]
            buff_results = [r for r in results if r.algorithm == "fedbuff"]
            if sync_results and buff_results:
                avg_sync_ts = np.mean([r.final_timeslot for r in sync_results])
                buff_ts = buff_results[0].final_timeslot
                speedup = avg_sync_ts / max(buff_ts, 1)
                print(f"\n  时隙效率: FedBuff 用时隙 {buff_ts} vs 同步平均 {avg_sync_ts:.0f} "
                      f"({speedup:.1f}× 更快)")

        print(f"\n  输出: {os.path.abspath(args.output)}/")
        print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
