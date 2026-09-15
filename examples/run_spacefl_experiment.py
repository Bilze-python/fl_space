"""
SpaceFL 太空联邦学习实验 — 异构轨道 + 多地面站对比

实验设计：
    - 10 颗卫星，不同轨道高度 (350-800 km)，同一轨道面
    - 地面站：1 / 3 / 5 三级对比
    - FedProx 同步训练，300 轮，准确率 > 90% 早停
    - 标准 FL 基线对比（无轨道约束）

用法：
    python examples/run_spacefl_experiment.py [选项]
    python examples/run_spacefl_experiment.py --gs-counts 1 3 5 --rounds 300 --output ./results

CLI 快速启动：
    fl-space experiment --sats 10 --gs 1 3 5 --rounds 300 --dataset mnist
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
import sys
import time as _time
from typing import Any

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── SpaceFL 依赖 ──────────────────────────────────────────────

from fl_space.environment import (
    CelestialBody,
    GroundStation,
    GroundStationNetwork,
)
from fl_space.fl.fedprox import create_fedprox_components
from fl_space.fl.runner import FLRunner
from fl_space.fl.scheduler import CommunicationScheduler
from fl_space.fl.server import FLConfig, FLServer
from fl_space.orbit import (
    KeplerOrbit,
    create_circular_orbit,
)
from fl_space.simulator import OrbitSimulator
from fl_space.utils.viz import (
    get_contact_statistics,
    plot_contact_heatmap,
    plot_ground_station_map,
    save_experiment_report,
)

# ── 预定义地面站组 ────────────────────────────────────────────

# 全球分布的 5 个地面站（覆盖不同经度）
ALL_GROUND_STATIONS = [
    GroundStation("Beijing",    39.9,  116.4, 0.05),   # 东亚
    GroundStation("Svalbard",   78.2,   15.6, 0.02),   # 北极
    GroundStation("Santiago",  -33.4,  -70.7, 0.5),    # 南美
    GroundStation("Singapore",   1.3,  103.8, 0.02),    # 东南亚
    GroundStation("Washington", 38.9,  -77.0, 0.05),   # 北美
]

def get_gs_network(n_gs: int) -> GroundStationNetwork:
    """获取指定数量的地面站网络（从预定义列表中选取）。"""
    selected = ALL_GROUND_STATIONS[:min(n_gs, len(ALL_GROUND_STATIONS))]
    return GroundStationNetwork(selected)


# ── 异构轨道配置 ──────────────────────────────────────────────

# 10 颗卫星在不同高度（350-800 km），同一倾角，均匀分布于一个轨道面
HETERO_ALTITUDES_KM = [350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
HETERO_INCLINATION_DEG = 53.0   # Starlink 风格倾角
HETERO_RAAN_DEG = 0.0            # 同一轨道面
HETERO_NUM_SATELLITES = 10       # 卫星总数


def create_heterogeneous_orbits(
    body: CelestialBody | None = None,
    altitudes_km: list[float] | None = None,
    inclination_deg: float = 53.0,
    raan_deg: float = 0.0,
) -> list[KeplerOrbit]:
    """
    创建异构轨道：不同高度的卫星在同一倾角和 RAAN 的轨道面。

    不同高度 → 不同轨道周期 → 自然的时间差 → 多样化的接触窗口。

    Parameters
    ----------
    body : CelestialBody
        中心天体。
    altitudes_km : list[float]
        各卫星的轨道高度（km），默认 [350, 400, ..., 800]。
    inclination_deg : float
        轨道倾角。
    raan_deg : float
        升交点赤经。

    Returns
    -------
    list[KeplerOrbit]
        轨道对象列表。
    """
    if body is None:
        body = CelestialBody.earth()
    if altitudes_km is None:
        altitudes_km = HETERO_ALTITUDES_KM

    orbits = []
    n = len(altitudes_km)
    for i, alt in enumerate(altitudes_km):
        # 均匀分布真近点角，制造初始相位差
        true_anomaly = i * (360.0 / n)
        orb = create_circular_orbit(
            altitude_km=alt,
            inclination_deg=inclination_deg,
            raan_deg=raan_deg,
            true_anomaly_deg=true_anomaly,
            body=body,
        )
        orbits.append(orb)

    return orbits


# ── 实验运行器 ────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """单组实验结果。"""
    name: str
    gs_count: int
    config: dict = field(default_factory=dict)
    spacefl_history: list[dict] = field(default_factory=list)
    baseline_history: list[dict] = field(default_factory=list)
    contact_stats: dict = field(default_factory=dict)
    sim: Any = None
    elapsed_sec: float = 0.0


def run_spacefl_experiment(
    sim: OrbitSimulator,
    fl_config: FLConfig,
    dataset: str = "mnist",
    iid: bool = False,
    alpha: float = 0.3,
    verbose: bool = True,
) -> tuple[list[dict], FLServer]:
    """
    运行带轨道约束的 SpaceFL 实验。

    Parameters
    ----------
    sim : OrbitSimulator
        轨道模拟器。
    fl_config : FLConfig
        FL 配置。
    dataset : str
        数据集名称。
    iid : bool
        True 为 IID，False 为 non-IID (Dirichlet)，默认 non-IID。
    alpha : float
        Dirichlet 参数（越小越偏斜），默认 0.3。
    verbose : bool
        是否打印进度。

    Returns
    -------
    tuple[list[dict], FLServer]
        (训练历史字典列表, FLServer 实例)
    """
    # 创建通信调度器
    scheduler = CommunicationScheduler(sim)

    # SpaceFL 组件：min_clients=1（有多少可连的就选多少）
    components = create_fedprox_components(
        fraction=fl_config.fraction,
        min_clients=1,  # SpaceFL: 轨道约束下选中所有可连接卫星
        local_epochs=fl_config.local_epochs,
        batch_size=fl_config.batch_size,
        learning_rate=fl_config.learning_rate,
        mu=fl_config.mu,
        device=fl_config.device,
        seed=fl_config.seed,
    )

    runner = FLRunner(fl_config, *components, scheduler=scheduler)

    if verbose:
        print(f"\n  [SpaceFL] 启动训练 (GS={sim.num_ground_stations}, "
              f"Sats={sim.num_satellites})")

    history = runner.run(
        dataset_name=dataset,
        iid=iid,
        alpha=alpha,
        verbose=verbose,
    )

    return runner.history_dict, runner._server


def run_baseline_fl(
    fl_config: FLConfig,
    dataset: str = "mnist",
    iid: bool = False,
    alpha: float = 0.3,
    verbose: bool = True,
) -> list[dict]:
    """
    运行标准 FL 基线（无轨道约束，所有客户端始终可通信）。

    Parameters
    ----------
    fl_config : FLConfig
        FL 配置。
    dataset : str
        数据集名称。
    iid : bool
        True 为 IID，False 为 non-IID (Dirichlet)，默认 non-IID。
    alpha : float
        Dirichlet 参数。
    verbose : bool
        是否打印进度。

    Returns
    -------
    list[dict]
        训练历史字典列表。
    """
    # 基线FL组件：所有客户端始终可通信，选中全部
    components = create_fedprox_components(
        fraction=fl_config.fraction,
        min_clients=fl_config.num_clients,  # 基线: 全部客户端参与
        local_epochs=fl_config.local_epochs,
        batch_size=fl_config.batch_size,
        learning_rate=fl_config.learning_rate,
        mu=fl_config.mu,
        device=fl_config.device,
        seed=fl_config.seed,
    )

    # 无 scheduler → 始终可通信
    runner = FLRunner(fl_config, *components, scheduler=None)

    if verbose:
        print("\n  [Baseline FL] 启动训练 (无轨道约束)")

    history = runner.run(
        dataset_name=dataset,
        iid=iid,
        alpha=alpha,
        verbose=verbose,
    )

    return runner.history_dict


def run_experiment_suite(
    gs_counts: list[int] = (1, 3, 5),
    num_satellites: int = 10,
    num_rounds: int = 300,
    altitudes_km: list[float] | None = None,
    inclination_deg: float = 53.0,
    dataset: str = "mnist",
    iid: bool = False,
    alpha: float = 0.3,
    device: str = "cpu",
    local_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    mu: float = 0.01,
    early_stop_acc: float = 0.90,
    num_train_workers: int = 1,
    num_workers: int = 0,
    sim_hours: float = 168.0,  # 7 天
    timeslot_duration_min: float = 1.0,
    seed: int = 42,
    output_dir: str = "experiment_output",
    verbose: bool = True,
    lang: str = "en",
) -> dict[str, Any]:
    """
    运行完整实验套件：多组地面站 + 基线对比。

    Parameters
    ----------
    gs_counts : list[int]
        地面站数量列表（如 [1, 3, 5]）。
    num_satellites : int
        卫星数量。
    num_rounds : int
        最大训练轮次。
    altitudes_km : list[float] | None
        各卫星高度列表。
    inclination_deg : float
        轨道倾角。
    dataset : str
        数据集。
    device : str
        计算设备。
    local_epochs : int
        本地训练 epoch。
    batch_size : int
        训练 batch size。
    learning_rate : float
        学习率。
    mu : float
        FedProx proximal term 系数。
    early_stop_acc : float
        早停准确率阈值。
    num_train_workers : int
        并行训练线程数。
    num_workers : int
        DataLoader 并行进程数。
    sim_hours : float
        初始预计算时长（小时）。实际模拟时间由 FL 训练动态决定，
        超出此范围时自动按需扩展接触矩阵。
    seed : int
        随机种子。
    output_dir : str
        输出目录。
    verbose : bool
        是否打印详细信息。

    Returns
    -------
    dict
        完整实验结果（JSON 可序列化）。
    """
    os.makedirs(output_dir, exist_ok=True)

    if altitudes_km is None:
        altitudes_km = HETERO_ALTITUDES_KM[:num_satellites]

    body = CelestialBody.earth()
    num_timeslots = int(sim_hours * 60 / timeslot_duration_min)

    # ── FL 基础配置 ──
    base_fl_config = {
        "algorithm": "fedprox",
        "num_rounds": num_rounds,
        "num_clients": num_satellites,
        "fraction": 1.0,          # 全部可用客户端参与
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "mu": mu,
        "device": device,
        "seed": seed,
        "time_model": "slot",
        "time_model_kwargs": {"slots_per_epoch": 1},
        "num_workers": num_workers,
        "num_train_workers": num_train_workers,
        "early_stop_acc": early_stop_acc,
    }

    all_results: list[ExperimentResult] = []

    if verbose:
        print("=" * 70)
        print("  SpaceFL 太空联邦学习实验套件")
        print("=" * 70)
        print(f"  卫星: {num_satellites} (异构轨道 {altitudes_km[0]}-{altitudes_km[-1]} km)")
        print(f"  倾角: {inclination_deg}°")
        print(f"  模拟时长: {sim_hours}h ({num_timeslots} slots)")
        print(f"  地面站组: {gs_counts}")
        print(f"  数据集: {dataset}")
        print(f"  最大轮次: {num_rounds}")
        print(f"  早停阈值: {early_stop_acc}")
        print(f"  设备: {device}")
        print(f"  并行训练: {num_train_workers} 线程")
        print("=" * 70)

    # ── 基线 FL: 无轨道约束 ──
    if verbose:
        print("\n" + "─" * 50)
        print("  [基线] 标准 FL (无轨道约束)")
        print("─" * 50)

    t0 = _time.time()
    baseline_fl_config = FLConfig(**{**base_fl_config, "num_rounds": min(num_rounds, 100)})
    baseline_history = run_baseline_fl(
        baseline_fl_config, dataset=dataset, iid=iid, alpha=alpha, verbose=verbose,
    )
    baseline_elapsed = _time.time() - t0

    if verbose:
        final_acc = baseline_history[-1].get("accuracy", 0) if baseline_history else 0
        print(f"\n  [基线] 完成: {len(baseline_history)} 轮, "
              f"准确率 {final_acc:.4f}, 耗时 {baseline_elapsed:.1f}s")

    # ── 各组 SpaceFL 实验 ──
    for gs_count in gs_counts:
        if verbose:
            print("\n" + "─" * 50)
            print(f"  [SpaceFL] 地面站: {gs_count}")
            print("─" * 50)

        t0 = _time.time()

        # 创建异构轨道
        orbits = create_heterogeneous_orbits(
            body=body,
            altitudes_km=altitudes_km,
            inclination_deg=inclination_deg,
        )

        # 创建地面站网络
        gs_network = get_gs_network(gs_count)

        # 创建模拟器
        sim = OrbitSimulator(
            body=body,
            orbits=orbits,
            ground_station_network=gs_network,
            num_timeslots=num_timeslots,
            timeslot_duration_min=timeslot_duration_min,
            backend="kepler",
            contact_mode="simple",  # 内存友好
            verbose=False,
        )

        if verbose:
            print(f"\n  {sim.summary()}")

        # 接触统计
        contact_stats = get_contact_statistics(sim)

        # 生成热力图
        heatmap_path = os.path.join(output_dir, f"heatmap_gs{gs_count}.png")
        plot_contact_heatmap(sim, heatmap_path,
                             title=f"Contact Matrix (GS={gs_count}, Sats={num_satellites})",
                             lang=lang)

        # 地面站分布图
        gs_map_path = os.path.join(output_dir, f"gs_map_gs{gs_count}.png")
        plot_ground_station_map(sim, gs_map_path,
                                title=f"Ground Stations (GS={gs_count})",
                                show_tracks=True,
                                lang=lang)

        # 运行 SpaceFL
        fl_config = FLConfig(**base_fl_config)
        spacefl_history, server = run_spacefl_experiment(
            sim, fl_config, dataset=dataset, iid=iid, alpha=alpha, verbose=verbose,
        )

        elapsed = _time.time() - t0
        final_acc = spacefl_history[-1].get("accuracy", 0) if spacefl_history else 0

        result = ExperimentResult(
            name=f"gs{gs_count}",
            gs_count=gs_count,
            config={
                "num_satellites": num_satellites,
                "altitudes_km": altitudes_km,
                "inclination_deg": inclination_deg,
                "gs_count": gs_count,
                "gs_names": [gs.name for gs in gs_network],
                "num_rounds": len(spacefl_history),
                "num_timeslots_pre": num_timeslots,
                "sim_hours_pre": sim_hours,
                "total_timeslots": sim.num_timeslots if hasattr(sim, 'num_timeslots') else num_timeslots,
                "total_virtual_hours": sim.num_timeslots * timeslot_duration_min / 60.0 if hasattr(sim, 'num_timeslots') else sim_hours,
            },
            spacefl_history=spacefl_history,
            contact_stats=contact_stats,
            sim=sim,
            elapsed_sec=elapsed,
        )
        all_results.append(result)

        if verbose:
            print(f"\n  [SpaceFL GS={gs_count}] 完成: {len(spacefl_history)} 轮, "
                  f"准确率 {final_acc:.4f}, 耗时 {elapsed:.1f}s")

    # ── 组装最终报告 ──
    report = {
        "config": {
            "num_satellites": num_satellites,
            "altitudes_km": altitudes_km,
            "inclination_deg": inclination_deg,
            "dataset": dataset,
            "algorithm": "fedprox",
            "max_rounds": num_rounds,
            "early_stop_acc": early_stop_acc,
            "device": device,
            "sim_hours": sim_hours,
            "timeslot_duration_min": timeslot_duration_min,
        },
        "baseline": {
            "rounds": len(baseline_history),
            "final_accuracy": baseline_history[-1].get("accuracy", 0) if baseline_history else 0,
            "history": baseline_history,
        },
        "experiments": [
            {
                "name": r.name,
                "gs_count": r.gs_count,
                "config": r.config,
                "spacefl_history": r.spacefl_history,
                "baseline_history": baseline_history,
                "contact_stats": r.contact_stats,
                "elapsed_sec": r.elapsed_sec,
            }
            for r in all_results
        ],
    }

    # 保存报告
    report_path = save_experiment_report(report, output_dir)

    # 生成 GS 对比准确率图
    if all_results:
        _plot_gs_comparison(all_results, baseline_history, output_dir)

    if verbose:
        print("\n" + "=" * 70)
        print("  实验完成!")
        print(f"  报告: {report_path}")
        print(f"  输出: {output_dir}/")
        print("=" * 70)

    return report


def _plot_gs_comparison(
    results: list[ExperimentResult],
    baseline_history: list[dict],
    output_dir: str,
) -> None:
    """绘制多组地面站准确率对比图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

        for i, r in enumerate(results):
            color = colors[i % len(colors)]
            sh = r.spacefl_history
            rounds = [h.get("round", j) for j, h in enumerate(sh)]
            acc = [h.get("accuracy", 0) for h in sh]
            ax1.plot(rounds, acc, "-o", color=color, markersize=2, linewidth=1,
                     label=f"SpaceFL GS={r.gs_count}")

        # 基线
        if baseline_history:
            br = [h.get("round", j) for j, h in enumerate(baseline_history)]
            ba = [h.get("accuracy", 0) for h in baseline_history]
            ax1.plot(br, ba, "k--", linewidth=1.5, label="Standard FL (no orbit)")

        ax1.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5)
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy vs Rounds")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)

        # 按时间槽
        for i, r in enumerate(results):
            color = colors[i % len(colors)]
            sh = r.spacefl_history
            ts = [h.get("timeslot", h.get("round", j) * 10) for j, h in enumerate(sh)]
            acc = [h.get("accuracy", 0) for h in sh]
            ax2.plot(ts, acc, "-o", color=color, markersize=2, linewidth=1,
                     label=f"SpaceFL GS={r.gs_count}")

        if baseline_history:
            ts_base = [h.get("timeslot", h.get("round", i) * 10) for i, h in enumerate(baseline_history)]
            ax2.plot(ts_base, ba, "k--", linewidth=1.5, label="Standard FL")

        ax2.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5)
        ax2.set_xlabel("Timeslot")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy vs Virtual Time")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)

        fig.suptitle("SpaceFL: Multi-GS Accuracy Comparison", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "gs_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  [警告] GS 对比图生成失败: {e}")


# ── CLI 入口 ──────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """构建实验 CLI 参数解析器。"""
    p = argparse.ArgumentParser(
        description="SpaceFL 太空联邦学习实验 — 异构轨道 + 多地面站对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python examples/run_spacefl_experiment.py --gs-counts 1 3 5 --rounds 300
  python examples/run_spacefl_experiment.py --sats 10 --gs 1 3 --device cuda --workers 4
  python examples/run_spacefl_experiment.py --dataset cifar10 --epochs 5 --sim-hours 336
        """,
    )
    p.add_argument("--sats", type=int, default=10, help="卫星数量 (默认: 10)")
    p.add_argument("--gs-counts", "--gs", type=int, nargs="+", default=[1, 3, 5],
                   help="地面站数量列表 (默认: 1 3 5)")
    p.add_argument("--rounds", type=int, default=300, help="最大训练轮次 (默认: 300)")
    p.add_argument("--epochs", type=int, default=3, help="本地训练 epoch (默认: 3)")
    p.add_argument("--batch-size", type=int, default=32, help="batch size (默认: 32)")
    p.add_argument("--lr", type=float, default=0.01, help="学习率 (默认: 0.01)")
    p.add_argument("--mu", type=float, default=0.01, help="FedProx mu (默认: 0.01)")
    p.add_argument("--iid", action="store_true", help="使用 IID 数据分配 (默认 non-IID)")
    p.add_argument("--alpha", type=float, default=0.3,
                   help="non-IID Dirichlet alpha (默认: 0.3, 越小越偏斜)")
    p.add_argument("--dataset", choices=["mnist", "fashion_mnist", "cifar10"],
                   default="mnist", help="数据集 (默认: mnist)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                   help="计算设备 (默认: cpu)")
    p.add_argument("--early-stop", type=float, default=0.90,
                   help="早停准确率阈值 (默认: 0.90)")
    p.add_argument("--train-workers", type=int, default=1,
                   help="并行训练线程数 (默认: 1)")
    p.add_argument("--data-workers", type=int, default=0,
                   help="DataLoader 并行进程数 (默认: 0)")
    p.add_argument("--inclination", type=float, default=53.0,
                   help="轨道倾角° (默认: 53)")
    p.add_argument("--sim-hours", type=float, default=168.0,
                   help="模拟时长/小时 (默认: 168 = 7天)")
    p.add_argument("--timeslot-min", type=float, default=1.0,
                   help="每 timeslot 分钟数 (默认: 1.0)")
    p.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    p.add_argument("--output", "-o", type=str, default="experiment_output",
                   help="输出目录 (默认: experiment_output)")
    p.add_argument("--altitudes", type=float, nargs="+", default=None,
                   help="自定义卫星高度列表 km (默认: 350-800 均匀分布)")
    p.add_argument("--quiet", "-q", action="store_true", help="安静模式")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    altitudes = args.altitudes
    if altitudes is None:
        # 均匀分布在 350-800 km
        altitudes = [
            350 + i * (800 - 350) / (args.sats - 1)
            for i in range(args.sats)
        ] if args.sats > 1 else [500.0]

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 保存实验配置
    config_path = os.path.join(output_dir, "experiment_config.json")
    config = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    config["altitudes_km"] = altitudes
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)
    if not args.quiet:
        print(f"实验配置: {config_path}")

    # 运行实验
    t_start = _time.time()
    report = run_experiment_suite(
        gs_counts=args.gs_counts,
        num_satellites=args.sats,
        num_rounds=args.rounds,
        altitudes_km=altitudes,
        inclination_deg=args.inclination,
        dataset=args.dataset,
        iid=args.iid,
        alpha=args.alpha,
        device=args.device,
        local_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mu=args.mu,
        early_stop_acc=args.early_stop,
        num_train_workers=args.train_workers,
        num_workers=args.data_workers,
        sim_hours=args.sim_hours,
        timeslot_duration_min=args.timeslot_min,
        seed=args.seed,
        output_dir=output_dir,
        verbose=not args.quiet,
    )

    total_elapsed = _time.time() - t_start
    print(f"\n总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
