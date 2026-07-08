"""
SpaceFL 快速测试 — 10异构卫星 + 5地面站 + FedProxSat (自适应μ, non-IID, 2类/卫星)

用法:
    # 默认: 自适应 μ (FedProxSat)
    python examples/quick_test.py

    # 固定 μ (传统 FedProx)
    python examples/quick_test.py --mu 0.1 --no-adaptive

    # 自定义自适应范围
    python examples/quick_test.py --mu 0.01 --mu-min 0.001 --mu-max 0.5

每次运行自动保存：
    {output_dir}/
    ├── experiment_config.json  — 实验配置
    ├── experiment_results.json — 完整训练历史 (含 μ 变化)
    ├── accuracy_curve.png      — 准确率曲线
    ├── contact_heatmap.png     — 接触热力图
    └── gs_map.png              — 地面站地图
"""
import argparse
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.run_spacefl_experiment import (
    create_heterogeneous_orbits,
    get_gs_network,
)
from fl_space.environment import CelestialBody
from fl_space.fl.fedavg import CappedSelector, StandardEvaluator, SyncWeightedAggregator
from fl_space.fl.fedprox import AdaptiveProximalTrainer, ProximalTrainer
from fl_space.fl.runner import FLRunner
from fl_space.fl.scheduler import CommunicationScheduler
from fl_space.fl.server import FLConfig
from fl_space.simulator import OrbitSimulator
from fl_space.utils import (
    plot_contact_heatmap,
    plot_ground_station_map,
)


# ── 命令行参数 ──
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SpaceFL 快速测试 — FedProxSat 自适应μ")
    p.add_argument("--output", "-o", default="results", help="输出目录 (默认: results)")
    p.add_argument("--mu", type=float, default=0.01, help="基础 μ / 固定 μ (默认: 0.01)")
    p.add_argument("--mu-min", type=float, default=0.001,
                   help="自适应 μ 下限 (默认: 0.001)")
    p.add_argument("--mu-max", type=float, default=1.0,
                   help="自适应 μ 上限 (默认: 1.0)")
    p.add_argument("--oscillation-threshold", type=float, default=0.1,
                   help="震荡阈值，超过触发 μ↑ (默认: 0.1)")
    p.add_argument("--stability-threshold", type=float, default=0.03,
                   help="稳定阈值，低于触发 μ↓ (默认: 0.03)")
    p.add_argument("--no-adaptive", action="store_true",
                   help="禁用自适应 μ，使用固定 μ (传统 FedProx)")
    p.add_argument("--rounds", type=int, default=300, help="最大轮数 (默认: 300)")
    p.add_argument("--epochs", type=int, default=2, help="本地epoch (默认: 2)")
    p.add_argument("--early-stop", type=float, default=0.9, help="早停阈值 (默认: 0.9)")
    p.add_argument("--gs", type=int, default=5, help="地面站数 (默认: 5)")
    p.add_argument("--lang", choices=["en", "zh"], default="en",
                   help="输出图表语言 (默认: en)")
    p.add_argument("--quiet", "-q", action="store_true", help="安静模式")
    return p


def run_quick_test(args: argparse.Namespace) -> int:
    """SpaceFL 快速测试入口，可通过 CLI 或 Python API 调用。

    Parameters
    ----------
    args : argparse.Namespace
        解析后的命令行参数。

    Returns
    -------
    int
        0 表示成功，非 0 表示失败。
    """
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # ── 轨道：10颗异构轨道卫星 (350-800km, 不同周期) ──
    body = CelestialBody.earth()
    orbits = create_heterogeneous_orbits(body)
    gs_count = args.gs
    gs_network = get_gs_network(gs_count)

    sim = OrbitSimulator(
        body=body, orbits=orbits,
        ground_station_network=gs_network,
        num_timeslots=(168 * 60),
        timeslot_duration_min=1.0,
        backend="kepler", contact_mode="simple",
        verbose=False,
    )
    if not args.quiet:
        print(sim.summary())

    # ── 保存实验配置 ──
    exp_config = {
        "timestamp": datetime.now().isoformat(),
        "algorithm": "fedprox" if args.no_adaptive else "FedProxSat (adaptive μ)",
        "dataset": "mnist",
        "distribution": "non-IID (2 classes per satellite, sliding window)",
        "satellites": 10,
        "altitudes_km": "350-800 heterogeneous",
        "inclination": 53.0,
        "ground_stations": gs_count,
        "selector": f"CappedSelector(max_count={gs_count})",
        "mu": args.mu,
        "adaptive_mu": not args.no_adaptive,
        "mu_min": args.mu_min if not args.no_adaptive else None,
        "mu_max": args.mu_max if not args.no_adaptive else None,
        "oscillation_threshold": args.oscillation_threshold if not args.no_adaptive else None,
        "stability_threshold": args.stability_threshold if not args.no_adaptive else None,
        "local_epochs": args.epochs,
        "batch_size": 32,
        "learning_rate": 0.01,
        "max_rounds": args.rounds,
        "early_stop_acc": args.early_stop,
        "device": "cpu",
        "seed": 42,
        "orbit_stats": {
            "contact_rate": sim.stats.get("contact_rate", 0),
            "total_contacts": sim.stats.get("total_contacts", 0),
            "sim_duration_h": 168.0,
            "timeslot_min": 1.0,
        },
    }
    with open(os.path.join(output_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(exp_config, f, ensure_ascii=False, indent=2)

    # ── 生成接触热力图和GS地图 ──
    try:
        plot_contact_heatmap(sim, os.path.join(output_dir, "contact_heatmap.png"),
                             title=f"Contact Heatmap (GS={gs_count}, 10 Sats)",
                             lang=getattr(args, 'lang', 'en'))
        if not args.quiet:
            print(f"[保存] 热力图 → {output_dir}/contact_heatmap.png")
    except Exception as e:
        if not args.quiet:
            print(f"[警告] 热力图生成失败: {e}")

    try:
        plot_ground_station_map(sim, os.path.join(output_dir, "gs_map.png"),
                                title=f"Ground Stations (GS={gs_count})",
                                lang=getattr(args, 'lang', 'en'))
        if not args.quiet:
            print(f"[保存] GS地图 → {output_dir}/gs_map.png")
    except Exception as e:
        if not args.quiet:
            print(f"[警告] GS地图生成失败: {e}")

    # ── FL 配置 ──
    config = FLConfig(
        algorithm="fedprox", num_rounds=args.rounds, num_clients=10,
        fraction=1.0, local_epochs=args.epochs, batch_size=32,
        learning_rate=0.01, mu=args.mu, device="cpu", seed=42,
        time_model="slot", time_model_kwargs={"slots_per_epoch": 1},
        early_stop_acc=args.early_stop, num_train_workers=1, num_workers=2,
    )

    scheduler = CommunicationScheduler(sim)

    selector = CappedSelector(max_count=gs_count, min_clients=1, seed=42)
    if args.no_adaptive:
        trainer = ProximalTrainer(local_epochs=args.epochs, batch_size=32,
                                  learning_rate=0.01, mu=args.mu, device="cpu")
    else:
        trainer = AdaptiveProximalTrainer(
            local_epochs=args.epochs, batch_size=32,
            learning_rate=0.01, base_mu=args.mu,
            mu_min=args.mu_min, mu_max=args.mu_max,
            oscillation_threshold=args.oscillation_threshold,
            stability_threshold=args.stability_threshold,
            device="cpu",
        )
    aggregator = SyncWeightedAggregator(min_updates=1)
    evaluator = StandardEvaluator(device="cpu")

    runner = FLRunner(config, selector, trainer, aggregator, evaluator, scheduler=scheduler)

    mode_str = "FedProxSat (自适应 μ)" if not args.no_adaptive else "FedProx (固定 μ)"
    if not args.quiet:
        print("\n=== SpaceFL 快速测试 ===")
        print(f"  {mode_str}")
        print(f"  GS={gs_count}, Sats=10, 异构轨道(350-800km)")
        if args.no_adaptive:
            print(f"  μ={args.mu} (固定), {args.epochs} epoch/轮, 每卫星2类数字")
        else:
            print(f"  μ_base={args.mu}, μ∈[{args.mu_min}, {args.mu_max}]")
            print(f"  震荡阈值={args.oscillation_threshold}, 稳定阈值={args.stability_threshold}")
            print(f"  {args.epochs} epoch/轮, 每卫星2类数字")
        print(f"  选择器: min(GS, 在线) = min({gs_count}, N)")
        print(f"  输出: {output_dir}/\n")

    history = runner.run(
        dataset_name="mnist", iid=False, classes_per_client=2, verbose=not args.quiet,
    )

    # ── 保存训练历史为 JSON ──
    if history:
        acc = history[-1].eval_metrics.get("accuracy", 0)
        if not args.quiet:
            print(f"\n完成: {len(history)} 轮, 最终准确率 {acc:.4f}")

        # 转换为可序列化格式
        results_data = {
            "config": exp_config,
            "summary": {
                "total_rounds": len(history),
                "final_accuracy": round(acc, 4),
                "max_accuracy": round(max(
                    h.eval_metrics.get("accuracy", 0) for h in history
                ), 4),
                "min_accuracy": round(min(
                    h.eval_metrics.get("accuracy", 0) for h in history
                ), 4),
                "virtual_time_slots": history[-1].timeslot_end if hasattr(history[-1], 'timeslot_end') else None,
            },
            "history": [],
        }

        # 附加自适应 μ 统计
        if hasattr(trainer, 'mu_stats'):
            results_data["mu_stats"] = trainer.mu_stats

        for h in history:
            entry = {
                "round": h.round_num,
                "timeslot_start": getattr(h, "timeslot_start", None),
                "timeslot_end": getattr(h, "timeslot_end", None),
                "accuracy": h.eval_metrics.get("accuracy", 0),
                "loss": h.eval_metrics.get("loss", 0),
                "online_clients": h.num_online if hasattr(h, 'num_online') else None,
                "selected_clients": h.num_selected if hasattr(h, 'num_selected') else None,
                "train_loss": h.train_loss if hasattr(h, 'train_loss') else None,
            }
            results_data["history"].append(entry)

        results_path = os.path.join(output_dir, "experiment_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(f"[保存] 训练历史 → {results_path}")

        # ── 准确率曲线图 ──
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            rounds_list = [h.round_num for h in history]
            acc_list = [h.eval_metrics.get("accuracy", 0) for h in history]

            # 左：准确率 vs 轮次
            ax = axes[0]
            ax.plot(rounds_list, acc_list, "b-o", markersize=2, linewidth=0.8)
            ax.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5, label="90%")
            ax.set_xlabel("Round")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Accuracy vs Rounds (μ={args.mu}, GS={gs_count})")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

            # 标注关键统计
            max_acc = max(acc_list)
            max_round = rounds_list[acc_list.index(max_acc)]
            ax.annotate(f"max={max_acc:.3f} @ R{max_round}",
                        xy=(max_round, max_acc), fontsize=8,
                        color="red", fontweight="bold")

            # 右：准确率分布直方图
            ax = axes[1]
            ax.hist(acc_list, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
            ax.axvline(x=np.mean(acc_list), color="red", linestyle="--",
                       label=f"mean={np.mean(acc_list):.3f}")
            ax.axvline(x=np.median(acc_list), color="orange", linestyle="--",
                       label=f"median={np.median(acc_list):.3f}")
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Accuracy Distribution (σ={np.std(acc_list):.3f})")
            ax.legend(fontsize=8)

            fig.suptitle(f"SpaceFL FedProx — {gs_count}GS × 10 Sats (2 classes/sat)",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()

            curve_path = os.path.join(output_dir, "accuracy_curve.png")
            fig.savefig(curve_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            if not args.quiet:
                print(f"[保存] 准确率曲线 → {curve_path}")
        except Exception as e:
            if not args.quiet:
                print(f"[警告] 曲线图生成失败: {e}")

        if not args.quiet:
            print(f"\n{'='*50}")
            print(f"输出目录: {os.path.abspath(output_dir)}/")
            print("  experiment_config.json   — 实验配置")
            print("  experiment_results.json  — 完整训练历史")
            print("  accuracy_curve.png       — 准确率曲线")
            print("  contact_heatmap.png      — 接触热力图")
            print("  gs_map.png               — 地面站地图")
            print(f"{'='*50}")
        return 0
    else:
        if not args.quiet:
            print("\n[错误] 无训练结果")
        return 1


def main(argv: list[str] | None = None) -> int:
    """CLI 入口（通过 fls quick-test 或直接 python quick_test.py 调用）。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run_quick_test(args)


if __name__ == "__main__":
    sys.exit(main())
