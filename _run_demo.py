"""
SpaceFL 快速演示 — 轨道模拟 + FL训练一步跑通

用法:
    python _run_demo.py            # 英文版
    python _run_demo.py --lang zh  # 中文版
"""
import argparse
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fl_space.environment import CelestialBody, create_default_network
from fl_space.orbit import create_circular_orbit, KeplerOrbit
from fl_space.simulator import OrbitSimulator
from fl_space.viz.i18n import setup_cjk_font, t as i18n_t

# 解析 --lang 参数
_parser = argparse.ArgumentParser()
_parser.add_argument("--lang", choices=["en", "zh"], default="en")
_args, _ = _parser.parse_known_args()
LANG = _args.lang
if LANG == "zh":
    setup_cjk_font()

# ═══════════════════════════════════════════════════════
#  Part 1: 轨道模拟
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("  Part 1: 轨道环境模拟")
print("=" * 60)

earth = CelestialBody.earth()
print(f"\n  天体: {earth.name} (R={earth.radius_km}km, GM={earth.GM:.1f})")

# 3颗卫星 @ 500km, 倾角53°, 均分轨道
orbits = [create_circular_orbit(500, 53, 0, i * 120, earth) for i in range(3)]
for i, orb in enumerate(orbits):
    lat, lon = orb.position_at_time_deg(0)
    print(f"  SAT-{i}: 周期={orb.period_min:.1f}min, 初始位置=({lat:+.1f}°, {lon:+.1f}°)")

# 7个地面站
gss = create_default_network(7)
print(f"\n  地面站网络: {gss.count} 站")

# 运行24小时模拟
sim = OrbitSimulator(
    body=earth, orbits=orbits, ground_station_network=gss,
    num_timeslots=1440, timeslot_duration_min=1.0,  # 24h
    backend="kepler", verbose=False,
)
print(f"\n  {sim.summary()}")

# 接触率统计
print(f"\n  接触率: {sim.stats.get('contact_rate', 0):.1%}")

# 某颗卫星的通信记录
record = sim.get_communication_record(0)
print(f"  SAT-0 24h内通信窗口数: {len(record)}")

# ═══════════════════════════════════════════════════════
#  Part 2: 生成接触热力图
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Part 2: 接触热力图")
print("=" * 60)

from fl_space.utils import plot_contact_heatmap, plot_ground_station_map

out_dir = "demo_output"
os.makedirs(out_dir, exist_ok=True)

plot_contact_heatmap(sim, f"{out_dir}/contact_heatmap.png",
                     title="Contact Heatmap (3 Sats × 7 GS, 24h)", lang=LANG)
print(f"  热力图已保存: {out_dir}/contact_heatmap.png")

plot_ground_station_map(sim, f"{out_dir}/gs_map.png",
                        title="Ground Stations (7 stations)", lang=LANG)
print(f"  地面站地图已保存: {out_dir}/gs_map.png")

# ═══════════════════════════════════════════════════════
#  Part 3: FL 训练 (FedAvg, 20轮快速测试)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Part 3: 联邦学习训练 (FedAvg, 20轮)")
print("=" * 60)

from fl_space.fl.fedavg import (
    CappedSelector, FixedEpochTrainer, StandardEvaluator, SyncWeightedAggregator,
)
from fl_space.fl.runner import FLRunner
from fl_space.fl.scheduler import CommunicationScheduler
from fl_space.fl.server import FLConfig

config = FLConfig(
    algorithm="fedavg", num_rounds=20, num_clients=3,
    fraction=1.0, local_epochs=1, batch_size=32,
    learning_rate=0.01, device="cpu", seed=42,
    time_model="slot",
)

scheduler = CommunicationScheduler(sim)
selector = CappedSelector(max_count=3, min_clients=1, seed=42)
trainer = FixedEpochTrainer(local_epochs=1, batch_size=32, learning_rate=0.01, device="cpu")
aggregator = SyncWeightedAggregator(min_updates=1)
evaluator = StandardEvaluator(device="cpu")

runner = FLRunner(config, selector, trainer, aggregator, evaluator, scheduler=scheduler)

t0 = time.time()
history = runner.run(dataset_name="mnist", iid=True, verbose=True)
elapsed = time.time() - t0

if history:
    accs = [h.eval_metrics.get("accuracy", 0) for h in history]
    print(f"\n  完成! {len(history)} 轮, 耗时 {elapsed:.1f}s")
    print(f"  初始准确率: {accs[0]:.4f}")
    print(f"  最终准确率: {accs[-1]:.4f}")
    print(f"  最高准确率: {max(accs):.4f}")

    # 准确率曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    rounds = list(range(1, len(accs) + 1))
    ax.plot(rounds, accs, "b-o", markersize=4, linewidth=1)
    ax.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5, label=i18n_t("90%", LANG))
    ax.set_xlabel(i18n_t("Round", LANG))
    ax.set_ylabel(i18n_t("Accuracy", LANG))
    ax.set_title(f"FedAvg on MNIST — 3 Sats × 7 GS (20 rounds, {elapsed:.1f}s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(f"{out_dir}/accuracy_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  准确率曲线: {out_dir}/accuracy_curve.png")

print(f"\n{'='*60}")
print(f"  演示完成! 输出目录: {os.path.abspath(out_dir)}/")
print(f"{'='*60}")

input("按回车键退出...")