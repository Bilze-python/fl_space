"""
FedLEO 实验运行器 — FedLEO vs 基线 对比实验

运行流程:
    1. 创建 OrbitSimulator（星座拓扑可视化）
    2. 加载 MNIST 数据集并分配（non-IID, 固定类别数）
    3. 创建 MLP 模型
    4. 运行 FedLEO（卸载 + 分层聚合）
    5. 运行基线 FedAvg（中心化聚合 + 通信调度）
    6. 输出对比结果

使用示例::

    from fl_space.fedleo.experiment import run_fedleo_vs_baseline

    results = run_fedleo_vs_baseline(
        num_planes=3, sats_per_plane=5, num_ground_stations=5,
        num_rounds=30, verbose=True,
    )
"""

import csv
from dataclasses import dataclass, field
import json
import os
import time as _time
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from fl_space.fedleo.metrics import compute_data_balance_entropy
from fl_space.fedleo.scheduler import FedLEOConfig, FedLEOScheduler

# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════


@dataclass
class ExperimentResult:
    """单个实验的完整结果。"""

    name: str
    config: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    final_accuracy: float = 0.0
    peak_accuracy: float = 0.0
    total_delay_slots: float = 0.0
    total_offloaded: int = 0
    elapsed_sec: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """对比实验结果。"""

    fedleo: ExperimentResult = field(default_factory=lambda: ExperimentResult(name="FedLEO"))
    baseline: ExperimentResult = field(default_factory=lambda: ExperimentResult(name="FedAvg-Baseline"))
    config: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# 数据准备
# ═══════════════════════════════════════════════════════════════


def _prepare_data(
    num_clients: int,
    dataset_name: str = "mnist",
    data_dir: str = "./data",
    classes_per_client: int = 2,
    max_samples_per_client: int = 1000,
    sample_imbalance: float = 0.0,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[dict[int, DataLoader], DataLoader, list[int]]:
    """
    准备 MNIST/CIFAR-10 数据并按 non-IID 分配到卫星。

    Returns
    -------
    train_loaders : dict[int, DataLoader]
        {sat_id: DataLoader}
    test_loader : DataLoader
        测试集。
    data_sizes : list[int]
        各卫星初始样本数。
    """
    rng = np.random.default_rng(seed)

    # 加载数据集
    if dataset_name in ("mnist", "fashion_mnist"):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        ds_cls = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
        train_ds = ds_cls(data_dir, train=True, download=True, transform=transform)
        test_ds = ds_cls(data_dir, train=False, download=True, transform=transform)
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    n_samples = len(train_ds)
    targets = np.array([train_ds[i][1] for i in range(n_samples)])
    classes = np.unique(targets)

    # ── non-IID 分配: 固定类别数 + 20% 随机溢散 ──
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    p_preferred = 0.8  # 80% 概率分给偏好客户端

    for class_pos, c in enumerate(classes):
        # 每个类别指定 classes_per_client 个偏好客户端（循环分配）
        preferred_count = max(1, min(classes_per_client, num_clients))
        preferred = [
            (class_pos + offset) % num_clients
            for offset in range(preferred_count)
        ]

        class_indices = np.where(targets == c)[0]
        rng.shuffle(class_indices)

        for sample_idx in class_indices:
            if rng.random() < p_preferred:
                cid = int(rng.choice(preferred))
            else:
                cid = int(rng.integers(0, num_clients))
            client_indices[cid].append(int(sample_idx))

    # ── 样本数截断 ──
    if max_samples_per_client > 0:
        for cid in range(num_clients):
            indices = client_indices[cid]
            cap = max_samples_per_client
            if sample_imbalance > 0:
                lower = max(0.05, 1.0 - min(sample_imbalance, 0.95))
                cap = max(1, int(max_samples_per_client * rng.uniform(lower, 1.0)))
            if len(indices) > cap:
                rng.shuffle(indices)
                client_indices[cid] = indices[:cap]

    # ── 创建 DataLoader ──
    train_loaders: dict[int, DataLoader] = {}
    data_sizes: list[int] = []
    for cid in range(num_clients):
        indices = client_indices[cid]
        data_sizes.append(len(indices))
        if len(indices) == 0:
            continue
        subset = Subset(train_ds, indices)
        generator = torch.Generator()
        generator.manual_seed(seed + cid)
        train_loaders[cid] = DataLoader(
            subset, batch_size=batch_size, shuffle=True, drop_last=False,
            generator=generator,
        )

    test_loader = DataLoader(test_ds, batch_size=max(batch_size * 2, 1), shuffle=False)

    return train_loaders, test_loader, data_sizes


# ═══════════════════════════════════════════════════════════════
# FedLEO 实验
# ═══════════════════════════════════════════════════════════════


def _write_history_csv(path: str, history: list[dict[str, Any]]) -> None:
    """把每轮指标导出为 CSV，嵌套字段以 JSON 字符串保留。"""
    if not history:
        return
    fieldnames = sorted({key for row in history for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            normalized = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    normalized[key] = json.dumps(value, ensure_ascii=False)
                else:
                    normalized[key] = value
            writer.writerow(normalized)


def _create_model(dataset_name: str) -> torch.nn.Module:
    """创建 MLP 模型（784→128→64→10）。"""
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.net(x)

    return MLP()


def run_fedleo_experiment(
    num_planes: int = 3,
    sats_per_plane: int = 5,
    num_rounds: int = 50,
    local_epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    dataset: str = "mnist",
    data_dir: str = "./data",
    device: str = "cpu",
    enable_offloading: bool = True,
    offload_every_n_rounds: int = 5,
    max_offload_iter: int = 3,
    bandwidth_mbps: float = 10.0,
    bytes_per_sample: int | None = None,
    timeslot_duration_sec: float = 60.0,
    discrete_ratios: list[float] | None = None,
    delay_weight: float = 1.0,
    divergence_weight: float = 0.5,
    comm_cost_weight: float = 0.3,
    eval_every_n_rounds: int = 1,
    classes_per_client: int = 2,
    max_samples_per_client: int = 1000,
    sample_imbalance: float = 0.0,
    seed: int = 42,
    verbose: bool = True,
    output_dir: str | None = None,
) -> ExperimentResult:
    """
    运行单个 FedLEO 实验。

    Parameters
    ----------
    num_planes : int
        轨道面数。
    sats_per_plane : int
        每轨道面卫星数。
    num_rounds : int
        训练轮次。
    local_epochs : int
        本地训练 epoch。
    dataset : str
        数据集名称。
    data_dir : str
        数据目录。
    device : str
        计算设备。
    enable_offloading : bool
        是否启用数据卸载。
    seed : int
        随机种子。
    verbose : bool
        是否打印进度。
    output_dir : str | None
        输出目录。

    Returns
    -------
    ExperimentResult
    """
    if not TORCH_AVAILABLE:
        raise ImportError("FedLEO 实验需要 PyTorch")
    if verbose:
        print(f"\n{'='*60}")
        print(f"FedLEO 实验: {num_planes}轨面 × {sats_per_plane}星/面")
        print(f"{'='*60}")
    total_sats = num_planes * sats_per_plane
    t_start = _time.time()

    # 1. 准备数据
    if verbose:
        print("[1/4] 加载数据...")
    train_loaders, test_loader, data_sizes = _prepare_data(
        num_clients=total_sats,
        dataset_name=dataset,
        data_dir=data_dir,
        classes_per_client=classes_per_client,
        max_samples_per_client=max_samples_per_client,
        sample_imbalance=sample_imbalance,
        batch_size=batch_size,
        seed=seed,
    )
    if verbose:
        balance = compute_data_balance_entropy(data_sizes)
        print(f"  卫星数: {total_sats} | 数据均衡度: {balance:.4f}")
        print(f"  每星样本: min={min(data_sizes)} max={max(data_sizes)}")

    # 2. 创建模型
    if verbose:
        print("[2/4] 创建模型...")
    model = _create_model(dataset)
    model.to(device)

    # 3. 构建 plane_map 和邻接图
    plane_map: dict[int, int] = {}
    for sat_id in range(total_sats):
        plane_map[sat_id] = sat_id // sats_per_plane

    # 环形邻接图
    adjacency: dict[int, list[int]] = {}
    for sat_id in range(total_sats):
        plane = plane_map[sat_id]
        pos = sat_id % sats_per_plane
        adj = []
        # 同轨相邻
        adj.append(plane * sats_per_plane + (pos - 1) % sats_per_plane)
        adj.append(plane * sats_per_plane + (pos + 1) % sats_per_plane)
        # 跨轨相邻
        if num_planes > 1:
            prev_p = (plane - 1) % num_planes
            next_p = (plane + 1) % num_planes
            adj.append(prev_p * sats_per_plane + pos)
            adj.append(next_p * sats_per_plane + pos)
        adjacency[sat_id] = sorted(set(adj))

    # 4. 配置并运行 FedLEO
    if verbose:
        print("[3/4] 配置 FedLEO...")
    cfg = FedLEOConfig(
        num_satellites=total_sats,
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        enable_offloading=enable_offloading,
        offload_every_n_rounds=offload_every_n_rounds,
        max_offload_iter=max_offload_iter,
        bandwidth_mbps=bandwidth_mbps,
        bytes_per_sample=bytes_per_sample or (3072 if dataset == "cifar10" else 784),
        timeslot_duration_sec=timeslot_duration_sec,
        discrete_ratios=discrete_ratios,
        delay_weight=delay_weight,
        divergence_weight=divergence_weight,
        comm_cost_weight=comm_cost_weight,
        eval_every_n_rounds=eval_every_n_rounds,
        seed=seed,
        verbose=verbose,
    )

    scheduler = FedLEOScheduler(config=cfg, plane_map=plane_map)
    scheduler.init_planner(adjacency)

    if verbose:
        print("[4/4] 运行训练...")
    history = scheduler.run(
        model=model,
        train_loaders=train_loaders,
        test_loader=test_loader,
        initial_data_sizes=data_sizes,
        reference_weights=None,  # 目前不计算散度（太慢）
    )

    elapsed = _time.time() - t_start

    # 汇总结果
    accs = [m.accuracy for m in history if m.accuracy > 0]
    final_acc = accs[-1] if accs else 0.0
    peak_acc = max(accs) if accs else 0.0
    total_delay = sum(m.total_delay for m in history)
    total_offloaded = sum(m.total_offloaded_samples for m in history)

    result = ExperimentResult(
        name="FedLEO" + ("+Offload" if enable_offloading else ""),
        config={
            "num_planes": num_planes,
            "sats_per_plane": sats_per_plane,
            "total_sats": total_sats,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "offloading": enable_offloading,
            "offload_every_n_rounds": offload_every_n_rounds,
            "max_offload_iter": max_offload_iter,
            "bandwidth_mbps": bandwidth_mbps,
            "bytes_per_sample": cfg.bytes_per_sample,
            "timeslot_duration_sec": timeslot_duration_sec,
            "discrete_ratios": discrete_ratios or [0.0, 0.25, 0.5, 0.75, 1.0],
            "delay_weight": delay_weight,
            "divergence_weight": divergence_weight,
            "comm_cost_weight": comm_cost_weight,
            "eval_every_n_rounds": eval_every_n_rounds,
            "classes_per_client": classes_per_client,
            "max_samples_per_client": max_samples_per_client,
            "sample_imbalance": sample_imbalance,
            "dataset": dataset,
            "seed": seed,
        },
        history=[m.to_dict() for m in history],
        final_accuracy=round(final_acc, 6),
        peak_accuracy=round(peak_acc, 6),
        total_delay_slots=total_delay,
        total_offloaded=total_offloaded,
        elapsed_sec=round(elapsed, 1),
        extra={"data_balance": compute_data_balance_entropy(data_sizes)},
    )

    if verbose:
        print(f"\nFedLEO 完成: "
              f"最终准确率={final_acc:.4f} "
              f"最高={peak_acc:.4f} "
              f"总时延={total_delay:.0f}slots "
              f"卸载={total_offloaded}样本 "
              f"耗时={elapsed:.0f}s")

    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"fedleo_{'offload' if enable_offloading else 'no_offload'}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": result.config,
                "history": result.history,
                "final_accuracy": result.final_accuracy,
                "peak_accuracy": result.peak_accuracy,
                "total_delay_slots": result.total_delay_slots,
                "total_offloaded": result.total_offloaded,
                "elapsed_sec": result.elapsed_sec,
                "extra": result.extra,
            }, f, ensure_ascii=False, indent=2)
        csv_path = os.path.join(
            output_dir,
            f"fedleo_{'offload' if enable_offloading else 'no_offload'}_history.csv",
        )
        _write_history_csv(csv_path, result.history)
        if verbose:
            print(f"结果已保存: {output_path}")
            print(f"CSV已保存: {csv_path}")

    return result


# ═══════════════════════════════════════════════════════════════
# FedAvg 基线（通信用 CappedSelector + SyncWeightedAggregator）
# ═══════════════════════════════════════════════════════════════


def _run_baseline_fedavg(
    num_satellites: int,
    num_ground_stations: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    dataset: str,
    data_dir: str,
    device: str,
    classes_per_client: int,
    max_samples_per_client: int,
    sample_imbalance: float,
    seed: int,
    verbose: bool,
) -> ExperimentResult:
    """运行基线 FedAvg（中心化聚合 + 轨道模拟通信约束）。"""
    if verbose:
        print(f"\n  基线 FedAvg: {num_satellites}星 + {num_ground_stations}站")
    t_start = _time.time()

    from fl_space.fl.fedavg import (
        create_fedavg_components,
    )
    from fl_space.fl.runner import FLRunner
    from fl_space.fl.server import FLConfig

    # 创建轨道模拟器
    from fl_space.simulator.orbit_simulator import OrbitSimulator

    sim = OrbitSimulator(
        num_satellites=num_satellites,
        num_ground_stations=num_ground_stations,
        orbit_altitude_km=500.0,
        orbit_inclination_deg=53.0,
        timeslot_duration_min=1.0,
        num_timeslots=num_rounds * 10,
        random_seed=seed,
        verbose=False,
    )

    # 通信调度器
    from fl_space.fl.scheduler import CommunicationScheduler
    comm_scheduler = CommunicationScheduler(sim)

    # FedAvg 组件
    selector, trainer, aggregator, evaluator = create_fedavg_components(
        fraction=0.5, min_clients=1, local_epochs=local_epochs,
        batch_size=batch_size, learning_rate=learning_rate, device=device, seed=seed,
    )

    # 配置
    config = FLConfig(
        algorithm="fedavg",
        num_rounds=num_rounds,
        num_clients=num_satellites,
        timeslots_per_round=10,
        fraction=0.5,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        classes_per_client=classes_per_client,
        max_samples_per_client=max_samples_per_client,
        device=device,
        seed=seed,
    )

    # Runner
    runner = FLRunner(
        config=config,
        selector=selector,
        trainer=trainer,
        aggregator=aggregator,
        evaluator=evaluator,
        scheduler=comm_scheduler,
    )

    history = runner.run(
        dataset_name=dataset,
        iid=False,
        classes_per_client=classes_per_client,
        max_samples_per_client=max_samples_per_client,
        data_dir=data_dir,
        verbose=verbose,
    )

    elapsed = _time.time() - t_start
    accs = [r.eval_metrics.get("accuracy", 0.0) for r in history if r.eval_metrics]
    final_acc = accs[-1] if accs else 0.0
    peak_acc = max(accs) if accs else 0.0

    return ExperimentResult(
        name="FedAvg-Baseline",
        config={
            "num_satellites": num_satellites,
            "num_ground_stations": num_ground_stations,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "algorithm": "fedavg",
            "dataset": dataset,
            "classes_per_client": classes_per_client,
            "max_samples_per_client": max_samples_per_client,
            "sample_imbalance": sample_imbalance,
        },
        history=[
            {
                "round": r.round_num,
                "accuracy": r.eval_metrics.get("accuracy", 0.0),
                "loss": r.train_loss,
                "timeslot": r.timeslot,
            }
            for r in history
        ],
        final_accuracy=round(final_acc, 6),
        peak_accuracy=round(peak_acc, 6),
        elapsed_sec=round(elapsed, 1),
    )


# ═══════════════════════════════════════════════════════════════
# 对比实验入口
# ═══════════════════════════════════════════════════════════════


def run_fedleo_vs_baseline(
    num_planes: int = 3,
    sats_per_plane: int = 5,
    num_ground_stations: int = 5,
    num_rounds: int = 50,
    local_epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    dataset: str = "mnist",
    data_dir: str = "./data",
    device: str = "cpu",
    offload_every_n_rounds: int = 5,
    max_offload_iter: int = 3,
    bandwidth_mbps: float = 10.0,
    bytes_per_sample: int | None = None,
    timeslot_duration_sec: float = 60.0,
    discrete_ratios: list[float] | None = None,
    delay_weight: float = 1.0,
    divergence_weight: float = 0.5,
    comm_cost_weight: float = 0.3,
    eval_every_n_rounds: int = 1,
    classes_per_client: int = 2,
    max_samples_per_client: int = 1000,
    sample_imbalance: float = 0.0,
    seed: int = 42,
    verbose: bool = True,
    output_dir: str | None = None,
    run_baseline: bool = True,
) -> ComparisonResult:
    """
    运行 FedLEO vs 基线 FedAvg 对比实验。

    Parameters
    ----------
    num_planes : int
        FedLEO 轨道面数。
    sats_per_plane : int
        FedLEO 每轨道面卫星数。
    num_ground_stations : int
        基线实验的地面站数。
    num_rounds : int
        训练轮次。
    local_epochs : int
        本地训练 epoch。
    dataset : str
        数据集。
    data_dir : str
        数据目录。
    device : str
        计算设备。
    seed : int
        随机种子。
    verbose : bool
        是否打印进度。
    output_dir : str | None
        输出目录。
    run_baseline : bool
        是否运行基线对比。

    Returns
    -------
    ComparisonResult
        对比结果。
    """
    total_sats = num_planes * sats_per_plane

    print(f"\n{'#'*60}")
    print("# FedLEO vs 基线 对比实验")
    print(f"# 配置: {num_planes}轨面 × {sats_per_plane}星 = {total_sats}星")
    print(f"# {num_rounds}轮 | {local_epochs} epoch/轮 | {dataset} | {device}")
    print(f"{'#'*60}")

    # ── 1. FedLEO + 卸载 ──
    print(f"\n{'─'*40}")
    print("实验 1/2: FedLEO + 数据卸载")
    print(f"{'─'*40}")
    fedleo_result = run_fedleo_experiment(
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dataset=dataset,
        data_dir=data_dir,
        device=device,
        enable_offloading=True,
        offload_every_n_rounds=offload_every_n_rounds,
        max_offload_iter=max_offload_iter,
        bandwidth_mbps=bandwidth_mbps,
        bytes_per_sample=bytes_per_sample,
        timeslot_duration_sec=timeslot_duration_sec,
        discrete_ratios=discrete_ratios,
        delay_weight=delay_weight,
        divergence_weight=divergence_weight,
        comm_cost_weight=comm_cost_weight,
        eval_every_n_rounds=eval_every_n_rounds,
        classes_per_client=classes_per_client,
        max_samples_per_client=max_samples_per_client,
        sample_imbalance=sample_imbalance,
        seed=seed,
        verbose=verbose,
        output_dir=output_dir,
    )

    # ── 2. 基线 FedAvg ──
    baseline_result: ExperimentResult | None = None
    if run_baseline:
        print(f"\n{'─'*40}")
        print("实验 2/2: 基线 FedAvg")
        print(f"{'─'*40}")
        baseline_result = _run_baseline_fedavg(
            num_satellites=total_sats,
            num_ground_stations=num_ground_stations,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dataset=dataset,
            data_dir=data_dir,
            device=device,
            classes_per_client=classes_per_client,
            max_samples_per_client=max_samples_per_client,
            sample_imbalance=sample_imbalance,
            seed=seed,
            verbose=verbose,
        )

    # ── 对比 ──
    if run_baseline and baseline_result:
        print(f"\n{'='*60}")
        print("对比结果")
        print(f"{'='*60}")
        print(f"{'指标':<25} {'FedLEO':>12} {'基线':>12} {'增益':>12}")
        print(f"{'-'*61}")
        print(f"{'最终准确率':<25} {fedleo_result.final_accuracy:>12.4f} "
              f"{baseline_result.final_accuracy:>12.4f} "
              f"{fedleo_result.final_accuracy - baseline_result.final_accuracy:>+11.4f}")
        print(f"{'最高准确率':<25} {fedleo_result.peak_accuracy:>12.4f} "
              f"{baseline_result.peak_accuracy:>12.4f} "
              f"{fedleo_result.peak_accuracy - baseline_result.peak_accuracy:>+11.4f}")
        print(f"{'总卸载样本':<25} {fedleo_result.total_offloaded:>12d} {'—':>12} {'—':>12}")
        print(f"{'耗时':<25} {fedleo_result.elapsed_sec:>11.1f}s "
              f"{baseline_result.elapsed_sec:>11.1f}s {'—':>12}")

    # 保存对比结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        comp_path = os.path.join(output_dir, "comparison_summary.json")
        comp_data = {
            "config": {
                "num_planes": num_planes,
                "sats_per_plane": sats_per_plane,
                "total_sats": total_sats,
                "num_ground_stations": num_ground_stations,
                "num_rounds": num_rounds,
                "local_epochs": local_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "dataset": dataset,
                "offload_every_n_rounds": offload_every_n_rounds,
                "max_offload_iter": max_offload_iter,
                "bandwidth_mbps": bandwidth_mbps,
                "bytes_per_sample": bytes_per_sample,
                "timeslot_duration_sec": timeslot_duration_sec,
                "discrete_ratios": discrete_ratios or [0.0, 0.25, 0.5, 0.75, 1.0],
                "delay_weight": delay_weight,
                "divergence_weight": divergence_weight,
                "comm_cost_weight": comm_cost_weight,
                "eval_every_n_rounds": eval_every_n_rounds,
                "classes_per_client": classes_per_client,
                "max_samples_per_client": max_samples_per_client,
                "sample_imbalance": sample_imbalance,
            },
            "fedleo": {
                "final_accuracy": fedleo_result.final_accuracy,
                "peak_accuracy": fedleo_result.peak_accuracy,
                "total_offloaded": fedleo_result.total_offloaded,
                "elapsed_sec": fedleo_result.elapsed_sec,
                "history": fedleo_result.history,
            },
        }
        if baseline_result:
            comp_data["baseline"] = {
                "final_accuracy": baseline_result.final_accuracy,
                "peak_accuracy": baseline_result.peak_accuracy,
                "elapsed_sec": baseline_result.elapsed_sec,
                "history": baseline_result.history,
            }
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comp_data, f, ensure_ascii=False, indent=2)
        _write_history_csv(
            os.path.join(output_dir, "comparison_fedleo_history.csv"),
            fedleo_result.history,
        )
        if baseline_result:
            _write_history_csv(
                os.path.join(output_dir, "comparison_baseline_history.csv"),
                baseline_result.history,
            )
        print(f"\n对比结果已保存: {comp_path}")

    return ComparisonResult(
        fedleo=fedleo_result,
        baseline=baseline_result if baseline_result else ExperimentResult(name="FedAvg-Baseline"),
        config={
            "num_planes": num_planes,
            "sats_per_plane": sats_per_plane,
            "total_sats": total_sats,
            "num_ground_stations": num_ground_stations,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dataset": dataset,
            "seed": seed,
            "offload_every_n_rounds": offload_every_n_rounds,
            "max_offload_iter": max_offload_iter,
            "bandwidth_mbps": bandwidth_mbps,
            "discrete_ratios": discrete_ratios or [0.0, 0.25, 0.5, 0.75, 1.0],
            "delay_weight": delay_weight,
            "divergence_weight": divergence_weight,
            "comm_cost_weight": comm_cost_weight,
            "classes_per_client": classes_per_client,
            "max_samples_per_client": max_samples_per_client,
            "sample_imbalance": sample_imbalance,
        },
    )


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════

def parse_discrete_ratios(value: str | None) -> list[float] | None:
    """解析逗号分隔的卸载比例列表。"""
    if not value:
        return None
    ratios = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not ratios:
        return None
    for ratio in ratios:
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("离散卸载比例必须在 [0, 1] 范围内")
    return sorted(set(ratios))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FedLEO 实验运行器")
    parser.add_argument("--planes", type=int, default=3, help="轨道面数")
    parser.add_argument("--sats-per-plane", type=int, default=5, help="每轨道面卫星数")
    parser.add_argument("--gs", type=int, default=5, help="地面站数(基线)")
    parser.add_argument("--rounds", type=int, default=50, help="训练轮次")
    parser.add_argument("--epochs", type=int, default=2, help="本地epoch")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--dataset", default="mnist", help="数据集")
    parser.add_argument("--data-dir", default="./data", help="数据目录")
    parser.add_argument("--device", default="cpu", help="计算设备")
    parser.add_argument("--offload-every", type=int, default=5, help="每N轮执行一次卸载")
    parser.add_argument("--max-offload-iter", type=int, default=3, help="单次卸载最大迭代数")
    parser.add_argument("--bandwidth-mbps", type=float, default=10.0, help="ISL带宽Mbps")
    parser.add_argument("--bytes-per-sample", type=int, default=None, help="每样本字节数")
    parser.add_argument("--timeslot-sec", type=float, default=60.0, help="timeslot秒数")
    parser.add_argument("--ratios", default=None, help="离散卸载比例, 如 0,0.25,0.5,1")
    parser.add_argument("--delay-weight", type=float, default=1.0, help="时延改善权重")
    parser.add_argument("--divergence-weight", type=float, default=0.5, help="数据均衡/散度代理权重")
    parser.add_argument("--comm-cost-weight", type=float, default=0.3, help="通信成本权重")
    parser.add_argument("--eval-every", type=int, default=1, help="每N轮评估一次")
    parser.add_argument("--classes-per-client", type=int, default=2, help="每类偏好客户端数")
    parser.add_argument("--max-samples-per-client", type=int, default=1000, help="每客户端最大样本数")
    parser.add_argument("--sample-imbalance", type=float, default=0.0, help="样本截断异构度[0,0.95]")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", "-o", default="fedleo_output", help="输出目录")
    parser.add_argument("--no-baseline", action="store_true", help="不跑基线对比")
    parser.add_argument("--quiet", "-q", action="store_true", help="安静模式")
    args = parser.parse_args()

    run_fedleo_vs_baseline(
        num_planes=args.planes,
        sats_per_plane=args.sats_per_plane,
        num_ground_stations=args.gs,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dataset=args.dataset,
        data_dir=args.data_dir,
        device=args.device,
        offload_every_n_rounds=args.offload_every,
        max_offload_iter=args.max_offload_iter,
        bandwidth_mbps=args.bandwidth_mbps,
        bytes_per_sample=args.bytes_per_sample,
        timeslot_duration_sec=args.timeslot_sec,
        discrete_ratios=parse_discrete_ratios(args.ratios),
        delay_weight=args.delay_weight,
        divergence_weight=args.divergence_weight,
        comm_cost_weight=args.comm_cost_weight,
        eval_every_n_rounds=args.eval_every,
        classes_per_client=args.classes_per_client,
        max_samples_per_client=args.max_samples_per_client,
        sample_imbalance=args.sample_imbalance,
        seed=args.seed,
        verbose=not args.quiet,
        output_dir=args.output,
        run_baseline=not args.no_baseline,
    )
