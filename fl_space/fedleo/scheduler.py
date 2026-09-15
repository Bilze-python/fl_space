"""
FedLEOScheduler — 完整 FedLEO 实验调度编排

每个训练轮次 = 4 个阶段:
    Phase 0: 数据卸载 (可选, 在每轮开始前执行)
    Phase 1: 本地 SGD 训练 (所有卫星并行)
    Phase 2: 同轨 Ring-Allreduce 聚合
    Phase 3: 跨轨 Ring-Allreduce 聚合
    Phase 4: 评估

与 FLServer.run_sync() 的关键差异:
    - 无中心服务器: 卫星之间直接通过 ISL 通信
    - 分层聚合: 先轨道面内, 再轨道面间
    - 卸载阶段: 每轮可选执行数据再均衡
    - 不依赖 CommunicationScheduler: 使用静态邻接图
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import time as _time
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from fl_space.fedleo.aggregator import FedLEOAggregator
from fl_space.fedleo.metrics import (
    FedLEOMetrics,
    compute_data_balance_entropy,
    compute_system_delay,
    compute_weight_divergence,
)
from fl_space.fedleo.planner import FedLEOPlanner, OffloadPlan

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════


@dataclass
class FedLEOConfig:
    """FedLEO 实验配置。"""

    num_satellites: int = 15          # 总卫星数
    num_planes: int = 3               # 轨道面数
    sats_per_plane: int = 5           # 每轨道面卫星数
    num_rounds: int = 50              # 训练轮次
    local_epochs: int = 2             # 本地训练 epoch
    batch_size: int = 32              # batch size
    learning_rate: float = 0.01       # 学习率
    device: str = "cpu"               # 计算设备

    # 卸载参数
    enable_offloading: bool = True    # 是否启用数据卸载
    offload_every_n_rounds: int = 5   # 每 N 轮执行一次卸载
    max_offload_iter: int = 5         # 单次卸载最大迭代数
    bandwidth_mbps: float = 10.0      # ISL 带宽
    bytes_per_sample: int = 784       # 每样本字节数，用于卸载通信开销估算
    timeslot_duration_sec: float = 60.0
    discrete_ratios: list[float] | None = None
    delay_weight: float = 1.0
    divergence_weight: float = 0.5
    comm_cost_weight: float = 0.3

    # 聚合参数
    use_weighted_average: bool = True

    # 评估参数
    eval_every_n_rounds: int = 1

    # 其他
    seed: int = 42
    verbose: bool = True


# ═══════════════════════════════════════════════════════════════
# FedLEOScheduler
# ═══════════════════════════════════════════════════════════════


class FedLEOScheduler:
    """
    FedLEO 完整调度器。

    不依赖 FLServer，独立管理训练循环。

    Parameters
    ----------
    config : FedLEOConfig
        实验配置。
    plane_map : dict[int, int]
        卫星 ID → 轨道面 ID。
    """

    def __init__(
        self,
        config: FedLEOConfig,
        plane_map: dict[int, int],
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("FedLEOScheduler 需要 PyTorch")

        self.config = config
        self.plane_map = plane_map
        self._rng = np.random.default_rng(config.seed)

        # 子组件
        self.planner: FedLEOPlanner | None = None
        self.aggregator: FedLEOAggregator = FedLEOAggregator(
            plane_map=plane_map,
            use_weighted_average=config.use_weighted_average,
        )

        # 状态
        self.history: list[FedLEOMetrics] = []
        self._reference_weights: list[Any] | None = None

    # ── 初始化 ──────────────────────────────────────────────

    def init_planner(self, adjacency: dict[int, list[int]]) -> None:
        """初始化卸载规划器（需在创建数据后调用，因为需要邻接图）。"""
        self.planner = FedLEOPlanner(
            num_satellites=self.config.num_satellites,
            num_planes=self.config.num_planes,
            plane_map=self.plane_map,
            adjacency=adjacency,
            bandwidth_mbps=self.config.bandwidth_mbps,
            bytes_per_sample=self.config.bytes_per_sample,
            timeslot_duration_sec=self.config.timeslot_duration_sec,
            max_offload_iter=self.config.max_offload_iter,
            discrete_ratios=self.config.discrete_ratios,
            delay_weight=self.config.delay_weight,
            divergence_weight=self.config.divergence_weight,
            comm_cost_weight=self.config.comm_cost_weight,
            seed=self.config.seed,
        )

    # ── 完整训练循环 ────────────────────────────────────────

    def run(
        self,
        model: nn.Module,
        train_loaders: dict[int, DataLoader],
        test_loader: DataLoader,
        initial_data_sizes: list[int],
        reference_weights: list[Any] | None = None,
    ) -> list[FedLEOMetrics]:
        """
        执行 FedLEO 完整训练循环。

        Parameters
        ----------
        model : nn.Module
            初始全局模型。
        train_loaders : dict[int, DataLoader]
            {sat_id: DataLoader} 各卫星的训练数据。
        test_loader : DataLoader
            测试数据加载器。
        initial_data_sizes : list[int]
            初始各卫星样本数。
        reference_weights : list | None
            集中式训练的参考权重（用于计算散度）。

        Returns
        -------
        list[FedLEOMetrics]
            每轮评估指标。
        """
        self._reference_weights = reference_weights
        cfg = self.config
        global_weights = [
            param.data.clone() for param in model.parameters()
        ]

        # 当前数据分布（卸载会修改）
        current_data_sizes = list(initial_data_sizes)
        current_train_loaders = dict(train_loaders)
        offload_dataset, client_indices = self._extract_offload_memberships(
            current_train_loaders
        )

        # ── 基线评估 ──
        if cfg.verbose:
            print(f"\n{'='*60}")
            print(f"FedLEO 训练开始: {cfg.num_satellites}星 {cfg.num_planes}轨面")
            print(f"轮次: {cfg.num_rounds} | 本地epoch: {cfg.local_epochs}")
            print(f"卸载: {'启用' if cfg.enable_offloading else '禁用'} "
                  f"({'每' + str(cfg.offload_every_n_rounds) + '轮' if cfg.enable_offloading else ''})")
            print(f"{'='*60}\n")

        baseline_metrics = self._evaluate(
            model, global_weights, test_loader, 0,
            current_data_sizes, 0.0,
        )
        if baseline_metrics:
            self.history.append(baseline_metrics)

        # ── 训练循环 ──
        for round_num in range(1, cfg.num_rounds + 1):
            round_start = _time.time()

            # Phase 0: 数据卸载（按周期执行）
            offload_plan: OffloadPlan | None = None
            offload_delay = 0.0
            if (
                cfg.enable_offloading
                and self.planner is not None
                and round_num % cfg.offload_every_n_rounds == 0
            ):
                # 计算当前散度
                current_div = 0.0
                if self._reference_weights:
                    current_div = compute_weight_divergence(
                        global_weights, self._reference_weights
                    )

                offload_plan = self.planner.plan(
                    current_data_sizes=current_data_sizes,
                    current_divergence=current_div,
                    round_num=round_num,
                )

                # 应用卸载到数据分布
                if offload_plan and offload_plan.actions:
                    current_train_loaders = self._apply_offload_plan(
                        current_train_loaders=current_train_loaders,
                        shared_dataset=offload_dataset,
                        client_indices=client_indices,
                        plan=offload_plan,
                        round_num=round_num,
                    )
                    current_data_sizes = [
                        len(client_indices.get(sat_id, []))
                        for sat_id in range(cfg.num_satellites)
                    ]
                    offload_delay = sum(
                        a.comm_cost_slots for a in offload_plan.actions
                    )

            # Phase 1: 本地训练（所有卫星并行）
            local_updates: list[tuple[int, list[Any], int]] = []
            train_losses: list[float] = []

            for sat_id in range(cfg.num_satellites):
                loader = current_train_loaders.get(sat_id)
                if loader is None or current_data_sizes[sat_id] <= 0:
                    continue

                update = self._local_train(
                    sat_id=sat_id,
                    model=model,
                    train_loader=loader,
                    global_weights=global_weights,
                    round_num=round_num,
                )
                if update is not None:
                    local_updates.append(
                        (sat_id, update[0], update[1])
                    )
                    train_losses.append(update[2])

            if not local_updates:
                continue

            avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            # Phase 2 & 3: 分层聚合
            global_weights = self.aggregator.aggregate(local_updates)

            # 将聚合结果写回模型
            with torch.no_grad():
                for param, w in zip(model.parameters(), global_weights):
                    param.data.copy_(w)

            # Phase 4: 评估
            round_elapsed = _time.time() - round_start
            train_delay = cfg.local_epochs  # timeslot 估算
            intra_delay = 1  # 同轨聚合 1 slot
            inter_delay = 1  # 跨轨聚合 1 slot
            total_delay = compute_system_delay(
                offload_time=offload_delay,
                train_time=train_delay,
                intra_agg_time=intra_delay,
                inter_agg_time=inter_delay,
            )

            # 散度
            weight_div = 0.0
            if self._reference_weights:
                weight_div = compute_weight_divergence(
                    global_weights, self._reference_weights
                )

            if round_num % cfg.eval_every_n_rounds == 0:
                metrics = self._evaluate(
                    model, global_weights, test_loader, round_num,
                    current_data_sizes, weight_div,
                )
                if metrics:
                    metrics.train_loss = avg_train_loss
                    metrics.offload_delay = offload_delay
                    metrics.train_delay = train_delay
                    metrics.intra_agg_delay = intra_delay
                    metrics.inter_agg_delay = inter_delay
                    metrics.total_delay = total_delay
                    if offload_plan:
                        metrics.total_offloaded_samples = offload_plan.total_offloaded
                        metrics.num_offload_actions = offload_plan.num_actions
                        metrics.extra["offload_actions"] = [
                            {
                                "from_sat": action.from_sat,
                                "to_sat": action.to_sat,
                                "offload_ratio": action.offload_ratio,
                                "offload_samples": action.offload_samples,
                                "comm_cost_slots": action.comm_cost_slots,
                                "score": round(action.score, 6),
                            }
                            for action in offload_plan.actions
                        ]
                    self.history.append(metrics)

                    if cfg.verbose:
                        offload_str = ""
                        if offload_plan and offload_plan.actions:
                            offload_str = (
                                f" | 卸载:{offload_plan.total_offloaded}样本 "
                                f"({offload_plan.num_actions}次)"
                            )
                        print(
                            f"轮次 {round_num:3d}/{cfg.num_rounds} "
                            f"| 准确率:{metrics.accuracy:.4f} "
                            f"| 损失:{metrics.train_loss:.4f} "
                            f"| 散度:{weight_div:.4f} "
                            f"| 时延:{total_delay:.0f}slots"
                            f"{offload_str}"
                            f" | {round_elapsed:.1f}s"
                        )

        return self.history

    def _extract_offload_memberships(
        self,
        train_loaders: dict[int, DataLoader],
    ) -> tuple[Any | None, dict[int, list[int]]]:
        """Extract mutable sample memberships from loaders backed by one dataset."""
        if not self.config.enable_offloading:
            return None, {}

        shared_dataset: Any | None = None
        memberships: dict[int, list[int]] = {
            sat_id: [] for sat_id in range(self.config.num_satellites)
        }
        for sat_id, loader in train_loaders.items():
            dataset = loader.dataset
            if not isinstance(dataset, Subset):
                raise ValueError(
                    "FedLEO offloading requires DataLoaders backed by torch Subset objects"
                )
            if shared_dataset is None:
                shared_dataset = dataset.dataset
            elif dataset.dataset is not shared_dataset:
                raise ValueError(
                    "FedLEO offloading requires every satellite Subset to share one dataset"
                )
            memberships[sat_id] = [int(index) for index in dataset.indices]

        return shared_dataset, memberships

    def _apply_offload_plan(
        self,
        current_train_loaders: dict[int, DataLoader],
        shared_dataset: Any | None,
        client_indices: dict[int, list[int]],
        plan: OffloadPlan,
        round_num: int,
    ) -> dict[int, DataLoader]:
        """Move the planned samples and rebuild affected satellite loaders."""
        if shared_dataset is None:
            raise ValueError("FedLEO offloading cannot move samples without a shared dataset")

        affected: set[int] = set()
        for action in plan.actions:
            source = client_indices[action.from_sat]
            if action.offload_samples > len(source):
                raise ValueError(
                    f"Offload plan requests {action.offload_samples} samples from satellite "
                    f"{action.from_sat}, which only has {len(source)}"
                )
            permutation = self._rng.permutation(len(source))
            selected_positions = {
                int(position) for position in permutation[: action.offload_samples]
            }
            moved = [
                sample_index
                for position, sample_index in enumerate(source)
                if position in selected_positions
            ]
            client_indices[action.from_sat] = [
                sample_index
                for position, sample_index in enumerate(source)
                if position not in selected_positions
            ]
            client_indices[action.to_sat].extend(moved)
            affected.update((action.from_sat, action.to_sat))

        rebuilt = dict(current_train_loaders)
        for sat_id in affected:
            indices = client_indices[sat_id]
            if not indices:
                rebuilt.pop(sat_id, None)
                continue
            generator = torch.Generator()
            generator.manual_seed(
                self.config.seed + round_num * self.config.num_satellites + sat_id
            )
            rebuilt[sat_id] = DataLoader(
                Subset(shared_dataset, indices),
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=False,
                generator=generator,
            )

        return rebuilt

    # ── 内部方法 ────────────────────────────────────────────

    def _local_train(
        self,
        sat_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        global_weights: list[Any],
        round_num: int,
    ) -> tuple[list[Any], int, float] | None:
        """卫星本地 SGD 训练。"""
        cfg = self.config

        local_model = copy.deepcopy(model)
        local_model.to(cfg.device)
        local_model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            local_model.parameters(), lr=cfg.learning_rate,
        )

        data_size = len(train_loader.dataset)  # type: ignore
        total_loss = 0.0

        for _epoch in range(cfg.local_epochs):
            epoch_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(cfg.device), target.to(cfg.device)
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss

        avg_loss = total_loss / max(cfg.local_epochs, 1)

        local_weights = [
            param.data.clone() for param in local_model.parameters()
        ]
        return local_weights, data_size, avg_loss

    def _evaluate(
        self,
        model: nn.Module,
        weights: list[Any],
        test_loader: DataLoader,
        round_num: int,
        data_sizes: list[int],
        weight_div: float,
    ) -> FedLEOMetrics | None:
        """评估模型并返回指标。"""
        # 写回权重
        with torch.no_grad():
            for param, w in zip(model.parameters(), weights):
                param.data.copy_(w)

        model.to(self.config.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)
        balance = compute_data_balance_entropy(data_sizes)

        return FedLEOMetrics(
            round_num=round_num,
            accuracy=round(accuracy, 6),
            train_loss=round(avg_loss, 6),
            weight_divergence=round(weight_div, 6),
            data_balance_entropy=round(balance, 6),
            extra={"data_sizes": list(data_sizes)},
        )
