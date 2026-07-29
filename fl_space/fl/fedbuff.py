"""
FedBuff 算法 — 异步联邦缓冲聚合 (Federated Buffered)

论文: "FedBuff: Federated Learning with Buffered Asynchronous Aggregation"
      (Nguyen et al., AISTATS 2022)

与 FedAvg 的关键差异：
    - 异步聚合：服务端不需要等待所有选中客户端，
      而是维护一个大小为 K 的缓冲区。
    - 当缓冲区收集满 K 个更新后，立即用这 K 个更新
      执行一次聚合（先进先出 FIFO）。
    - 客户端可以随时提交更新，不受轮次同步约束。
    - 更适合太空场景：卫星通信窗口不可预测，
      无法保证同步等待所有客户端。

组件设计：
    客户端选择器 → AsyncSelector（所有已连接客户端都可参与）
    本地训练器   → AsyncTrainer（客户端异步训练后提交更新）
    聚合器       → BufferAggregator（FIFO 缓冲区聚合）
    评估器       → StandardEvaluator（与 FedAvg 相同）
"""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import threading
from typing import Any

from fl_space.fl.core import (
    Aggregator,
    ClientSelector,
    ClientState,
    ClientUpdate,
    Evaluator,
    LocalTrainer,
)
from fl_space.fl.fedavg import StandardEvaluator

# PyTorch 可选依赖
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── 1. 异步客户端选择器 ──────────────────────────────────────


class AsyncSelector(ClientSelector):
    """
    异步客户端选择器。

    所有已连接且准备好更新的客户端都可以参与，
    不需要等待固定轮次的同步。

    与 FedAvg 的 RandomSelector 不同：
    - RandomSelector 每轮随机选 C 比例
    - AsyncSelector 让所有连接者自由提交

    Parameters
    ----------
    min_clients : int
        最少参与者（用于初始轮）。
    """

    def __init__(self, min_clients: int = 2):
        self.min_clients = min_clients

    def select(
        self,
        clients: list[ClientState],
        round_num: int,
        **kwargs: Any,
    ) -> list[int]:
        """
        选择所有可连接的客户端。

        在异步模式下，所有连接的客户端都可以参与训练，
        无需同步等待。

        Parameters
        ----------
        clients : list[ClientState]
            所有客户端状态。
        round_num : int
            当前轮次（异步模式下为参考值）。
        **kwargs
            可包含 already_training: set[int]，排除已在训练中的客户端。

        Returns
        -------
        list[int]
            可参与训练的客户端 ID 列表。
        """
        already_training: set[int] = kwargs.get("already_training", set())
        connected = [
            c for c in clients
            if c.is_connected and c.client_id not in already_training
        ]
        return [c.client_id for c in connected]


# ── 2. 异步本地训练器 ────────────────────────────────────────


class AsyncTrainer(LocalTrainer):
    """
    异步本地训练器。

    与 FedAvg 的 FixedEpochTrainer 类似（固定 E 个 epoch 的 SGD），
    但设计用于异步场景：客户端独立训练并随时提交更新。

    支持动态 Epoch：通过 timing_cache 按轨道间隔自适应调整本地
    训练轮数，避免固定 epoch 浪费短窗口或不足长窗口。

    Parameters
    ----------
    local_epochs : int
        本地训练 epoch 数，默认 5。
    batch_size : int
        训练 batch size，默认 32。
    learning_rate : float
        学习率，默认 0.01。
    device : str
        计算设备。
    timing_cache : TimingCachePool | None
        定时缓存池，用于动态 Epoch 计算。None 时使用固定 epoch。
    max_epochs : int
        动态模式下最大 epoch 上限。
    warning_threshold_min : float
        窗口预警阈值（分钟），距下次接轨<此时强制停止训练。
    """

    def __init__(
        self,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        device: str = "cpu",
        timing_cache: Any | None = None,
        max_epochs: int = 10,
        warning_threshold_min: float = 5.0,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("FedBuff 需要 PyTorch，请运行: pip install torch")

        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self._timing_cache = timing_cache
        self.max_epochs = max_epochs
        self.warning_threshold_min = warning_threshold_min
        self._effective_epochs: int = local_epochs

    def set_client_context(self, client_id: int, timeslot: int) -> int:
        """
        设置当前客户端上下文，计算动态 Epoch。

        Parameters
        ----------
        client_id : int
            客户端 ID。
        timeslot : int
            当前虚拟时间。

        Returns
        -------
        int
            生效的 epoch 数。
        """
        if self._timing_cache is not None:
            dyn_epochs = self._timing_cache.get_dynamic_epochs(
                client_id, timeslot,
                max_epochs=self.max_epochs,
                warning_threshold_min=self.warning_threshold_min,
            )
            self._effective_epochs = dyn_epochs
            self.local_epochs = dyn_epochs
            return dyn_epochs
        self._effective_epochs = self.local_epochs
        return self.local_epochs

    def train(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        global_weights: list[Any],
        round_num: int,
        **kwargs: Any,
    ) -> ClientUpdate:
        """
        异步本地训练。

        客户端下载当前全局模型（可能有一定 staleness），
        在本地训练后提交更新。
        """
        local_model = copy.deepcopy(model)
        local_model.to(self.device)
        local_model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.learning_rate,
        )

        data_size = len(train_loader.dataset)  # type: ignore
        total_loss = 0.0

        for _epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss

        avg_loss = total_loss / max(self.local_epochs, 1)

        local_weights = [
            param.data.clone() for param in local_model.parameters()
        ]

        return ClientUpdate(
            client_id=client_id,
            weights=local_weights,
            data_size=data_size,
            train_loss=avg_loss,
            round_num=round_num,
        )


# ── 3. 缓冲区聚合器 ───────────────────────────────────────────


@dataclass
class StalenessConfig:
    """陈旧度配置 — 动态阈值与分层丢弃。

    Attributes
    ----------
    base_tau : int
        基础陈旧阈值 τ。
    dynamic_tau : bool
        是否启用基于轨道间隔的动态 τ。
    tiered_filter : bool
        是否启用分层陈旧丢弃（轻度/中度/重度）。
    mild_weight_decay : float
        轻度陈旧降权因子（0~1），默认 0.9。
    moderate_weight_decay : float
        中度陈旧降权因子（0~1），默认 0.5。
    adaptive_buffer : bool
        是否启用基于在线卫星数的动态缓冲区大小。
    timing_cache : Any | None
        定时缓存池引用。
    """

    base_tau: int = 5
    dynamic_tau: bool = False
    tiered_filter: bool = False
    mild_weight_decay: float = 0.9
    moderate_weight_decay: float = 0.5
    adaptive_buffer: bool = False
    timing_cache: Any | None = None


class BufferAggregator(Aggregator):
    """
    异步缓冲区聚合器 (FedBuff 核心) — 含动态陈旧度优化。

    维护一个大小为 K 的 FIFO 缓冲区。
    当缓冲区满时：
        1. 取出最旧的 K 个更新
        2. 分层陈旧度过滤（轻度/中度/重度）
        3. 加权平均聚合
        4. 更新全局模型

    这种方式不需要同步等待所有客户端，
    天然适应卫星通信不可预测的特点。

    新增（论文第四方向 FedBuff Async Staleness Timing）：
        - 动态陈旧阈值 τ：长间隔卫星放宽，短间隔收紧
        - 自适应缓冲区 D：在线卫星多增大，少缩小
        - 分层陈旧丢弃：轻度保留 → 中度降权 → 重度丢弃

    Parameters
    ----------
    buffer_size : int
        缓冲区大小 K。K 越大越接近同步聚合，
        K 越小越异步但 staleness 越大。
    staleness_weight : bool
        是否对陈旧更新降权。True 时较旧的更新权重降低。
    staleness_config : StalenessConfig | None
        陈旧度动态配置。None 时使用传统固定策略。
    """

    def __init__(
        self,
        buffer_size: int = 5,
        staleness_weight: bool = False,
        staleness_config: StalenessConfig | None = None,
    ):
        self.buffer_size = buffer_size
        self.staleness_weight = staleness_weight
        self._staleness_cfg = staleness_config

        # 动态缓冲区范围
        self._base_buffer_size = buffer_size
        self._min_buffer_size = max(2, buffer_size // 2)
        self._max_buffer_size = buffer_size * 2

        # FIFO 缓冲区
        self._buffer: deque[ClientUpdate] = deque(maxlen=buffer_size)

        # 全局轮次计数器（用于 staleness 计算）
        self._global_round: int = 0
        self._last_aggregate_count: int = 0

        # 统计
        self._drop_count: int = 0
        self._total_updates_received: int = 0

        # 线程安全锁
        self._lock = threading.Lock()

    def update_buffer_size(self, online_sat_count: int) -> int:
        """
        基于在线卫星数动态调整缓冲区大小。

        在线卫星多 → 增大 D，批量聚合提升效率
        在线卫星少（1-2颗） → 缩小 D，减少轮次等待

        Parameters
        ----------
        online_sat_count : int
            当前在线卫星数。

        Returns
        -------
        int
            调整后的缓冲区大小。
        """
        if not (self._staleness_cfg and self._staleness_cfg.adaptive_buffer):
            return self.buffer_size

        if online_sat_count <= 2:
            new_size = max(self._min_buffer_size, online_sat_count)
        elif online_sat_count <= 5:
            new_size = online_sat_count
        else:
            new_size = min(self._max_buffer_size, online_sat_count * 2 // 3)

        with self._lock:
            self.buffer_size = new_size
            # 用新的 maxlen 重新创建 deque
            old_buf = list(self._buffer)
            self._buffer = deque(old_buf, maxlen=new_size)

        return new_size

    def _get_dynamic_tau(self, sat_id: int) -> int:
        """获取某卫星的动态陈旧阈值。

        Parameters
        ----------
        sat_id : int
            卫星 ID（-1 时返回基础值）。

        Returns
        -------
        int
            动态 τ 值。
        """
        cfg = self._staleness_cfg
        if cfg is None or not cfg.dynamic_tau or cfg.timing_cache is None:
            return cfg.base_tau if cfg else 5

        # 使用 timing_cache 获取动态 τ
        if sat_id >= 0:
            return cfg.timing_cache.get_staleness_threshold(sat_id, 0, cfg.base_tau)
        return cfg.base_tau

    def _classify_staleness(self, staleness: int, sat_id: int) -> str:
        """分层陈旧度分类。

        Parameters
        ----------
        staleness : int
            陈旧轮次数。
        sat_id : int
            卫星 ID。

        Returns
        -------
        str
            "mild" | "moderate" | "severe" | "normal"
        """
        cfg = self._staleness_cfg
        if cfg is None or not cfg.tiered_filter:
            return "normal"

        tau = self._get_dynamic_tau(sat_id)

        if staleness <= tau:
            return "mild"
        elif staleness <= 2 * tau:
            return "moderate"
        else:
            return "severe"

    def add_update(self, update: ClientUpdate) -> bool:
        """
        向缓冲区添加一个客户端更新（线程安全 + 分层陈旧过滤）。

        分层陈旧丢弃机制：
            1. 轻度陈旧（τ内）：正常入缓冲区（可能适度降权）
            2. 中度陈旧（τ~2τ）：降权入缓冲区，不直接丢弃
            3. 重度陈旧（>2τ）：定时判定失效，拒绝入队

        Parameters
        ----------
        update : ClientUpdate
            客户端训练结果。

        Returns
        -------
        bool
            True 表示成功入队，False 表示因陈旧度被丢弃。
        """
        self._total_updates_received += 1

        # 分层陈旧度检查
        cfg = self._staleness_cfg
        if cfg is not None and cfg.tiered_filter:
            staleness = max(0, self._global_round - update.round_num)
            classification = self._classify_staleness(staleness, update.client_id)

            if classification == "severe":
                # 重度陈旧：丢弃
                with self._lock:
                    self._drop_count += 1
                return False

        with self._lock:
            self._buffer.append(update)
        return True

    def should_aggregate(
        self,
        collected_updates: list[ClientUpdate],
        round_num: int,
        **kwargs: Any,
    ) -> bool:
        """
        当缓冲区达到 buffer_size 时触发聚合。

        注意：异步模式下 collected_updates 参数通常为空或忽略，
        实际判断基于内部缓冲区大小。
        """
        with self._lock:
            return len(self._buffer) >= self.buffer_size

    def aggregate(
        self,
        global_weights: list[torch.Tensor],
        updates: list[ClientUpdate],
        round_num: int,
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Aggregate buffered updates with tiered staleness weighting."""
        with self._lock:
            self._global_round += 1

            if len(self._buffer) < self.buffer_size:
                self._last_aggregate_count = 0
                return global_weights

            batch_updates = [
                self._buffer.popleft()
                for _ in range(self.buffer_size)
            ]
            self._last_aggregate_count = len(batch_updates)

        total_size = sum(u.data_size for u in batch_updates)
        if total_size == 0:
            return global_weights

        cfg = self._staleness_cfg
        effective_weights: list[tuple[ClientUpdate, float]] = []
        total_effective_weight = 0.0
        for update in batch_updates:
            staleness = max(0, self._global_round - update.round_num)
            base_weight = update.data_size / total_size

            # 分层陈旧降权
            if cfg is not None and cfg.tiered_filter:
                classification = self._classify_staleness(staleness, update.client_id)
                if classification == "moderate":
                    base_weight *= cfg.moderate_weight_decay
                elif classification == "mild" and staleness > 0:
                    base_weight *= cfg.mild_weight_decay
            elif self.staleness_weight and staleness > 0:
                base_weight = base_weight / (1 + staleness)

            effective_weights.append((update, base_weight))
            total_effective_weight += base_weight

        if total_effective_weight <= 0:
            return global_weights

        aggregated = [
            torch.zeros_like(w, dtype=torch.float32)
            for w in global_weights
        ]

        for update, effective_weight in effective_weights:
            weight_ratio = effective_weight / total_effective_weight
            for agg_w, client_w in zip(aggregated, update.weights):
                if isinstance(client_w, torch.Tensor):
                    agg_w.add_(client_w.float() * weight_ratio)
                else:
                    agg_w.add_(
                        torch.tensor(client_w, dtype=torch.float32) * weight_ratio
                    )

        return aggregated

    def buffer_status(self) -> dict[str, Any]:
        """
        查询缓冲区状态。

        Returns
        -------
        dict
            包含 buffer_size, current_count, global_round 等信息的字典。
        """
        with self._lock:
            return {
                "buffer_size": self.buffer_size,
                "current_count": len(self._buffer),
                "global_round": self._global_round,
                "last_aggregate_count": self._last_aggregate_count,
                "total_updates_received": self._total_updates_received,
                "drop_count": self._drop_count,
                "drop_rate": (
                    self._drop_count / max(self._total_updates_received, 1)
                ),
            }


# ── 4. 便捷构建函数 ───────────────────────────────────────────


def create_fedbuff_components(
    min_clients: int = 2,
    local_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    buffer_size: int = 5,
    staleness_weight: bool = False,
    device: str = "cpu",
    staleness_config: StalenessConfig | None = None,
    timing_cache: Any | None = None,
    max_epochs: int = 10,
    warning_threshold_min: float = 5.0,
) -> tuple[ClientSelector, LocalTrainer, Aggregator, Evaluator]:
    """
    一键创建 FedBuff 的四件套组件。

    Parameters
    ----------
    min_clients : int
        最少参与者。
    local_epochs : int
        本地训练 epoch 数。
    batch_size : int
        训练 batch size。
    learning_rate : float
        学习率。
    buffer_size : int
        聚合缓冲区大小 K。
    staleness_weight : bool
        是否对陈旧更新降权。
    device : str
        计算设备。
    staleness_config : StalenessConfig | None
        陈旧度动态配置（含动态τ、分层丢弃、自适应D）。
    timing_cache : TimingCachePool | None
        定时缓存池（用于动态 Epoch）。
    max_epochs : int
        动态模式下最大 epoch 数。
    warning_threshold_min : float
        窗口预警阈值（分钟）。

    Returns
    -------
    tuple
        (selector, trainer, aggregator, evaluator)

    使用示例::

        from fl_space.fl.fedbuff import create_fedbuff_components, StalenessConfig

        sc = StalenessConfig(base_tau=5, dynamic_tau=True, tiered_filter=True)
        selector, trainer, aggregator, evaluator = create_fedbuff_components(
            buffer_size=5,
            staleness_weight=True,
            staleness_config=sc,
            device="cuda",
        )
    """
    selector = AsyncSelector(min_clients=min_clients)
    trainer = AsyncTrainer(
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        timing_cache=timing_cache,
        max_epochs=max_epochs,
        warning_threshold_min=warning_threshold_min,
    )
    aggregator = BufferAggregator(
        buffer_size=buffer_size,
        staleness_weight=staleness_weight,
        staleness_config=staleness_config,
    )
    evaluator = StandardEvaluator(device=device)

    return selector, trainer, aggregator, evaluator
