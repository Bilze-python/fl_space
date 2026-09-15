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
    """

    def __init__(
        self,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        mu: float = 0.0,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("FedBuff 需要 PyTorch，请运行: pip install torch")

        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mu = max(0.0, float(mu))
        self.device = device

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

        actual_epochs = max(1, int(kwargs.get("local_epochs_override", self.local_epochs)))
        global_anchors = [weight.detach().clone().to(self.device) for weight in global_weights]
        for _epoch in range(actual_epochs):
            epoch_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                if self.mu > 0:
                    proximal = sum(
                        torch.sum((parameter - anchor) ** 2)
                        for parameter, anchor in zip(local_model.parameters(), global_anchors)
                    )
                    loss = loss + (self.mu / 2.0) * proximal
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss

        avg_loss = total_loss / actual_epochs

        local_weights = [param.data.clone() for param in local_model.parameters()]
        model_delta = [
            global_param.detach().clone().to(local_param.device) - local_param
            for global_param, local_param in zip(global_weights, local_weights)
        ]

        return ClientUpdate(
            client_id=client_id,
            weights=local_weights,
            data_size=data_size,
            train_loss=avg_loss,
            round_num=round_num,
            model_delta=model_delta,
            base_version=round_num,
            started_at=int(kwargs.get("started_at", 0)),
            metadata={"actual_local_epochs": actual_epochs, "proximal_mu": self.mu},
        )


# ── 3. 缓冲区聚合器 ───────────────────────────────────────────


class BufferAggregator(Aggregator):
    """
    异步缓冲区聚合器 (FedBuff 核心)。

    维护一个大小为 K 的 FIFO 缓冲区。
    当缓冲区满时：
        1. 取出最旧的 K 个更新
        2. 加权平均聚合
        3. 更新全局模型

    这种方式不需要同步等待所有客户端，
    天然适应卫星通信不可预测的特点。

    Parameters
    ----------
    buffer_size : int
        缓冲区大小 K。K 越大越接近同步聚合，
        K 越小越异步但 staleness 越大。
    staleness_weight : bool
        是否对陈旧更新降权。True 时较旧的更新权重降低。
    """

    def __init__(
        self,
        buffer_size: int = 5,
        staleness_weight: bool = False,
        server_learning_rate: float = 1.0,
        max_staleness: int | None = None,
    ):
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        if server_learning_rate <= 0:
            raise ValueError("server_learning_rate must be positive")
        self.buffer_size = buffer_size
        self.staleness_weight = staleness_weight
        self.server_learning_rate = server_learning_rate
        self.max_staleness = None if max_staleness is None else max(0, int(max_staleness))

        # Do not cap the deque: arrivals beyond K belong to the next server update.
        self._buffer: deque[ClientUpdate] = deque()

        # 全局轮次计数器（用于 staleness 计算）
        self._global_round: int = 0
        self._last_aggregate_count: int = 0
        self._last_batch: list[ClientUpdate] = []
        self._last_staleness: list[int] = []
        self._dropped_stale: int = 0

        # 线程安全锁
        self._lock = threading.Lock()

    def add_update(self, update: ClientUpdate) -> None:
        """
        向缓冲区添加一个客户端更新（线程安全）。

        在异步场景中，训练任务可能在不同线程中完成，
        此方法确保缓冲区操作的线程安全。

        Parameters
        ----------
        update : ClientUpdate
            客户端训练结果。
        """
        with self._lock:
            self._buffer.append(update)

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
            if self.max_staleness is not None:
                retained: deque[ClientUpdate] = deque()
                for update in self._buffer:
                    if max(0, round_num - update.base_version) <= self.max_staleness:
                        retained.append(update)
                    else:
                        self._dropped_stale += 1
                self._buffer = retained
            return len(self._buffer) >= self.buffer_size

    def aggregate(
        self,
        global_weights: list[torch.Tensor],
        updates: list[ClientUpdate],
        round_num: int,
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Apply the mean buffered client delta, following FedBuff Algorithm 1."""
        with self._lock:
            if len(self._buffer) < self.buffer_size:
                self._last_aggregate_count = 0
                return global_weights

            batch_updates = [
                self._buffer.popleft()
                for _ in range(self.buffer_size)
            ]
            self._last_aggregate_count = len(batch_updates)
            self._last_batch = list(batch_updates)

        effective_weights: list[tuple[ClientUpdate, float]] = []
        total_effective_weight = 0.0
        staleness_values: list[int] = []
        for update in batch_updates:
            staleness = max(0, round_num - update.base_version)
            staleness_values.append(staleness)
            base_weight = 1.0
            if self.staleness_weight and staleness > 0:
                effective_weight = base_weight / (1 + staleness)
            else:
                effective_weight = base_weight
            effective_weights.append((update, effective_weight))
            total_effective_weight += effective_weight

        if total_effective_weight <= 0:
            return global_weights

        mean_delta = [torch.zeros_like(w, dtype=torch.float32) for w in global_weights]

        for update, effective_weight in effective_weights:
            weight_ratio = effective_weight / total_effective_weight
            deltas = update.model_delta
            if deltas is None:
                deltas = [
                    global_weight.detach().clone() - client_weight
                    for global_weight, client_weight in zip(global_weights, update.weights)
                ]
            for aggregate_delta, client_delta in zip(mean_delta, deltas):
                if isinstance(client_delta, torch.Tensor):
                    aggregate_delta.add_(client_delta.float() * weight_ratio)
                else:
                    aggregate_delta.add_(
                        torch.tensor(client_delta, dtype=torch.float32) * weight_ratio
                    )

        self._global_round = round_num + 1
        self._last_staleness = staleness_values
        return [
            global_weight.float() - self.server_learning_rate * delta
            for global_weight, delta in zip(global_weights, mean_delta)
        ]

    def buffer_status(self) -> dict[str, Any]:
        """
        查询缓冲区状态。

        Returns
        -------
        dict
            包含 buffer_size, current_count, global_round 的字典。
        """
        with self._lock:
            return {
                "buffer_size": self.buffer_size,
                "current_count": len(self._buffer),
                "global_round": self._global_round,
                "last_aggregate_count": self._last_aggregate_count,
                "last_client_ids": [u.client_id for u in self._last_batch],
                "last_staleness": list(self._last_staleness),
                "dropped_stale": self._dropped_stale,
            }


# ── 4. 便捷构建函数 ───────────────────────────────────────────


def create_fedbuff_components(
    min_clients: int = 2,
    local_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    buffer_size: int = 5,
    staleness_weight: bool = False,
    server_learning_rate: float = 1.0,
    mu: float = 0.0,
    max_staleness: int | None = None,
    device: str = "cpu",
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

    Returns
    -------
    tuple
        (selector, trainer, aggregator, evaluator)

    使用示例::

        from fl_space.fl.fedbuff import create_fedbuff_components

        selector, trainer, aggregator, evaluator = create_fedbuff_components(
            buffer_size=5,
            staleness_weight=True,
            device="cuda",
        )
    """
    selector = AsyncSelector(min_clients=min_clients)
    trainer = AsyncTrainer(
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mu=mu,
        device=device,
    )
    aggregator = BufferAggregator(
        buffer_size=buffer_size,
        staleness_weight=staleness_weight,
        server_learning_rate=server_learning_rate,
        max_staleness=max_staleness,
    )
    evaluator = StandardEvaluator(device=device)

    return selector, trainer, aggregator, evaluator
