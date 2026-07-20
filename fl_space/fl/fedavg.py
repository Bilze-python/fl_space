"""
FedAvg 算法 — 联邦平均 (Federated Averaging)

论文: "Communication-Efficient Learning of Deep Networks from Decentralized Data"
      (McMahan et al., AISTATS 2017)

算法核心：
    1. 客户端选择：随机采样 C 比例的客户端
    2. 本地训练：每个客户端执行 E 个 epoch 的 SGD
    3. 聚合：同步加权平均（权重 = 客户端数据量占比）
    4. 评估：标准准确率/损失计算

组件可替换性：
    每个组件均可独立替换。例如：
    - 将 RandomSelector 替换为基于连接质量的 selector
    - 将 FixedEpochTrainer 替换为自适应 epoch 的 trainer
    - 将 SyncWeightedAggregator 替换为中位数聚合
"""

from __future__ import annotations

import copy
import random
from typing import Any

from fl_space.fl.core import (
    Aggregator,
    ClientSelector,
    ClientState,
    ClientUpdate,
    Evaluator,
    LocalTrainer,
)

# PyTorch 可选依赖
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── 1. 客户端选择器 ───────────────────────────────────────────


class RandomSelector(ClientSelector):
    """
    随机客户端选择。

    每轮从可连接的客户端中随机采样 C 比例参与训练。

    Parameters
    ----------
    fraction : float
        每轮参与训练的客户端比例，范围 (0, 1]。
    min_clients : int
        最少参与客户端数，防止小数比例导致 0 客户端。
    seed : int | None
        随机种子，用于可复现实验。
    """

    def __init__(
        self,
        fraction: float = 0.5,
        min_clients: int = 2,
        seed: int | None = None,
        connected_only: bool = False,
    ):
        self.fraction = fraction
        self.min_clients = min_clients
        self.connected_only = connected_only
        self._rng = random.Random(seed)

    def select(
        self,
        clients: list[ClientState],
        round_num: int,
        **kwargs: Any,
    ) -> list[int]:
        """随机选择客户端。"""
        eligible = (
            [c for c in clients if c.is_connected]
            if self.connected_only
            else list(clients)
        )
        if not eligible:
            return []

        n_select = max(self.min_clients, int(len(eligible) * self.fraction))
        n_select = min(n_select, len(eligible))

        selected = self._rng.sample(eligible, n_select)
        return [c.client_id for c in selected]


class CappedSelector(ClientSelector):
    """
    带数量上限的随机客户端选择器。

    从可连接客户端中随机选取，但不超过 max_count 个。
    适用于 SpaceFL：max_count = min(GS数, 在线卫星数)。

    Parameters
    ----------
    max_count : int
        最多选择的客户端数。
    min_clients : int
        最少参与客户端数。
    seed : int | None
        随机种子。
    """

    def __init__(
        self,
        max_count: int = 3,
        min_clients: int = 1,
        seed: int | None = None,
    ):
        self.max_count = max_count
        self.min_clients = min_clients
        self._rng = random.Random(seed)

    def select(
        self,
        clients: list[ClientState],
        round_num: int,
        **kwargs: Any,
    ) -> list[int]:
        connected = [c for c in clients if c.is_connected]
        if not connected:
            return []

        n_select = max(self.min_clients, len(connected))
        n_select = min(n_select, self.max_count)
        n_select = min(n_select, len(connected))

        selected = self._rng.sample(connected, n_select)
        return [c.client_id for c in selected]


# ── 2. 本地训练器 ────────────────────────────────────────────


class FixedEpochTrainer(LocalTrainer):
    """
    FedAvg 标准本地训练器。

    每个客户端用 SGD 在本地数据上训练固定 E 个 epoch。

    Parameters
    ----------
    local_epochs : int
        每轮本地训练的 epoch 数，默认 5。
    batch_size : int
        本地训练的 batch size，默认 32。
    learning_rate : float
        学习率，默认 0.01。
    device : str
        计算设备，默认 "cpu"。可设为 "cuda"。
    """

    def __init__(
        self,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("FedAvg 需要 PyTorch，请运行: pip install torch")

        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
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
        执行 FedAvg 本地训练。

        1. 加载全局模型参数
        2. 在本地数据上训练 E 个 epoch
        3. 返回更新后的权重和训练损失
        """
        # 深拷贝模型用于本地训练
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

        # 提取本地训练后的参数
        local_weights = [
            param.data.clone() for param in local_model.parameters()
        ]

        return ClientUpdate(
            client_id=client_id,
            weights=local_weights,
            data_size=data_size,
            train_loss=avg_loss,
            round_num=round_num,
            base_version=round_num,
        )


# ── 3. 聚合器 ─────────────────────────────────────────────────


class SyncWeightedAggregator(Aggregator):
    """
    同步加权平均聚合器 (FedAvg 标准聚合)。

    当收集到足够数量的客户端更新后触发聚合，
    按客户端数据量加权平均模型参数。

    Parameters
    ----------
    min_updates : int
        最少需要的更新数才触发聚合。设为 1 表示有一个就聚合。
        实际使用中通常等于目标选中的客户端数。
    """

    def __init__(self, min_updates: int = 1):
        self.min_updates = min_updates

    def should_aggregate(
        self,
        collected_updates: list[ClientUpdate],
        round_num: int,
        **kwargs: Any,
    ) -> bool:
        """
        当收集到至少 1 个更新时触发聚合。

        在同步 FL 中，服务器先选定客户端、训练全部完成后收集更新，
        因此收到的 update 数量 = 实际参与的客户端数。
        min_updates 仅作为安全下限（避免空聚合），默认为 1。
        """
        return len(collected_updates) >= max(self.min_updates, 1)

    def aggregate(
        self,
        global_weights: list[torch.Tensor],
        updates: list[ClientUpdate],
        round_num: int,
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """
        加权平均聚合。

        权重 = 客户端数据量 / 总数据量
        W_new = Σ (n_k / N) * W_k

        Parameters
        ----------
        global_weights : list
            当前全局模型参数（用于初始化聚合结果的 shape）。
        updates : list[ClientUpdate]
            本轮收集的客户端更新。
        round_num : int
            当前轮次。

        Returns
        -------
        list
            聚合后的新全局模型参数。
        """
        total_size = sum(u.data_size for u in updates)
        if total_size == 0:
            return global_weights

        # 初始化聚合结果为零
        aggregated = [
            torch.zeros_like(w, dtype=torch.float32)
            for w in global_weights
        ]

        for update in updates:
            weight_ratio = update.data_size / total_size
            for agg_w, client_w in zip(aggregated, update.weights):
                if isinstance(client_w, torch.Tensor):
                    agg_w.add_(client_w.float() * weight_ratio)
                else:
                    agg_w.add_(
                        torch.tensor(client_w, dtype=torch.float32) * weight_ratio
                    )

        return aggregated


# ── 4. 评估器 ─────────────────────────────────────────────────


class StandardEvaluator(Evaluator):
    """
    标准评估器。

    在测试集上计算准确率和损失。

    Parameters
    ----------
    device : str
        计算设备，默认 "cpu"。
    """

    def __init__(self, device: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise ImportError("评估需要 PyTorch，请运行: pip install torch")
        self.device = device

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        round_num: int,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        评估模型性能。

        Returns
        -------
        dict[str, float]
            包含 "accuracy" 和 "loss" 的字典。
        """
        model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        return {
            "accuracy": round(accuracy, 6),
            "loss": round(avg_loss, 6),
        }


# ── 5. 便捷构建函数 ───────────────────────────────────────────


def create_fedavg_components(
    fraction: float = 0.5,
    min_clients: int = 2,
    local_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    device: str = "cpu",
    seed: int | None = None,
    connected_only: bool = False,
) -> tuple[ClientSelector, LocalTrainer, Aggregator, Evaluator]:
    """
    一键创建 FedAvg 的四件套组件。

    Parameters
    ----------
    fraction : float
        每轮客户端参与比例。
    min_clients : int
        最少参与客户端数。
    local_epochs : int
        本地训练 epoch 数。
    batch_size : int
        训练 batch size。
    learning_rate : float
        学习率。
    device : str
        计算设备。
    seed : int | None
        随机种子。

    Returns
    -------
    tuple
        (selector, trainer, aggregator, evaluator)

    使用示例::

        from fl_space.fl.fedavg import create_fedavg_components

        selector, trainer, aggregator, evaluator = create_fedavg_components(
            fraction=0.5,
            local_epochs=5,
            device="cuda",
        )
    """
    selector = RandomSelector(
        fraction=fraction,
        min_clients=min_clients,
        seed=seed,
        connected_only=connected_only,
    )
    trainer = FixedEpochTrainer(
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    # min_updates=1: 同步 FL 中训练完全部选中客户端后才聚合，
    # 收到的 update 数 = 实际参与客户端数，不需要额外门槛
    aggregator = SyncWeightedAggregator(min_updates=1)
    evaluator = StandardEvaluator(device=device)

    return selector, trainer, aggregator, evaluator
