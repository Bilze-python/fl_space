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
    ):
        self.fraction = fraction
        self.min_clients = min_clients
        self._rng = random.Random(seed)

    def select(
        self,
        clients: list[ClientState],
        round_num: int,
        **kwargs: Any,
    ) -> list[int]:
        """随机选择客户端。"""
        # 仅考虑已连接的客户端
        connected = [c for c in clients if c.is_connected]
        if not connected:
            return []

        n_select = max(self.min_clients, int(len(connected) * self.fraction))
        n_select = min(n_select, len(connected))

        selected = self._rng.sample(connected, n_select)
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


class LoadBalancedCappedSelector(CappedSelector):
    """
    带负载均衡感知的数量上限选择器 (Round 12)。

    在 CappedSelector 基础上增加负载均衡感知：
    - 优先选择当前负载较低的 GS 覆盖的卫星
    - 打散多 GS 争抢同一卫星的扎堆现象
    - 多站少星场景下自动触发休眠调度

    Parameters
    ----------
    max_count : int
        最多选择的客户端数。
    min_clients : int
        最少参与客户端数。
    load_balance_weight : float
        负载均衡权重 (0=纯随机, 1=完全负载优先)。
    scheduler : CommunicationScheduler | None
        通信调度器引用（用于获取 GS 负载信息）。
    seed : int | None
        随机种子。
    """

    def __init__(
        self,
        max_count: int = 3,
        min_clients: int = 1,
        load_balance_weight: float = 0.3,
        scheduler: Any | None = None,
        seed: int | None = None,
    ):
        super().__init__(max_count=max_count, min_clients=min_clients, seed=seed)
        self.load_balance_weight = load_balance_weight
        self._scheduler = scheduler
        self._gs_selection_counts: dict[int, int] = {}

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

        # 无调度器或权重为 0 → 纯随机
        if self._scheduler is None or self.load_balance_weight <= 0 or n_select <= 1:
            selected = self._rng.sample(connected, n_select)
            return [c.client_id for c in selected]

        # 负载均衡感知选择：优先低负载 GS 覆盖的卫星
        scores = []
        for c in connected:
            gs_ids = kwargs.get("sat_to_gs", {}).get(c.client_id, [])
            if gs_ids:
                avg_gs_load = sum(
                    self._gs_selection_counts.get(g, 0) for g in gs_ids
                ) / len(gs_ids)
            else:
                avg_gs_load = 0.0
            # 分数 = 随机因子 + 负载均衡因子（负载越低分数越高）
            random_score = self._rng.random()
            load_score = 1.0 / (1.0 + avg_gs_load)  # 负载低的分数高
            combined = (
                (1 - self.load_balance_weight) * random_score
                + self.load_balance_weight * load_score
            )
            scores.append((c, combined))

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [c for c, _ in scores[:n_select]]

        # 更新 GS 选择计数
        for c in selected:
            for gs_id in kwargs.get("sat_to_gs", {}).get(c.client_id, []):
                self._gs_selection_counts[gs_id] = (
                    self._gs_selection_counts.get(gs_id, 0) + 1
                )

        return [c.client_id for c in selected]


class SmoothCappedSelector(CappedSelector):
    """
    带时序平滑约束的选择器 (Round 12)。

    在 GS=7 这类临界配置中限制相邻轮次卫星频繁切换，
    抑制输出结果震荡。

    Parameters
    ----------
    max_count : int
        最多选择的客户端数。
    min_clients : int
        最少参与客户端数。
    smoothness_factor : float
        平滑权重 (0=无约束, 1=完全保留上一轮选择)。
    seed : int | None
        随机种子。
    """

    def __init__(
        self,
        max_count: int = 3,
        min_clients: int = 1,
        smoothness_factor: float = 0.3,
        seed: int | None = None,
    ):
        super().__init__(max_count=max_count, min_clients=min_clients, seed=seed)
        self.smoothness_factor = smoothness_factor
        self._last_selection: list[int] = []
        self._switch_count: int = 0

    @property
    def switch_count(self) -> int:
        return self._switch_count

    def select(
        self,
        clients: list[ClientState],
        round_num: int,
        **kwargs: Any,
    ) -> list[int]:
        connected = [c for c in clients if c.is_connected]
        if not connected:
            self._last_selection = []
            return []

        n_select = max(self.min_clients, len(connected))
        n_select = min(n_select, self.max_count)
        n_select = min(n_select, len(connected))

        connected_ids = {c.client_id for c in connected}

        # 保留上一轮仍在线的客户端
        carry_over = [
            cid for cid in self._last_selection if cid in connected_ids
        ]

        if self.smoothness_factor > 0 and carry_over and n_select > 1:
            # 保留部分上一轮选择
            keep_count = max(1, int(n_select * self.smoothness_factor))
            keep_count = min(keep_count, len(carry_over))
            keep = carry_over[:keep_count]

            # 剩余名额随机补充
            remaining_connected = [
                c for c in connected if c.client_id not in keep
            ]
            need = n_select - len(keep)
            if need > 0 and remaining_connected:
                new_picks = self._rng.sample(
                    remaining_connected, min(need, len(remaining_connected))
                )
                keep.extend(c.client_id for c in new_picks)
            result = keep[:n_select]
        else:
            selected = self._rng.sample(connected, n_select)
            result = [c.client_id for c in selected]

        # 统计切换次数
        if self._last_selection:
            old_set = set(self._last_selection)
            new_set = set(result)
            self._switch_count += len(old_set.symmetric_difference(new_set)) // 2

        self._last_selection = result
        return result


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
            for _i, (agg_w, client_w) in enumerate(
                zip(aggregated, update.weights)
            ):
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
