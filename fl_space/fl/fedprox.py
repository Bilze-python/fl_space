"""
FedProx 算法 — 联邦近端优化 (Federated Proximal)

论文: "Federated Optimization in Heterogeneous Networks"
      (Li et al., MLSys 2020)

与 FedAvg 的关键差异：
    - 本地训练时在损失函数中加入 proximal term:
      L_prox = L(w) + (μ/2) * ||w - w_global||²
    - 这约束了本地更新不要偏离全局模型太远，
      有效处理数据/系统异构性问题。
    - μ = 0 时退化为标准 FedAvg。

组件复用：
    客户端选择器 → 复用 FedAvg 的 RandomSelector
    聚合器       → 复用 FedAvg 的 SyncWeightedAggregator
    评估器       → 复用 FedAvg 的 StandardEvaluator
    本地训练器   → 新增 ProximalTrainer（仅此不同）
"""

from __future__ import annotations

from collections import deque
import copy
from typing import Any

from fl_space.fl.core import (
    Aggregator,
    ClientSelector,
    ClientUpdate,
    Evaluator,
    LocalTrainer,
)
from fl_space.fl.fedavg import (
    EarliestReturnSelector,
    RandomSelector,
    StandardEvaluator,
    SyncWeightedAggregator,
)

# PyTorch 可选依赖
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── FedProx 特有的本地训练器 ─────────────────────────────────


class ProximalTrainer(LocalTrainer):
    """
    FedProx 本地训练器。

    在标准交叉熵损失上添加 proximal term:
        L_total = L_CE(w) + (μ/2) * ||w - w_global||²

    这使得本地更新不会因为数据异构而偏离全局模型太远。

    Parameters
    ----------
    local_epochs : int
        每轮本地训练的 epoch 数，默认 5。
    batch_size : int
        训练 batch size，默认 32。
    learning_rate : float
        学习率，默认 0.01。
    mu : float
        近端项系数 μ。μ 越大，本地更新越接近全局模型。
        μ = 0 退化为 FedAvg。
    device : str
        计算设备，默认 "cpu"。
    """

    def __init__(
        self,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        mu: float = 0.01,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("FedProx 需要 PyTorch，请运行: pip install torch")

        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mu = mu
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
        执行 FedProx 本地训练（含 proximal term）。
        """
        local_model = copy.deepcopy(model)
        local_model.to(self.device)
        local_model.train()

        # 保存全局参数作为 proximal term 的锚点
        global_params = [w.clone().detach().to(self.device) for w in global_weights]

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.learning_rate,
        )

        data_size = len(train_loader.dataset)  # type: ignore
        total_loss = 0.0

        actual_epochs = max(1, int(kwargs.get("local_epochs_override", self.local_epochs)))
        for _epoch in range(actual_epochs):
            epoch_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = local_model(data)

                # 标准交叉熵损失
                ce_loss = criterion(output, target)

                # Proximal term: (μ/2) * Σ||w_local - w_global||²
                proximal_term = 0.0
                for local_param, global_param in zip(
                    local_model.parameters(), global_params
                ):
                    proximal_term += torch.sum(
                        (local_param - global_param) ** 2
                    )

                loss = ce_loss + (self.mu / 2.0) * proximal_term

                loss.backward()
                optimizer.step()
                epoch_loss += ce_loss.item()  # 记录纯 CE 损失，不含 proximal

            total_loss += epoch_loss

        avg_loss = total_loss / actual_epochs

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
            metadata={"actual_local_epochs": actual_epochs},
        )


# ── FedProxSat: 准确率感知自适应 μ ──────────────────────────────


class AdaptiveProximalTrainer(ProximalTrainer):
    """
    FedProxSat 自适应近端训练器。

    在 ProximalTrainer 基础上增加准确率感知的动态 μ 调度：

    - 当最近 N 轮准确率震荡 ≥ oscillation_threshold 时，
      加强约束（μ↑），防止客户端漂移破坏全局模型。
    - 当准确率趋于稳定时，
      放松约束（μ↓），允许模型更快探索。

    类似于 PID 控制器思想：震荡大 → 增加阻尼，稳定 → 减小阻尼。

    Parameters
    ----------
    base_mu : float
        基础近端系数（用户设定），默认 0.01。
    mu_min : float
        μ 下限（稳定时不会低于此值），默认 0.001。
    mu_max : float
        μ 上限（震荡时不会超过此值），默认 1.0。
    oscillation_threshold : float
        准确率震荡阈值。当窗口内 max-min ≥ 此值时触发加强约束。
        默认 0.1（10个百分点）。
    stability_threshold : float
        稳定阈值。当窗口内 max-min < 此值时触发放松约束。
        默认 0.03（3个百分点）。
    window_size : int
        滑动窗口大小（最近 N 轮准确率），默认 5。
    local_epochs, batch_size, learning_rate, device :
        与 ProximalTrainer 相同。
    """

    def __init__(
        self,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        base_mu: float = 0.01,
        mu_min: float = 0.001,
        mu_max: float = 1.0,
        oscillation_threshold: float = 0.1,
        stability_threshold: float = 0.03,
        window_size: int = 5,
        device: str = "cpu",
    ):
        super().__init__(
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            mu=base_mu,  # 初始 μ = base_mu
            device=device,
        )
        self.base_mu = base_mu
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.oscillation_threshold = oscillation_threshold
        self.stability_threshold = stability_threshold
        self._acc_window: deque[float] = deque(maxlen=window_size)
        self._mu_history: list[float] = []  # 记录 μ 变化历史

    def update_accuracy(self, acc: float) -> float:
        """
        根据最新准确率更新有效 μ。

        由 FLServer 在每轮评估后调用。

        Parameters
        ----------
        acc : float
            当前轮聚合后的全局准确率。

        Returns
        -------
        float
            更新后的有效 μ。
        """
        self._acc_window.append(acc)

        if len(self._acc_window) < 3:
            # 前几轮数据不够，使用 base_mu
            self.mu = self.base_mu
            return self.mu

        oscillation = max(self._acc_window) - min(self._acc_window)

        if oscillation >= self.oscillation_threshold:
            # 震荡大 → 加强约束（μ 增大）
            # 因子: 1 + (oscillation/threshold - 1) * 5, 最大 10x
            excess = (oscillation - self.oscillation_threshold) / self.oscillation_threshold
            factor = min(1.0 + excess * 5.0, 10.0)
            self.mu = min(self.base_mu * factor, self.mu_max)

        elif oscillation < self.stability_threshold and len(self._acc_window) >= self._acc_window.maxlen:
            # 非常稳定且窗口已满 → 逐步放松约束
            self.mu = max(self.mu * 0.8, self.mu_min)

        else:
            # 正常范围 → 使用 base_mu
            self.mu = self.base_mu

        self._mu_history.append(self.mu)
        return self.mu

    @property
    def effective_mu(self) -> float:
        """当前生效的 μ 值。"""
        return self.mu

    @property
    def mu_stats(self) -> dict[str, Any]:
        """μ 调度统计信息。"""
        accs = list(self._acc_window)
        return {
            "current_mu": round(self.mu, 6),
            "base_mu": self.base_mu,
            "mu_history": [round(m, 6) for m in self._mu_history[-20:]],
            "acc_window": [round(a, 4) for a in accs],
            "oscillation": round(max(accs) - min(accs), 4) if len(accs) >= 2 else 0,
            "window_size": len(accs),
        }


# ── 便捷构建函数 ──────────────────────────────────────────────


def create_fedprox_components(
    fraction: float = 0.5,
    min_clients: int = 2,
    local_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    mu: float = 0.01,
    device: str = "cpu",
    seed: int | None = None,
    selection_strategy: str = "random",
) -> tuple[ClientSelector, LocalTrainer, Aggregator, Evaluator]:
    """
    一键创建 FedProx 的四件套组件。

    注意：selector、aggregator、evaluator 与 FedAvg 完全相同，
    只有 trainer 使用 ProximalTrainer。

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
    mu : float
        近端项系数。μ=0 即退化为 FedAvg。
    device : str
        计算设备。
    seed : int | None
        随机种子。

    Returns
    -------
    tuple
        (selector, trainer, aggregator, evaluator)

    使用示例::

        from fl_space.fl.fedprox import create_fedprox_components

        selector, trainer, aggregator, evaluator = create_fedprox_components(
            fraction=0.5,
            mu=0.1,  # 更强的近端约束
            device="cuda",
        )
    """
    if selection_strategy == "earliest_return":
        selector = EarliestReturnSelector(fraction=fraction, min_clients=min_clients)
    else:
        selector = RandomSelector(
            fraction=fraction,
            min_clients=min_clients,
            seed=seed,
        )
    trainer = ProximalTrainer(
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mu=mu,
        device=device,
    )
    # min_updates=1: 同步 FL 中训练完全部选中客户端后才聚合，
    # 收到的 update 数 = 实际参与客户端数，不需要额外门槛
    aggregator = SyncWeightedAggregator(min_updates=1)
    evaluator = StandardEvaluator(device=device)

    return selector, trainer, aggregator, evaluator
