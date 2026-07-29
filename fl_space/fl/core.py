"""
联邦学习核心抽象 — 四个可插拔组件接口

将 FL 算法分解为四个独立可替换的维度：
    1. ClientSelector  — 每轮选择哪些客户端参与
    2. LocalTrainer    — 客户端本地训练（epoch 次数 + 训练逻辑）
    3. Aggregator      — 何时聚合、如何聚合
    4. Evaluator       — 如何评估全局模型

用户可通过继承这些 ABC 实现自己的算法变体，
然后将四个组件组合成一个完整算法。

设计原则：
    - 每个组件接口只依赖基本数据结构（list, dict, 模型参数），
      不依赖特定 FL 框架
    - 组件之间通过明确的输入/输出解耦
    - 新增算法只需实现不同的组件组合，无需修改框架代码
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# ── 通用数据结构 ──────────────────────────────────────────────


@dataclass
class ClientState:
    """
    单个客户端的运行时状态。

    Attributes
    ----------
    client_id : int
        客户端唯一标识（对应卫星编号）。
    data_size : int
        该客户端持有的训练样本数。
    is_connected : bool
        当前时刻是否可通信（由通信调度器设置）。
    last_update_round : int
        该客户端上次参与聚合的轮次（用于 staleness 计算）。
    """
    client_id: int
    data_size: int = 100
    is_connected: bool = False
    last_update_round: int = -1


@dataclass
class ClientUpdate:
    """
    客户端提交的本地更新。

    Attributes
    ----------
    client_id : int
        来源客户端。
    weights : list
        本地训练后的模型参数（与全局模型同结构）。
    data_size : int
        该客户端的数据量（用于加权聚合）。
    train_loss : float
        本地训练最终损失。
    round_num : int
        对应的全局轮次。
    """
    client_id: int
    weights: list[Any]
    data_size: int
    train_loss: float
    round_num: int


@dataclass
class FLRoundResult:
    """
    单轮 FL 训练的结果。

    Attributes
    ----------
    round_num : int
        轮次编号。
    num_clients : int
        本轮参与的客户端数量。
    train_loss : float
        本轮平均训练损失（参与客户端均值）。
    eval_metrics : dict
        评估指标，如 {"accuracy": 0.85, "loss": 0.42}。
    timeslot : int
        本轮结束时对应的模拟时间槽编号。
    timeslot_start : int
        本轮开始时的模拟时间槽编号。
    time_breakdown : dict | None
        本轮时间分解（由 TimeModel 产生），None 表示未启用。
    """
    round_num: int
    num_clients: int
    train_loss: float
    eval_metrics: dict[str, float] = field(default_factory=dict)
    timeslot: int = 0
    timeslot_start: int = 0
    time_breakdown: dict[str, Any] | None = None


# ── 四个核心抽象接口 ──────────────────────────────────────────


class ClientSelector(ABC):
    """
    客户端选择策略抽象。

    决定每轮 FL 训练中哪些客户端参与。
    不同算法可使用不同的选择策略：
        - FedAvg / FedProx：随机采样固定比例
        - FedBuff：所有可连接客户端（异步）

    使用方式：
        继承此类并实现 select()，然后传入 FLServer。
    """

    @abstractmethod
    def select(
        self,
        clients: list[ClientState],
        round_num: int,
        **kwargs: Any,
    ) -> list[int]:
        """
        从客户端列表中选出本轮参与的客户端。

        Parameters
        ----------
        clients : list[ClientState]
            所有客户端状态列表（含连接状态）。
        round_num : int
            当前全局轮次。

        Returns
        -------
        list[int]
            被选中的客户端 ID 列表。
        """
        ...


class LocalTrainer(ABC):
    """
    本地训练策略抽象。

    控制客户端本地训练的 epoch 次数和训练逻辑。
    不同算法的核心差异在此体现：
        - FedAvg：固定 E 次 SGD
        - FedProx：E 次带有 proximal term 的 SGD
        - FedBuff：异步训练（epoch 次数可动态调整）

    使用方式：
        继承此类并实现 train()，传入模型和数据。
    """

    @abstractmethod
    def train(
        self,
        client_id: int,
        model: Any,
        train_loader: Any,
        global_weights: list[Any],
        round_num: int,
        **kwargs: Any,
    ) -> ClientUpdate:
        """
        在客户端本地执行训练。

        Parameters
        ----------
        client_id : int
            客户端 ID。
        model : Any
            模型实例（将被深拷贝用于本地训练）。
        train_loader : Any
            本地训练数据的 DataLoader。
        global_weights : list
            全局模型的当前参数（用于 proximal term 等）。
        round_num : int
            当前全局轮次。

        Returns
        -------
        ClientUpdate
            包含本地训练后的权重、损失等信息。
        """
        ...


class Aggregator(ABC):
    """
    聚合策略抽象。

    控制两个维度：
        - 何时触发聚合（同步等待所有客户端 vs 异步缓冲区满）
        - 如何聚合客户端更新（加权平均 vs 中位数 vs 其他）

    使用方式：
        继承此类并实现 should_aggregate() 和 aggregate()。
    """

    @abstractmethod
    def should_aggregate(
        self,
        collected_updates: list[ClientUpdate],
        round_num: int,
        **kwargs: Any,
    ) -> bool:
        """
        判断当前是否应该执行聚合。

        Parameters
        ----------
        collected_updates : list[ClientUpdate]
            已收集到的客户端更新列表。
        round_num : int
            当前全局轮次。

        Returns
        -------
        bool
            True 表示应触发聚合。
        """
        ...

    @abstractmethod
    def aggregate(
        self,
        global_weights: list[Any],
        updates: list[ClientUpdate],
        round_num: int,
        **kwargs: Any,
    ) -> list[Any]:
        """
        聚合客户端更新，生成新的全局模型参数。

        Parameters
        ----------
        global_weights : list
            当前全局模型参数。
        updates : list[ClientUpdate]
            本轮收集的客户端更新。
        round_num : int
            当前全局轮次。

        Returns
        -------
        list
            聚合后的新全局模型参数。
        """
        ...


class Evaluator(ABC):
    """
    评估策略抽象。

    控制如何评估全局模型的性能。
    可扩展为不同的评估指标、数据集或频率。

    使用方式：
        继承此类并实现 evaluate()。
    """

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        test_loader: Any,
        round_num: int,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        评估全局模型。

        Parameters
        ----------
        model : Any
            当前全局模型实例。
        test_loader : Any
            测试数据 DataLoader。
        round_num : int
            当前全局轮次。

        Returns
        -------
        dict[str, float]
            评估指标字典，如 {"accuracy": 0.85, "loss": 0.42}。
        """
        ...
