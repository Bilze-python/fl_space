"""
FedLEO 指标计算 — 权重散度 + 系统时延

论文核心优化目标:
    min α·D(W_fedleo, W_centralized) + β·T_total

其中:
    - D(·,·): FedLEO 权重与集中式 ML 权重的散度（越小越好）
    - T_total: 系统总时延 = 卸载时延 + 训练时延 + 聚合时延
    - α, β: 权重系数
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# 权重散度
# ═══════════════════════════════════════════════════════════════


def compute_weight_divergence(
    weights_a: list[Any],
    weights_b: list[Any],
) -> float:
    """
    计算两组权重的 L2 散度（归一化）。

    D(a, b) = ||a - b||₂ / max(||b||₂, ε)

    论文将此项作为准确率的代理指标：散度越小，
    FedLEO 的聚合结果越接近理想集中式训练。

    Parameters
    ----------
    weights_a : list
        FedLEO 聚合后的模型权重。
    weights_b : list
        集中式 ML 训练的参考权重。

    Returns
    -------
    float
        归一化 L2 散度，越小越好。
    """
    if not TORCH_AVAILABLE:
        return float("inf")

    diff_sq = 0.0
    norm_b_sq = 0.0
    for wa, wb in zip(weights_a, weights_b):
        a = torch.as_tensor(wa, dtype=torch.float32)
        b = torch.as_tensor(wb, dtype=torch.float32)
        diff_sq += float(torch.sum((a - b) ** 2))
        norm_b_sq += float(torch.sum(b ** 2))

    if norm_b_sq < 1e-12:
        return math.sqrt(diff_sq) if diff_sq > 0 else 0.0
    return math.sqrt(diff_sq) / math.sqrt(norm_b_sq)


def compute_weight_divergence_single(
    weights: list[Any],
    reference_weights: list[Any],
) -> float:
    """计算单组权重相对于参考权重的散度（非对称版本）。"""
    return compute_weight_divergence(weights, reference_weights)


# ═══════════════════════════════════════════════════════════════
# 系统时延
# ═══════════════════════════════════════════════════════════════


def compute_system_delay(
    offload_time: float = 0.0,
    train_time: float = 0.0,
    intra_agg_time: float = 0.0,
    inter_agg_time: float = 0.0,
) -> float:
    """
    计算单轮 FedLEO 的系统总时延。

    T_total = T_offload + T_train + T_intra_agg + T_inter_agg

    Parameters
    ----------
    offload_time : float
        数据卸载阶段耗时 (timeslots 或秒)。
    train_time : float
        所有卫星本地训练的最大耗时。
    intra_agg_time : float
        同轨 Ring-Allreduce 聚合耗时。
    inter_agg_time : float
        跨轨 Ring-Allreduce 聚合耗时。

    Returns
    -------
    float
        系统总时延。
    """
    return offload_time + train_time + intra_agg_time + inter_agg_time


def compute_offload_delay(
    source_data_size: int,
    target_data_size: int,
    offload_ratio: float,
    bandwidth_mbps: float = 10.0,
    bytes_per_sample: int = 784,  # MNIST: 28×28×1
) -> float:
    """
    计算单次卸载的通信时延。

    卸载数据量 = offload_ratio × source_data_size × bytes_per_sample
    时延 = 数据量(bits) / 带宽(bps)

    Parameters
    ----------
    source_data_size : int
        源卫星的样本数。
    target_data_size : int
        目标卫星的样本数（用于判断是否需要卸载）。
    offload_ratio : float
        卸载比例 (0.0 ~ 1.0)。
    bandwidth_mbps : float
        ISL 带宽 (Mbps)。
    bytes_per_sample : int
        每样本字节数。

    Returns
    -------
    float
        卸载时延（秒）。
    """
    n_samples = int(source_data_size * offload_ratio)
    if n_samples <= 0 or bandwidth_mbps <= 0:
        return 0.0
    total_bits = n_samples * bytes_per_sample * 8
    return total_bits / (bandwidth_mbps * 1e6)


# ═══════════════════════════════════════════════════════════════
# 综合指标
# ═══════════════════════════════════════════════════════════════


@dataclass
class FedLEOMetrics:
    """单轮 FedLEO 实验的综合指标。"""

    round_num: int = 0
    # 时延 (timeslots)
    offload_delay: float = 0.0
    train_delay: float = 0.0
    intra_agg_delay: float = 0.0
    inter_agg_delay: float = 0.0
    total_delay: float = 0.0
    # 模型精度
    accuracy: float = 0.0
    train_loss: float = 0.0
    weight_divergence: float = 0.0
    # 卸载统计
    total_offloaded_samples: int = 0
    num_offload_actions: int = 0
    # 数据均衡度
    data_balance_entropy: float = 0.0
    # 额外信息
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def system_delay(self) -> float:
        """系统总时延 = 卸载 + 训练 + 同轨聚合 + 跨轨聚合"""
        return (
            self.offload_delay
            + self.train_delay
            + self.intra_agg_delay
            + self.inter_agg_delay
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_num,
            "accuracy": self.accuracy,
            "train_loss": self.train_loss,
            "weight_divergence": self.weight_divergence,
            "offload_delay": self.offload_delay,
            "train_delay": self.train_delay,
            "intra_agg_delay": self.intra_agg_delay,
            "inter_agg_delay": self.inter_agg_delay,
            "total_delay": self.total_delay,
            "total_offloaded_samples": self.total_offloaded_samples,
            "num_offload_actions": self.num_offload_actions,
            "data_balance_entropy": self.data_balance_entropy,
            **self.extra,
        }


def compute_data_balance_entropy(data_sizes: list[int]) -> float:
    """
    计算各卫星数据分布的均衡度（归一化熵）。

    熵越大数据分布越均衡，最大值 = log₂(N_clients)。

    Parameters
    ----------
    data_sizes : list[int]
        各卫星的样本数列表。

    Returns
    -------
    float
        归一化熵 ∈ [0, 1]，1 表示完全均衡。
    """
    if not data_sizes:
        return 0.0
    total = sum(data_sizes)
    if total == 0:
        return 0.0
    max_ent = math.log2(len(data_sizes))
    if max_ent == 0:
        return 1.0
    entropy = 0.0
    for s in data_sizes:
        if s > 0:
            p = s / total
            entropy -= p * math.log2(p)
    return entropy / max_ent
