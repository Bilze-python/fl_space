"""
FL 实验预设配置

提供开箱即用的 FL 实验配置：
    - 算法预设：FedAvg / FedProx / FedBuff
    - 规模预设：小型 / 中型 / 大型（客户端数量差异）
    - 数据集预设：MNIST / CIFAR-10 对应的模型和参数

所有配置导出为 FLConfig 字典，可直接传入 FLServer。
"""

from __future__ import annotations

from typing import Any

from fl_space.fl.server import FLConfig

# ── 算法预设 ──────────────────────────────────────────────────


def fedavg_config(**overrides: Any) -> FLConfig:
    """
    标准 FedAvg 配置。

    可通过关键字参数覆盖任意字段。

    Parameters
    ----------
    **overrides
        覆盖默认配置的字段。

    Returns
    -------
    FLConfig
        FedAvg 配置对象。
    """
    defaults = {
        "algorithm": "fedavg",
        "num_rounds": 50,
        "num_clients": 10,
        "timeslots_per_round": 10,
        "fraction": 0.5,
        "local_epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.01,
        "classes_per_client": 2,
        "max_samples_per_client": 1000,
        "partition_strategy": "probability",
        "class_probability": 0.8,
        "preference_mode": "class_balanced",
        "preferred_clients_per_class": 1,
        "sample_cap_strategy": "preserve",
        "data_dir": "./data",
    }
    defaults.update(overrides)
    return FLConfig(**defaults)


def fedprox_config(**overrides: Any) -> FLConfig:
    """
    标准 FedProx 配置。

    Parameters
    ----------
    **overrides
        覆盖默认配置的字段。

    Returns
    -------
    FLConfig
        FedProx 配置对象。
    """
    defaults = {
        "algorithm": "fedprox",
        "num_rounds": 50,
        "num_clients": 10,
        "timeslots_per_round": 10,
        "fraction": 0.5,
        "local_epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.01,
        "mu": 0.01,
        "classes_per_client": 2,
        "max_samples_per_client": 1000,
        "partition_strategy": "probability",
        "class_probability": 0.8,
        "preference_mode": "class_balanced",
        "preferred_clients_per_class": 1,
        "sample_cap_strategy": "preserve",
        "data_dir": "./data",
    }
    defaults.update(overrides)
    return FLConfig(**defaults)


def fedbuff_config(**overrides: Any) -> FLConfig:
    """
    标准 FedBuff（异步）配置。

    Parameters
    ----------
    **overrides
        覆盖默认配置的字段。

    Returns
    -------
    FLConfig
        FedBuff 配置对象。
    """
    defaults = {
        "algorithm": "fedbuff",
        "num_rounds": 200,
        "num_clients": 10,
        "timeslots_per_round": 5,
        "fraction": 0.5,
        "local_epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.01,
        "buffer_size": 5,
        "staleness_weight": False,
        "classes_per_client": 2,
        "max_samples_per_client": 1000,
        "partition_strategy": "probability",
        "class_probability": 0.8,
        "preference_mode": "class_balanced",
        "preferred_clients_per_class": 1,
        "sample_cap_strategy": "preserve",
        "data_dir": "./data",
    }
    defaults.update(overrides)
    return FLConfig(**defaults)


# ── 规模预设 ──────────────────────────────────────────────────


SCALE_PRESETS: dict[str, dict[str, Any]] = {
    "small": {
        "num_clients": 5,
        "num_rounds": 30,
        "description": "小型实验 — 5 客户端，快速验证",
    },
    "medium": {
        "num_clients": 20,
        "num_rounds": 100,
        "description": "中型实验 — 20 客户端，标准评测",
    },
    "large": {
        "num_clients": 50,
        "num_rounds": 200,
        "description": "大型实验 — 50 客户端，大规模仿真",
    },
}


# ── 数据集预设 ────────────────────────────────────────────────


DATASET_PRESETS: dict[str, dict[str, Any]] = {
    "mnist": {
        "model": "mlp",
        "model_kwargs": {"input_dim": 784, "hidden_dims": [128, 64], "num_classes": 10},
        "description": "MNIST 手写数字 (28×28 灰度, 10类)",
    },
    "fashion_mnist": {
        "model": "mlp",
        "model_kwargs": {"input_dim": 784, "hidden_dims": [128, 64], "num_classes": 10},
        "description": "Fashion-MNIST (28×28 灰度, 10类)",
    },
    "cifar10": {
        "model": "simplecnn",
        "model_kwargs": {"num_classes": 10, "in_channels": 3},
        "description": "CIFAR-10 (32×32 彩色, 10类)",
    },
    "imagefolder": {
        "model": "simplecnn",
        "model_kwargs": {"num_classes": 2, "in_channels": 3},
        "description": "Custom ImageFolder dataset (data_dir/train and data_dir/test)",
    },
    "custom": {
        "model": "simplecnn",
        "model_kwargs": {"num_classes": 2, "in_channels": 3},
        "description": "Alias of imagefolder for user-provided datasets",
    },
}


# ── 组合预设 ──────────────────────────────────────────────────


def get_preset_config(
    algorithm: str = "fedavg",
    scale: str = "small",
    dataset: str = "mnist",
    **overrides: Any,
) -> FLConfig:
    """
    根据预设组合获取完整实验配置。

    Parameters
    ----------
    algorithm : str
        算法: "fedavg", "fedprox", "fedbuff"。
    scale : str
        规模: "small", "medium", "large"。
    dataset : str
        数据集: "mnist", "fashion_mnist", "cifar10"。
    **overrides
        覆盖默认配置的字段。

    Returns
    -------
    FLConfig
        组合后的完整配置。

    使用示例::

        from fl_space.fl.config import get_preset_config

        config = get_preset_config(
            algorithm="fedprox",
            scale="medium",
            dataset="cifar10",
            mu=0.1,  # 覆盖近端系数
            device="cuda",
        )
    """
    algorithm = algorithm.lower()
    scale = scale.lower()

    if algorithm == "fedavg":
        config = fedavg_config()
    elif algorithm == "fedprox":
        config = fedprox_config()
    elif algorithm == "fedbuff":
        config = fedbuff_config()
    else:
        raise ValueError(
            f"未知算法: '{algorithm}'，支持: fedavg, fedprox, fedbuff"
        )

    # 应用规模预设
    if scale in SCALE_PRESETS:
        for k, v in SCALE_PRESETS[scale].items():
            if k != "description" and hasattr(config, k):
                setattr(config, k, v)

    # 应用数据集预设
    # 应用覆盖
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    return config


# ── 已注册预设列表 ────────────────────────────────────────────


EXPERIMENT_PRESETS: list[dict[str, Any]] = [
    {
        "name": "fedavg_small_mnist",
        "config": lambda: get_preset_config("fedavg", "small", "mnist"),
        "description": "FedAvg + 小型 + MNIST — 入门基准",
    },
    {
        "name": "fedavg_medium_mnist",
        "config": lambda: get_preset_config("fedavg", "medium", "mnist"),
        "description": "FedAvg + 中型 + MNIST — 标准评测",
    },
    {
        "name": "fedprox_small_mnist",
        "config": lambda: get_preset_config("fedprox", "small", "mnist", mu=0.01),
        "description": "FedProx + 小型 + MNIST — 异构性对比",
    },
    {
        "name": "fedprox_medium_cifar10",
        "config": lambda: get_preset_config("fedprox", "medium", "cifar10", mu=0.01),
        "description": "FedProx + 中型 + CIFAR-10 — 图像分类对比",
    },
    {
        "name": "fedbuff_small_mnist",
        "config": lambda: get_preset_config("fedbuff", "small", "mnist", buffer_size=5),
        "description": "FedBuff + 小型 + MNIST — 异步基准",
    },
    {
        "name": "fedbuff_medium_cifar10",
        "config": lambda: get_preset_config(
            "fedbuff", "medium", "cifar10", buffer_size=5
        ),
        "description": "FedBuff + 中型 + CIFAR-10 — 异步图像分类",
    },
]


def list_presets() -> list[str]:
    """
    列出所有可用预设。

    Returns
    -------
    list[str]
        预设名称列表。
    """
    return [p["name"] for p in EXPERIMENT_PRESETS]
