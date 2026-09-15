"""
示例神经网络模型 — 用于联邦学习实验

提供两个标准模型：
    - MLP     : 用于 MNIST / Fashion-MNIST 等数据集
    - SimpleCNN: 用于 CIFAR-10 / CIFAR-100 等数据集

所有模型遵循 PyTorch nn.Module 接口，
可通过 get_model() 工厂函数按名称获取。

PyTorch 为可选依赖，使用 try/except 静默回退。
"""

from __future__ import annotations

from typing import Any

# PyTorch 为可选依赖
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # 占位类型，在 torch 不可用时不报错
    class _Placeholder:
        pass

    nn = _Placeholder()  # type: ignore


# ── 模型定义 ──────────────────────────────────────────────────


if TORCH_AVAILABLE:

    class MLP(nn.Module):
        """
        简单多层感知机。

        适用于 MNIST (28×28 → 784) 等平坦输入数据集。

        Parameters
        ----------
        input_dim : int
            输入维度，默认 784 (MNIST)。
        hidden_dims : list[int]
            隐藏层维度列表，默认 [128, 64]。
        num_classes : int
            分类类别数，默认 10。
        dropout : float
            Dropout 比例，默认 0.0。
        """

        def __init__(
            self,
            input_dim: int = 784,
            hidden_dims: list[int] | None = None,
            num_classes: int = 10,
            dropout: float = 0.0,
        ):
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [128, 64]

            layers: list[nn.Module] = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, num_classes))

            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """前向传播。"""
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            return self.net(x)


    class SimpleCNN(nn.Module):
        """
        简单卷积神经网络。

        适用于 CIFAR-10 (3×32×32) 等图像数据集。

        结构: Conv → Conv → Pool → Conv → Conv → Pool → FC → FC

        Parameters
        ----------
        num_classes : int
            分类类别数，默认 10。
        in_channels : int
            输入通道数，默认 3 (RGB)。
        dropout : float
            Dropout 比例，默认 0.0。
        """

        def __init__(
            self,
            num_classes: int = 10,
            in_channels: int = 3,
            dropout: float = 0.0,
        ):
            super().__init__()

            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)

            self.flatten_dim = 64 * 8 * 8  # 输入 32×32 后两次 pool 得到 8×8
            self.fc1 = nn.Linear(self.flatten_dim, 256)
            self.fc2 = nn.Linear(256, num_classes)

            self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """前向传播。"""
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool1(x)
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = self.pool2(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.fc2(x)
            return x

else:
    MLP = None  # type: ignore
    SimpleCNN = None  # type: ignore


# ── 工厂函数 ──────────────────────────────────────────────────


_MODEL_REGISTRY: dict[str, type] = {}


def _build_registry() -> None:
    """构建模型注册表（仅在 torch 可用时）。"""
    if not TORCH_AVAILABLE:
        return
    _MODEL_REGISTRY.update({
        "mlp": MLP,
        "simplecnn": SimpleCNN,
    })


def get_model(
    name: str,
    **kwargs: Any,
) -> Any:
    """
    按名称获取模型实例。

    Parameters
    ----------
    name : str
        模型名称，支持: "mlp", "simplecnn"。
    **kwargs
        传递给模型构造函数的参数。

    Returns
    -------
    nn.Module
        模型实例。

    Raises
    ------
    ImportError
        当 torch 未安装时。
    ValueError
        当模型名称不存在时。
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch 未安装，无法创建模型。请运行: pip install torch"
        )
    if not _MODEL_REGISTRY:
        _build_registry()

    name_lower = name.lower()
    if name_lower not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"未知模型 '{name}'，可用: {available}"
        )
    return _MODEL_REGISTRY[name_lower](**kwargs)


def list_models() -> list[str]:
    """
    列出所有可用模型名称。

    Returns
    -------
    list[str]
        模型名称列表。
    """
    if not TORCH_AVAILABLE:
        return []
    if not _MODEL_REGISTRY:
        _build_registry()
    return list(_MODEL_REGISTRY.keys())


def register_model(name: str, model_cls: type) -> None:
    """
    注册自定义模型。

    用户可调用此函数将自己实现的模型注册到框架中。

    Parameters
    ----------
    name : str
        模型名称（不区分大小写）。
    model_cls : type
        nn.Module 子类。

    使用示例::

        class MyResNet(nn.Module):
            ...

        register_model("my_resnet", MyResNet)
        model = get_model("my_resnet", num_classes=100)
    """
    if not _MODEL_REGISTRY:
        _build_registry()
    _MODEL_REGISTRY[name.lower()] = model_cls
