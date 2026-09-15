"""
虚拟时间模型 — 可插拔的时间投影系统

将 FL 训练中的虚拟时间建模抽象为可替换组件，支持三种模式：
    - SlotTimeModel    : 方案A+C — timeslot 级粗粒度 + 时间分解输出
    - PhysicsTimeModel : 方案B   — 双时钟物理级精度（秒级 + 硬件感知）
    - 自定义模型       : 继承 TimeModel ABC 实现任意时间投影

设计原则：
    - 与 FL 算法解耦：时间模型只关心"耗时多少"，不关心训练逻辑
    - 可扩展：用户可通过文件路径导入自定义实现
    - CLI 可选：--time-model slot|physics|path/to/custom.py:ClassName
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any

# ── 时间分解数据结构 ──────────────────────────────────────────


@dataclass
class TimeBreakdown:
    """
    单轮 FL 训练的时间分解。

    记录本轮各阶段消耗的虚拟时间，便于分析瓶颈。

    Attributes
    ----------
    wait_distribution : int
        等待分发窗口的 timeslot 数。
    download : int
        模型下载耗时（timeslots）。
    train : int
        本地训练耗时（timeslots）。
    wait_return : int
        等待卫星返回窗口的 timeslot 数。
    upload : int
        模型上传耗时（timeslots）。
    aggregation : int
        聚合耗时（timeslots）。
    total : int
        本轮总 timeslot 数。
    per_satellite : dict[int, dict]
        每个卫星的细分耗时。
    """
    wait_distribution: int = 0
    download: int = 0
    train: int = 0
    wait_return: int = 0
    upload: int = 0
    aggregation: int = 0
    total: int = 0
    per_satellite: dict[int, dict[str, int]] = field(default_factory=dict)

    def summary_str(self) -> str:
        """生成单行摘要字符串。"""
        parts = []
        if self.wait_distribution > 0:
            parts.append(f"等待分发:{self.wait_distribution}")
        if self.download > 0:
            parts.append(f"下载:{self.download}")
        if self.train > 0:
            parts.append(f"训练:{self.train}")
        if self.wait_return > 0:
            parts.append(f"等待返回:{self.wait_return}")
        if self.upload > 0:
            parts.append(f"上传:{self.upload}")
        if self.aggregation > 0:
            parts.append(f"聚合:{self.aggregation}")
        parts.append(f"总计:{self.total}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """导出为字典。"""
        return {
            "wait_distribution": self.wait_distribution,
            "download": self.download,
            "train": self.train,
            "wait_return": self.wait_return,
            "upload": self.upload,
            "aggregation": self.aggregation,
            "total": self.total,
            "per_satellite": {
                str(k): v for k, v in self.per_satellite.items()
            },
        }


# ── 抽象基类 ──────────────────────────────────────────────────


class TimeModel(ABC):
    """
    虚拟时间模型抽象基类。

    子类需实现三个时间计算方法，框架自动调用。

    使用方式：
        1. 使用内置模型: TimeModel.create("slot") 或 TimeModel.create("physics")
        2. 自定义实现:   继承此类，实现所有抽象方法
        3. 从文件加载:   TimeModel.create("path/to/my_model.py:MyModel")
    """

    @abstractmethod
    def compute_train_slots(
        self,
        client_id: int,
        num_samples: int,
        num_epochs: int,
        **kwargs: Any,
    ) -> int:
        """
        计算单客户端训练所需 timeslot 数。

        Parameters
        ----------
        client_id : int
            客户端 ID（支持异构硬件）。
        num_samples : int
            该客户端持有的训练样本数。
        num_epochs : int
            本地训练 epoch 数。

        Returns
        -------
        int
            训练消耗的 timeslot 数。
        """
        ...

    @abstractmethod
    def compute_download_slots(
        self,
        model_size_bytes: int,
        **kwargs: Any,
    ) -> int:
        """
        计算模型下载所需 timeslot 数。

        Parameters
        ----------
        model_size_bytes : int
            模型参数字节数。

        Returns
        -------
        int
            下载消耗的 timeslot 数。
        """
        ...

    @abstractmethod
    def compute_upload_slots(
        self,
        model_size_bytes: int,
        **kwargs: Any,
    ) -> int:
        """
        计算模型上传所需 timeslot 数。

        Parameters
        ----------
        model_size_bytes : int
            模型参数字节数。

        Returns
        -------
        int
            上传消耗的 timeslot 数。
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """时间模型名称。"""
        ...

    @abstractmethod
    def get_config_dict(self) -> dict[str, Any]:
        """导出当前配置为字典。"""
        ...

    def slots_to_display(self, slots: int, timeslot_duration_min: float = 1.0) -> str:
        """
        将 timeslot 数转为可读时间字符串。

        Parameters
        ----------
        slots : int
            timeslot 数量。
        timeslot_duration_min : float
            每 timeslot 的分钟数。

        Returns
        -------
        str
            如 "30s", "5min", "2h15min"。
        """
        total_min = slots * timeslot_duration_min
        if total_min < 1:
            return f"{int(total_min * 60)}s"
        elif total_min < 60:
            return f"{total_min:.0f}min"
        else:
            h = int(total_min // 60)
            m = int(total_min % 60)
            if m == 0:
                return f"{h}h"
            return f"{h}h{m}min"

    # ── 工厂方法 ──────────────────────────────────────────

    _BUILTIN: dict[str, type[TimeModel]] = {}

    @classmethod
    def register(cls, name: str, model_cls: type[TimeModel]) -> None:
        """注册内置时间模型。"""
        cls._BUILTIN[name] = model_cls

    @classmethod
    def create(cls, spec: str, **kwargs: Any) -> TimeModel:
        """
        工厂方法：根据规格字符串创建时间模型实例。

        支持三种形式：
            - "slot"       → SlotTimeModel
            - "physics"    → PhysicsTimeModel
            - "path/to/file.py:ClassName" → 自定义导入

        Parameters
        ----------
        spec : str
            时间模型规格。
        **kwargs
            传递给构造函数的参数。

        Returns
        -------
        TimeModel
            时间模型实例。

        Raises
        ------
        ValueError
            规格无效或导入失败时。
        """
        # 1. 内置名称
        if spec in cls._BUILTIN:
            return cls._BUILTIN[spec](**kwargs)

        # 2. 文件路径导入: "path/to/file.py:ClassName"
        if ":" in spec or "/" in spec or "\\" in spec:
            return cls._import_from_file(spec, **kwargs)

        # 3. 未知
        available = list(cls._BUILTIN.keys()) + ["<path/to/file.py:ClassName>"]
        raise ValueError(
            f"未知时间模型: '{spec}'。"
            f"可用: {available}"
        )

    @classmethod
    def _import_from_file(cls, spec: str, **kwargs: Any) -> TimeModel:
        """
        从文件路径动态导入时间模型类。

        格式: "path/to/file.py:ClassName"
        """
        import importlib.util
        import os
        import sys

        if ":" in spec:
            filepath, class_name = spec.rsplit(":", 1)
        else:
            filepath = spec
            class_name = None

        filepath = os.path.abspath(filepath)
        if not os.path.exists(filepath):
            raise ValueError(f"时间模型文件不存在: {filepath}")

        module_name = f"_time_model_custom_{hash(filepath) % 100000}"
        spec_obj = importlib.util.spec_from_file_location(module_name, filepath)
        if spec_obj is None or spec_obj.loader is None:
            raise ValueError(f"无法加载模块: {filepath}")

        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[module_name] = module
        spec_obj.loader.exec_module(module)

        # 查找 TimeModel 子类
        if class_name:
            if not hasattr(module, class_name):
                raise ValueError(
                    f"模块 {filepath} 中未找到类 '{class_name}'"
                )
            candidate = getattr(module, class_name)
        else:
            # 自动查找第一个 TimeModel 子类
            candidates = []
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, TimeModel)
                    and obj is not TimeModel
                ):
                    candidates.append(obj)
            if not candidates:
                raise ValueError(
                    f"模块 {filepath} 中未找到 TimeModel 子类。"
                    f"请使用 'path:ClassName' 格式指定类名。"
                )
            candidate = candidates[0]

        return candidate(**kwargs)

    @classmethod
    def list_builtin(cls) -> list[str]:
        """列出所有内置时间模型名称。"""
        return list(cls._BUILTIN.keys())


# ── 方案 A+C：SlotTimeModel ──────────────────────────────────


@dataclass
class SlotTimeModel(TimeModel):
    """
    方案A+C — timeslot 级粗粒度时间投影 + 时间分解。

    所有时间成本以 timeslot（时间槽）为单位，配置简单直观。
    训练和传输时间通过固定倍率估算，适合：
        - 研究轨道几何对 FL 收敛的影响
        - 对比不同算法在通信受限下的表现
        - 快速原型验证

    Parameters
    ----------
    slots_per_epoch : int
        每个本地 epoch 消耗的 timeslot 数（默认 1）。
    slots_per_mb_down : int
        下载每 MB 模型消耗的 timeslot 数（默认 0，瞬时）。
    slots_per_mb_up : int
        上传每 MB 模型消耗的 timeslot 数（默认 0，瞬时）。
    timeslot_duration_min : float
        每 timeslot 的分钟数（用于显示转换，默认 1.0）。

    使用示例::

        # 训练有成本（每epoch 2 slots），传输无成本
        tm = SlotTimeModel(slots_per_epoch=2)

        # 完全零成本（等价旧行为）
        tm = SlotTimeModel(slots_per_epoch=0)

        # 训练 + 传输都有成本
        tm = SlotTimeModel(slots_per_epoch=1, slots_per_mb_down=1, slots_per_mb_up=2)
    """

    slots_per_epoch: int = 0
    slots_per_mb_down: int = 0
    slots_per_mb_up: int = 0
    timeslot_duration_min: float = 1.0

    @property
    def name(self) -> str:
        return "slot"

    def compute_train_slots(
        self,
        client_id: int,
        num_samples: int,
        num_epochs: int,
        **kwargs: Any,
    ) -> int:
        return num_epochs * self.slots_per_epoch

    def compute_download_slots(
        self,
        model_size_bytes: int,
        **kwargs: Any,
    ) -> int:
        if self.slots_per_mb_down <= 0:
            return 0
        mb = model_size_bytes / (1024 * 1024)
        return max(1, math.ceil(mb * self.slots_per_mb_down))

    def compute_upload_slots(
        self,
        model_size_bytes: int,
        **kwargs: Any,
    ) -> int:
        if self.slots_per_mb_up <= 0:
            return 0
        mb = model_size_bytes / (1024 * 1024)
        return max(1, math.ceil(mb * self.slots_per_mb_up))

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "type": "slot",
            "slots_per_epoch": self.slots_per_epoch,
            "slots_per_mb_down": self.slots_per_mb_down,
            "slots_per_mb_up": self.slots_per_mb_up,
            "timeslot_duration_min": self.timeslot_duration_min,
        }


# ── 方案 B：PhysicsTimeModel ─────────────────────────────────


@dataclass
class PhysicsTimeModel(TimeModel):
    """
    方案B — 双时钟物理级精度时间投影。

    基于硬件参数（算力、带宽）将训练和传输耗时精确计算到秒级，
    再转换为 timeslot。适合：
        - 研究不同硬件平台对 FL 收敛的影响
        - 对比带宽/算力瓶颈
        - 撰写物理真实性要求高的论文

    Parameters
    ----------
    compute_gflops : float
        卫星计算能力（GFLOPS FP32），默认 10.0（树莓派级别）。
    downlink_mbps : float
        星地下行带宽（Mbps），默认 10.0。
    uplink_mbps : float
        星地上行带宽（Mbps），默认 1.0。
    flops_per_sample_forward : float
        每样本前向传播 FLOPs 数（默认 109184，对应 MLP 784→128→64→10）。
    timeslot_duration_min : float
        每 timeslot 的分钟数（默认 1.0）。

    注意：
        - 反向传播 FLOPs 估算为前向的 2 倍
        - 最终结果向上取整到 timeslot

    使用示例::

        # 高性能边缘设备
        tm = PhysicsTimeModel(compute_gflops=472, downlink_mbps=100, uplink_mbps=20)

        # 低功耗卫星
        tm = PhysicsTimeModel(compute_gflops=0.2, downlink_mbps=1, uplink_mbps=0.1)

        # 仅计算训练时间，忽略传输
        tm = PhysicsTimeModel(downlink_mbps=0, uplink_mbps=0)
    """

    compute_gflops: float = 10.0
    downlink_mbps: float = 10.0
    uplink_mbps: float = 1.0
    flops_per_sample_forward: float = 109184.0  # MLP 784→128→64→10
    timeslot_duration_min: float = 1.0
    _backward_multiplier: float = 2.0

    @property
    def name(self) -> str:
        return "physics"

    def _seconds_to_slots(self, seconds: float) -> int:
        """秒转 timeslot（向上取整）。"""
        slot_seconds = self.timeslot_duration_min * 60.0
        return max(0, math.ceil(seconds / slot_seconds))

    def compute_train_slots(
        self,
        client_id: int,
        num_samples: int,
        num_epochs: int,
        **kwargs: Any,
    ) -> int:
        """
        基于 FLOPs + 硬件算力计算训练时间。

        total_flops = num_samples × num_epochs × flops_per_sample_forward × (1 + backward_multiplier)
        train_seconds = total_flops / (compute_gflops × 1e9)
        """
        if self.compute_gflops <= 0:
            return 0
        flops_per_sample_total = self.flops_per_sample_forward * (
            1.0 + self._backward_multiplier
        )
        total_flops = num_samples * num_epochs * flops_per_sample_total
        train_seconds = total_flops / (self.compute_gflops * 1e9)
        return self._seconds_to_slots(train_seconds)

    def compute_download_slots(
        self,
        model_size_bytes: int,
        **kwargs: Any,
    ) -> int:
        """基于带宽计算下载时间。"""
        if self.downlink_mbps <= 0:
            return 0
        bits = model_size_bytes * 8
        seconds = bits / (self.downlink_mbps * 1e6)
        return self._seconds_to_slots(seconds)

    def compute_upload_slots(
        self,
        model_size_bytes: int,
        **kwargs: Any,
    ) -> int:
        """基于带宽计算上传时间。"""
        if self.uplink_mbps <= 0:
            return 0
        bits = model_size_bytes * 8
        seconds = bits / (self.uplink_mbps * 1e6)
        return self._seconds_to_slots(seconds)

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "type": "physics",
            "compute_gflops": self.compute_gflops,
            "downlink_mbps": self.downlink_mbps,
            "uplink_mbps": self.uplink_mbps,
            "flops_per_sample_forward": self.flops_per_sample_forward,
            "timeslot_duration_min": self.timeslot_duration_min,
        }


# ── 注册内置模型 ──────────────────────────────────────────────

TimeModel.register("slot", SlotTimeModel)
TimeModel.register("physics", PhysicsTimeModel)
