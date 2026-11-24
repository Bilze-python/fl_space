"""独立通信层数据契约；距离 m、时间 s、速率 bit/s，参数均为假设。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


class Serializable:
    """无外部依赖的数据类 JSON 转换。"""

    def to_dict(self) -> dict[str, Any]:
        """导出基本类型字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        """构造并执行数据类自身校验；未知字段显式报错。"""
        return cls(**data)


def finite(value: float, name: str, minimum: float | None = None) -> None:
    """拒绝 NaN/Inf 及越界物理参数。"""
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} 必须有限且 >= {minimum}")


@dataclass(frozen=True)
class RFConfig(Serializable):
    """RF 一阶假设：理想跟踪、常数轴向增益、自由空间损耗、kTB 噪声。

    dynamic 使用效率折减 Shannon 上界并限速，非实测 MCS/BLER；fixed
    在相同可见性下使用固定速率，用于消融。带宽不等于数据速率。
    """

    mode: str = "dynamic"
    frequency_hz: float = 20e9
    bandwidth_hz: float = 10e6
    tx_power_w: float = 1.0
    tx_gain_dbi: float = 20.0
    rx_gain_dbi: float = 20.0
    system_loss_db: float = 2.0
    noise_temperature_k: float = 290.0
    noise_figure_db: float = 3.0
    efficiency: float = 0.5
    max_rate_bps: float = 20e6
    fixed_rate_bps: float = 10e6
    min_snr_db: float = -10.0
    acquisition_s: float = 0.2
    switching_s: float = 0.1
    source: str = "assumption"

    def __post_init__(self) -> None:
        if self.mode not in ("dynamic", "fixed"):
            raise ValueError("mode 必须为 dynamic 或 fixed")
        for name in (
            "frequency_hz",
            "bandwidth_hz",
            "tx_power_w",
            "noise_temperature_k",
            "max_rate_bps",
            "fixed_rate_bps",
        ):
            finite(getattr(self, name), name)
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须 > 0")
        for name in ("system_loss_db", "noise_figure_db", "acquisition_s", "switching_s"):
            finite(getattr(self, name), name, 0)
        for name in ("tx_gain_dbi", "rx_gain_dbi", "min_snr_db"):
            finite(getattr(self, name), name)
        finite(self.efficiency, "efficiency")
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency 必须在 (0, 1] 内")
        if not self.source:
            raise ValueError("source 不能为空")


@dataclass(frozen=True)
class GeometrySample(Serializable):
    """[start_s,end_s) 内保持不变的几何；end_s 不外推。

    遮挡/掩角/窗口边界应成为区间边界。link_id 变化表示切换终端。
    """

    start_s: float
    end_s: float
    distance_m: float
    visible: bool = True
    link_id: str = "link"
    pointing_loss_db: float = 0.0
    interference_w: float = 0.0

    def __post_init__(self) -> None:
        for name in ("start_s", "end_s", "distance_m", "pointing_loss_db", "interference_w"):
            finite(getattr(self, name), name, 0)
        if self.end_s <= self.start_s or self.distance_m <= 0:
            raise ValueError("区间必须有正时长，距离必须 > 0")
        if not isinstance(self.visible, bool) or not self.link_id:
            raise ValueError("visible 必须为 bool，link_id 不能为空")


@dataclass(frozen=True)
class LinkState(Serializable):
    """接收端链路预算；无干扰时 SNR 等于 SINR。"""

    start_s: float
    end_s: float
    link_id: str
    distance_m: float
    visible: bool
    fspl_db: float
    received_power_dbw: float
    noise_power_dbw: float
    snr_db: float
    rate_bps: float
    propagation_s: float
    reason: str
    interference_w: float = 0.0
    sinr_db: float = 0.0


@dataclass(frozen=True)
class Message(Serializable):
    """单向消息；期限约束接收完成时间，字节采用十进制整数。"""

    message_id: str
    size_bytes: int
    created_s: float = 0.0
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        if (
            not self.message_id
            or isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ValueError("消息必须有 ID 和正整数字节大小")
        finite(self.created_s, "created_s", 0)
        if self.deadline_s is not None:
            finite(self.deadline_s, "deadline_s", self.created_s)


@dataclass(frozen=True)
class CommunicationEvent(Serializable):
    """可回放的消息事件；发送片段有实际占用结束时间。"""

    kind: str
    time_s: float
    message_id: str
    link_id: str
    end_s: float | None = None
    bits: float = 0.0


@dataclass(frozen=True)
class TransferResult(Serializable):
    """仿真截止时刻的结果；只有 delivered 允许 FL 消费此更新。"""

    message_id: str
    status: str
    transmitted_bits: float
    delivered_bytes: int
    tx_finished_s: float | None
    arrival_s: float | None
    observed_until_s: float
    transmit_energy_j: float
    events: tuple[CommunicationEvent, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransferResult:
        """恢复嵌套事件。"""
        return cls(
            **{
                **data,
                "events": tuple(CommunicationEvent.from_dict(event) for event in data["events"]),
            }
        )
