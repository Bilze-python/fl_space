"""接收端带限干扰与带成本的因果换频策略；纯软件仿真。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

from .models import GeometrySample, RFConfig, Serializable, finite


@dataclass(frozen=True)
class InterferenceBurst(Serializable):
    """接收端已接收总功率 W，矩形平坦频谱，在 [start,end) 激活。

    不是发射功率；必须已计入干扰路径损耗、接收方向图和遮挡。
    可分别描述合法同频源和外部干扰源，每条记录是一个源的接收投影。
    """

    source_id: str
    receiver_id: str
    start_s: float
    end_s: float
    frequency_hz: float
    bandwidth_hz: float
    received_power_w: float
    source_kind: str = "jammer"
    provenance: str = "assumption_receiver_power"

    def __post_init__(self):
        for name in ("start_s", "end_s", "frequency_hz", "bandwidth_hz", "received_power_w"):
            finite(getattr(self, name), name, 0)
        if self.end_s <= self.start_s or self.frequency_hz <= 0 or self.bandwidth_hz <= 0:
            raise ValueError("干扰时间区间、频率、带宽必须有效")
        if not self.source_id or not self.receiver_id or not self.provenance:
            raise ValueError("干扰源、接收端和来源不能为空")
        if self.source_kind not in ("jammer", "cochannel"):
            raise ValueError("source_kind 必须为 jammer 或 cochannel")


@dataclass(frozen=True)
class AntiJammingConfig(Serializable):
    """多频段候选、延迟宽带观测与换频开销。"""

    policy: str = "observed"
    frequencies_hz: tuple[float, ...] = (20e9, 20.02e9, 20.04e9)
    initial_channel: int = 0
    observation_delay_s: float = 0.2
    sensing_s: float = 0.01
    sensing_power_w: float = 0.5
    retune_s: float = 0.02
    resync_s: float = 0.02
    retune_power_w: float = 1.0
    seed: int = 42

    def __post_init__(self):
        object.__setattr__(self, "frequencies_hz", tuple(self.frequencies_hz))
        if self.policy not in ("fixed", "random", "observed"):
            raise ValueError("policy 必须为 fixed/random/observed")
        if not self.frequencies_hz or len(set(self.frequencies_hz)) != len(self.frequencies_hz):
            raise ValueError("候选频率必须非空且唯一")
        for frequency in self.frequencies_hz:
            finite(frequency, "frequency_hz")
            if frequency <= 0:
                raise ValueError("频率必须 > 0")
        if (
            isinstance(self.initial_channel, bool)
            or not isinstance(self.initial_channel, int)
            or not 0 <= self.initial_channel < len(self.frequencies_hz)
        ):
            raise ValueError("initial_channel 越界")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed 必须为整数")
        for name in (
            "observation_delay_s",
            "sensing_s",
            "sensing_power_w",
            "retune_s",
            "resync_s",
            "retune_power_w",
        ):
            finite(getattr(self, name), name, 0)


class InterferenceEnvironment:
    """引擎真值与延迟观测分离；策略只读取过去时间的功率向量。"""

    def __init__(self, bursts: tuple[InterferenceBurst, ...] = ()):
        self.bursts = tuple(bursts)
        for index, burst in enumerate(self.bursts):
            if any(
                other.source_id == burst.source_id
                and other.receiver_id == burst.receiver_id
                and max(other.start_s, burst.start_s) < min(other.end_s, burst.end_s)
                for other in self.bursts[:index]
            ):
                raise ValueError("同源同接收端的重叠区间会重复计功率")

    def power_w(
        self, receiver: str, time_s: float, frequency_hz: float, bandwidth_hz: float
    ) -> float:
        """各源频谱重叠部分的线性 W 相加，绝不直接相加 dB。"""
        total = 0.0
        for burst in self.bursts:
            if burst.receiver_id != receiver or not burst.start_s <= time_s < burst.end_s:
                continue
            overlap = max(
                0.0,
                min(frequency_hz + bandwidth_hz / 2, burst.frequency_hz + burst.bandwidth_hz / 2)
                - max(frequency_hz - bandwidth_hz / 2, burst.frequency_hz - burst.bandwidth_hz / 2),
            )
            total += burst.received_power_w * overlap / burst.bandwidth_hz
        return total

    def choose(
        self,
        config: AntiJammingConfig,
        receiver: str,
        time_s: float,
        current: int,
        bandwidth_hz: float,
        key: str,
    ) -> tuple[int, float | None, list[float]]:
        """初始没有历史观测时保持信道；等价最小功率时优先当前信道。"""
        if config.policy == "fixed":
            return config.initial_channel, None, []
        if config.policy == "random":
            digest = hashlib.sha256(json.dumps([config.seed, key]).encode()).digest()
            return int.from_bytes(digest[:8], "big") % len(config.frequencies_hz), None, []
        observed_s = time_s - config.observation_delay_s
        if observed_s < 0:
            return current, None, []
        powers = [
            self.power_w(receiver, observed_s, frequency, bandwidth_hz)
            for frequency in config.frequencies_hz
        ]
        chosen = min(range(len(powers)), key=lambda index: (powers[index], index != current, index))
        return chosen, observed_s, powers

    def apply(
        self,
        samples: list[GeometrySample],
        receiver: str,
        frequency_hz: float,
        bandwidth_hz: float,
        ready_s: float,
    ) -> list[GeometrySample]:
        """在所有干扰起止边界处分割几何区间，不平均掉短脉冲。"""
        result = []
        for sample in samples:
            start = max(sample.start_s, ready_s)
            if start >= sample.end_s:
                continue
            edges = {start, sample.end_s}
            for burst in self.bursts:
                if burst.receiver_id == receiver:
                    edges.update(
                        edge for edge in (burst.start_s, burst.end_s) if start < edge < sample.end_s
                    )
            ordered = sorted(edges)
            for left, right in zip(ordered, ordered[1:]):
                result.append(
                    replace(
                        sample,
                        start_s=left,
                        end_s=right,
                        interference_w=sample.interference_w
                        + self.power_w(receiver, left, frequency_hz, bandwidth_hz),
                    )
                )
        return result


def interference_from_link(
    source_id: str,
    receiver_id: str,
    sample: GeometrySample,
    transmitter: RFConfig,
    source_kind: str = "jammer",
) -> InterferenceBurst:
    """用既有 RF 路径预算投影干扰源；transmitter 的 rx_gain 是该方向接收增益。"""
    from .rf_channel import evaluate_link

    state = evaluate_link(sample, transmitter)
    return InterferenceBurst(
        source_id,
        receiver_id,
        sample.start_s,
        sample.end_s,
        transmitter.frequency_hz,
        transmitter.bandwidth_hz,
        10 ** (state.received_power_dbw / 10) if sample.visible else 0.0,
        source_kind,
        "first_order_rf_link_projection",
    )
