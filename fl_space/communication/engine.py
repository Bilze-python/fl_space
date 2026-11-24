"""单消息动态传输：分段常速服务积分和确定性到达事件。"""

from __future__ import annotations

from collections.abc import Sequence

from .models import CommunicationEvent, GeometrySample, Message, RFConfig, TransferResult
from .rf_channel import evaluate_link


class DynamicCommunication:
    """无共享资源的首版单向流体模型，每次 transfer 独立运行。

    支持跨窗口保留发送进度。尚无分组、ACK、重传或多流竞争，因此
    不能将多个独立 transfer 解释为一个共享链路上的并发传输。
    """

    def __init__(self, config: RFConfig | None = None) -> None:
        self.config = config if config is not None else RFConfig()

    def transfer(self, message: Message, samples: Sequence[GeometrySample]) -> TransferResult:
        """在给定区间覆盖范围内模拟，不把预测到达提前当作交付。

        首次连接付 acquisition，重连/换终端付 acquisition+switching。
        缺失区间视为不可用。各片段传播时延按本区间几何冻结，取所有
        片段到达时间的最大值。接收机运动与在途遮挡不再追踪，属于
        一阶近似。过期按接收完成判定。
        """
        if not samples:
            raise ValueError("samples 不能为空")
        if any(right.start_s < left.end_s for left, right in zip(samples, samples[1:])):
            raise ValueError("samples 必须按时间排列且不重叠")
        horizon = samples[-1].end_s
        if message.created_s > horizon:
            raise ValueError("消息创建时间超出仿真范围")
        remaining = float(message.size_bytes * 8)
        events = []
        energy = 0.0
        ready_s = 0.0
        previous_end = None
        previous_link = None
        connected = False
        ever_connected = False
        finished_s = arrival_s = None
        latest_fragment_arrival = 0.0
        final_link = ""
        for sample in samples:
            start = max(sample.start_s, message.created_s)
            end = min(
                sample.end_s, message.deadline_s if message.deadline_s is not None else horizon
            )
            if end <= start:
                continue
            state = evaluate_link(sample, self.config)
            continuous = connected and previous_end == start and previous_link == sample.link_id
            if connected and not continuous:
                events.append(
                    CommunicationEvent(
                        "interrupted", previous_end, message.message_id, previous_link
                    )
                )
                connected = False
            if state.rate_bps <= 0:
                if connected:
                    events.append(
                        CommunicationEvent("interrupted", start, message.message_id, previous_link)
                    )
                connected = False
            else:
                if not continuous:
                    overhead = self.config.acquisition_s
                    if ever_connected:
                        overhead += self.config.switching_s
                    ready_s = start + overhead
                    events.append(
                        CommunicationEvent("connecting", start, message.message_id, sample.link_id)
                    )
                    ever_connected = True
                connected = True
                if start < ready_s:
                    events.append(
                        CommunicationEvent(
                            "acquiring",
                            start,
                            message.message_id,
                            sample.link_id,
                            min(ready_s, end),
                        )
                    )
                tx_start = max(start, ready_s)
                if tx_start < end:
                    bits = min(remaining, (end - tx_start) * state.rate_bps)
                    tx_end = tx_start + bits / state.rate_bps
                    events.append(
                        CommunicationEvent(
                            "transmitting",
                            tx_start,
                            message.message_id,
                            sample.link_id,
                            tx_end,
                            bits,
                        )
                    )
                    energy += (tx_end - tx_start) * self.config.tx_power_w
                    latest_fragment_arrival = max(
                        latest_fragment_arrival, tx_end + state.propagation_s
                    )
                    remaining -= bits
                    if remaining <= 0:
                        finished_s = tx_end
                        arrival_s = latest_fragment_arrival
                        final_link = sample.link_id
                        break
            previous_end, previous_link = end, sample.link_id
        deadline = message.deadline_s
        if (
            arrival_s is not None
            and arrival_s <= horizon
            and (deadline is None or arrival_s <= deadline)
        ):
            status = "delivered"
            events.append(
                CommunicationEvent("delivered", arrival_s, message.message_id, final_link)
            )
        elif deadline is not None and deadline <= horizon:
            status = "expired"
            events.append(CommunicationEvent("expired", deadline, message.message_id, final_link))
        elif finished_s is not None:
            status = "in_flight"
        else:
            status = "pending"
        return TransferResult(
            message.message_id,
            status,
            message.size_bytes * 8 - remaining,
            message.size_bytes if status == "delivered" else 0,
            finished_s,
            arrival_s,
            horizon,
            energy,
            tuple(events),
        )
