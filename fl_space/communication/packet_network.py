"""单共享 RF 数据信道、独立串行 ACK 信道的离散事件分组基线。"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import heapq
import json
from typing import Any

from .engine import DynamicCommunication
from .interference import AntiJammingConfig, InterferenceEnvironment
from .models import GeometrySample, Message, RFConfig, Serializable, finite
from .scheduling import PacketScheduler


@dataclass(frozen=True)
class PacketNetworkConfig(Serializable):
    """协议假设；单次 grant 最多一个完整包；未发完的包在下次整体重传。"""

    policy: str = "tdma_rr"
    slot_s: float = 1.0
    packet_bytes: int = 1200
    header_bytes: int = 32
    queue_bytes_per_flow: int = 1_000_000
    quantum_bytes: int = 1200
    max_attempts: int = 4
    ack_timeout_s: float = 1.0
    ack_bytes: int = 24
    ack_rate_bps: float = 1e6
    packet_loss_probability: float = 0.0
    ack_loss_probability: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.policy not in ("tdma_rr", "drr"):
            raise ValueError("policy 必须为 tdma_rr 或 drr")
        for name in ("slot_s", "ack_timeout_s", "ack_rate_bps"):
            finite(getattr(self, name), name)
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须 > 0")
        for name in (
            "packet_bytes",
            "queue_bytes_per_flow",
            "quantum_bytes",
            "max_attempts",
            "ack_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if (
            isinstance(self.header_bytes, bool)
            or not isinstance(self.header_bytes, int)
            or self.header_bytes < 0
        ):
            raise ValueError("header_bytes 必须为非负整数")
        for name in ("packet_loss_probability", "ack_loss_probability"):
            finite(getattr(self, name), name, 0)
            if getattr(self, name) > 1:
                raise ValueError(f"{name} 必须 <= 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed 必须为整数")


@dataclass(frozen=True)
class FlowMessage(Serializable):
    """消息、发送端与接收端；所有流共享唯一 RF 数据资源。"""

    flow_id: str
    sender_id: str
    receiver_id: str
    message: Message

    def __post_init__(self) -> None:
        if (
            not self.flow_id
            or not self.sender_id
            or not self.receiver_id
            or self.sender_id == self.receiver_id
        ):
            raise ValueError("流和端点 ID 必须有效且收发端不同")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowMessage:
        """恢复消息对象。"""
        return cls(**{**data, "message": Message.from_dict(data["message"])})


@dataclass
class _Packet:
    owner: FlowMessage
    index: int
    size_bytes: int
    attempts: int = 0
    waiting: bool = False
    received: bool = False
    acked: bool = False


@dataclass
class _MessageState:
    owner: FlowMessage
    packet_count: int
    received_count: int = 0
    acked_count: int = 0
    received_payload_bytes: int = 0
    arrival_s: float | None = None
    sender_status: str = "pending"
    admitted: bool = False
    released: bool = False


@dataclass(frozen=True)
class PacketNetworkResult(Serializable):
    """JSON 可回放结果；接收交付和发送确认分别报告。"""

    messages: dict[str, dict[str, Any]]
    events: list[dict[str, Any]]
    metrics: dict[str, Any]


class PacketNetwork:
    """停止等待 ARQ + TDMA-RR/DRR；仅完整接收消息可作为未来 FL 更新。

    数据信道全网每时隙最多一个 grant，故带宽、波束与端点发送并发均为 1。
    ACK 使用独立、串行控制信道，按 ack_rate_bps 排队并计入反向传播；
    ACK 路径暂假定可用，不冒充同一 RF 半双工链路。可选接收端干扰
    影响 SINR/服务速率，独立外生丢包概率可作为额外损失；尚无 BLER 校准。
    """

    def __init__(
        self,
        rf: RFConfig | None = None,
        config: PacketNetworkConfig | None = None,
        anti_jamming: AntiJammingConfig | None = None,
        interference: InterferenceEnvironment | None = None,
    ):
        self.rf = rf if rf is not None else RFConfig()
        self.config = config if config is not None else PacketNetworkConfig()
        self.anti_jamming = anti_jamming
        self.interference = interference if interference is not None else InterferenceEnvironment()
        if (anti_jamming is not None or self.interference.bursts) and self.rf.mode != "dynamic":
            raise ValueError("抗干扰实验必须使用 dynamic RF 速率，不能用忽略 SINR 的 fixed 基线")
        if self.interference.bursts and anti_jamming is None:
            raise ValueError("有干扰时必须显式指定抗干扰配置，可选择 fixed 策略")

    def _lost(self, packet: _Packet, attempt: int, kind: str, probability: float) -> bool:
        key = json.dumps(
            [self.config.seed, packet.owner.message.message_id, packet.index, attempt, kind],
            separators=(",", ":"),
        )
        draw = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
        return draw < probability

    def run(
        self,
        messages: Sequence[FlowMessage],
        geometry: Mapping[str, Sequence[GeometrySample]],
        until_s: float,
    ) -> PacketNetworkResult:
        """从 0 开始运行一次；输入轨迹不外推，消息按 created_s 入队。

        事件优先级：入队、数据到达、ACK、期限、超时、时隙选择。
        同时到达期限的完整消息有效；同一时间 ACK 先于超时。
        """
        finite(until_s, "until_s", 0)
        if until_s <= 0 or not messages:
            raise ValueError("需要消息及正仿真时长")
        ids = [item.message.message_id for item in messages]
        if len(set(ids)) != len(ids):
            raise ValueError("message_id 必须全局唯一")
        flows = sorted({item.flow_id for item in messages})
        endpoints = {}
        for item in messages:
            pair = (item.sender_id, item.receiver_id)
            if endpoints.setdefault(item.flow_id, pair) != pair:
                raise ValueError("同一流的端点必须一致")
        for flow in flows:
            if flow not in geometry:
                raise ValueError(f"缺少轨迹: {flow}")
            if any(
                right.start_s < left.end_s
                for left, right in zip(geometry[flow], geometry[flow][1:])
            ):
                raise ValueError("轨迹必须有序且不重叠")
        config = self.config
        scheduler = PacketScheduler(flows, config.policy, config.quantum_bytes)
        queues = {flow: deque() for flow in flows}
        occupied = dict.fromkeys(flows, 0)
        peak = dict.fromkeys(flows, 0)
        states = {
            item.message.message_id: _MessageState(
                item, (item.message.size_bytes + config.packet_bytes - 1) // config.packet_bytes
            )
            for item in messages
        }
        events = []
        agenda = []
        serial = 0
        ack_free_s = 0.0
        wire_bits = energy_j = 0.0
        attempts = 0
        channel = self.anti_jamming.initial_channel if self.anti_jamming is not None else 0
        sensing_energy_j = retune_energy_j = 0.0
        switches = 0

        def emit(kind, time, owner, **details):
            events.append(
                {
                    "kind": kind,
                    "time_s": time,
                    "flow_id": owner.flow_id,
                    "message_id": owner.message.message_id,
                    **details,
                }
            )

        def schedule(time, priority, kind, payload):
            nonlocal serial
            serial += 1
            heapq.heappush(agenda, (time, priority, serial, kind, payload))

        def release(state):
            if state.admitted and not state.released:
                occupied[state.owner.flow_id] -= state.owner.message.size_bytes
                state.released = True

        for item in messages:
            schedule(item.message.created_s, 0, "enqueue", item)
            if item.message.deadline_s is not None:
                schedule(item.message.deadline_s, 3, "expire", item)
        schedule(0.0, 5, "slot", 0)
        while agenda and agenda[0][0] <= until_s:
            time, _, _, kind, payload = heapq.heappop(agenda)
            if kind == "enqueue":
                owner = payload
                state = states[owner.message.message_id]
                if occupied[owner.flow_id] + owner.message.size_bytes > config.queue_bytes_per_flow:
                    state.sender_status = "overflow"
                    emit("queue_overflow", time, owner)
                    continue
                occupied[owner.flow_id] += owner.message.size_bytes
                peak[owner.flow_id] = max(peak[owner.flow_id], occupied[owner.flow_id])
                state.admitted = True
                for index in range(state.packet_count):
                    size = min(
                        config.packet_bytes, owner.message.size_bytes - index * config.packet_bytes
                    )
                    queues[owner.flow_id].append(_Packet(owner, index, size))
                emit("enqueued", time, owner, queue_bytes=occupied[owner.flow_id])
            elif kind == "expire":
                state = states[payload.message.message_id]
                if state.sender_status == "pending":
                    state.sender_status = "expired"
                    release(state)
                    emit("expired", time, payload)
            elif kind == "data":
                packet, attempt, propagation = payload
                state = states[packet.owner.message.message_id]
                deadline = packet.owner.message.deadline_s
                if deadline is not None and time > deadline:
                    continue
                if not packet.received:
                    packet.received = True
                    state.received_count += 1
                    state.received_payload_bytes += packet.size_bytes
                    emit("packet_received", time, packet.owner, packet=packet.index)
                    if state.received_count == state.packet_count:
                        state.arrival_s = time
                        emit("delivered", time, packet.owner, bytes=packet.owner.message.size_bytes)
                else:
                    emit("duplicate_received", time, packet.owner, packet=packet.index)
                ack_start = max(time, ack_free_s)
                ack_free_s = ack_start + config.ack_bytes * 8 / config.ack_rate_bps
                emit(
                    "ack_transmitting",
                    ack_start,
                    packet.owner,
                    end_s=ack_free_s,
                    packet=packet.index,
                    attempt=attempt,
                )
                if not self._lost(packet, attempt, "ack", config.ack_loss_probability):
                    schedule(ack_free_s + propagation, 2, "ack", packet)
            elif kind == "ack":
                packet = payload
                state = states[packet.owner.message.message_id]
                if not packet.acked:
                    packet.acked = True
                    state.acked_count += 1
                    emit("ack_received", time, packet.owner, packet=packet.index)
                    if state.acked_count == state.packet_count and state.sender_status == "pending":
                        state.sender_status = "confirmed"
                        release(state)
            elif kind == "timeout":
                packet, attempt = payload
                state = states[packet.owner.message.message_id]
                if packet.acked or packet.attempts != attempt or state.sender_status != "pending":
                    continue
                packet.waiting = False
                emit("ack_timeout", time, packet.owner, packet=packet.index, attempt=attempt)
                if attempt >= config.max_attempts:
                    state.sender_status = "retry_exhausted"
                    release(state)
                    emit("retry_exhausted", time, packet.owner)
            else:
                slot_index = payload
                end = min((slot_index + 1) * config.slot_s, until_s)
                if end <= time:
                    continue
                ready = {}
                for flow, queue in queues.items():
                    while queue and (
                        queue[0].acked
                        or states[queue[0].owner.message.message_id].sender_status != "pending"
                    ):
                        queue.popleft()
                    if not queue:
                        scheduler.reset(flow)
                    elif not queue[0].waiting:
                        ready[flow] = queue[0].size_bytes
                flow = scheduler.choose(ready)
                if flow is not None:
                    packet = queues[flow][0]
                    owner = packet.owner
                    current_rf = self.rf
                    grant_end = min(
                        end,
                        owner.message.deadline_s if owner.message.deadline_s is not None else end,
                    )
                    intervals = [
                        replace(
                            sample,
                            start_s=max(time, sample.start_s),
                            end_s=min(grant_end, sample.end_s),
                        )
                        for sample in geometry[flow]
                        if sample.end_s > time and sample.start_s < grant_end
                    ]
                    if intervals and self.anti_jamming is not None:
                        anti = self.anti_jamming
                        selected, observed_s, observed_powers = self.interference.choose(
                            anti,
                            owner.receiver_id,
                            time,
                            channel,
                            self.rf.bandwidth_hz,
                            f"{flow}:{slot_index}",
                        )
                        changed = selected != channel
                        sensing = anti.sensing_s if anti.policy == "observed" else 0.0
                        retune = anti.retune_s + anti.resync_s if changed else 0.0
                        available = intervals[-1].end_s - time
                        sense_used = min(sensing, available)
                        retune_used = min(retune, max(0.0, available - sense_used))
                        sensing_energy_j += sense_used * anti.sensing_power_w
                        retune_energy_j += retune_used * anti.retune_power_w
                        ready_s = time + sensing + retune
                        emit(
                            "channel_decision",
                            time,
                            owner,
                            channel=selected,
                            frequency_hz=anti.frequencies_hz[selected],
                            observed_s=observed_s,
                            observed_interference_w=observed_powers,
                            ready_s=ready_s,
                            sensing_energy_j=sense_used * anti.sensing_power_w,
                            retune_energy_j=retune_used * anti.retune_power_w,
                            switch_completed=changed and ready_s <= intervals[-1].end_s,
                        )
                        if changed and ready_s <= intervals[-1].end_s:
                            switches += 1
                            channel = selected
                        current_rf = replace(self.rf, frequency_hz=anti.frequencies_hz[selected])
                        intervals = self.interference.apply(
                            intervals,
                            owner.receiver_id,
                            current_rf.frequency_hz,
                            current_rf.bandwidth_hz,
                            ready_s,
                        )
                        from .rf_channel import evaluate_link

                        for interval in intervals:
                            budget = evaluate_link(interval, current_rf)
                            emit(
                                "link_budget",
                                interval.start_s,
                                owner,
                                end_s=interval.end_s,
                                channel=selected,
                                interference_w=budget.interference_w,
                                snr_db=budget.snr_db,
                                sinr_db=budget.sinr_db,
                                rate_bps=budget.rate_bps,
                                reason=budget.reason,
                            )
                    if intervals:
                        deadline = owner.message.deadline_s
                        wire = Message(
                            owner.message.message_id,
                            packet.size_bytes + config.header_bytes,
                            time,
                            deadline,
                        )
                        result = DynamicCommunication(current_rf).transfer(wire, intervals)
                        if result.transmitted_bits > 0:
                            packet.attempts += 1
                            attempts += 1
                            packet.waiting = True
                            wire_bits += result.transmitted_bits
                            energy_j += result.transmit_energy_j
                            emit(
                                "packet_attempt",
                                time,
                                owner,
                                packet=packet.index,
                                attempt=packet.attempts,
                                grant_end_s=end,
                                transmitted_bits=result.transmitted_bits,
                            )
                            for fragment in result.events:
                                if fragment.kind in ("transmitting", "acquiring"):
                                    emit(
                                        fragment.kind,
                                        fragment.time_s,
                                        owner,
                                        end_s=fragment.end_s,
                                        bits=fragment.bits,
                                        packet=packet.index,
                                        attempt=packet.attempts,
                                    )
                            finish = result.tx_finished_s
                            # 完整包需要连续发送；流体层跨窗口保留的进度不等于包解码。
                            if any(event.kind == "interrupted" for event in result.events):
                                finish = None
                            if finish is not None and not self._lost(
                                packet, packet.attempts, "data", config.packet_loss_probability
                            ):
                                propagation = result.arrival_s - finish
                                schedule(
                                    result.arrival_s,
                                    1,
                                    "data",
                                    (packet, packet.attempts, propagation),
                                )
                            else:
                                emit(
                                    "packet_lost" if finish is not None else "packet_incomplete",
                                    finish if finish is not None else end,
                                    owner,
                                    packet=packet.index,
                                )
                            schedule(
                                (finish if finish is not None else end) + config.ack_timeout_s,
                                4,
                                "timeout",
                                (packet, packet.attempts),
                            )
                if end < until_s:
                    schedule((slot_index + 1) * config.slot_s, 5, "slot", slot_index + 1)

        output = {}
        per_flow_bytes = dict.fromkeys(flows, 0)
        for identity, state in states.items():
            message = state.owner.message
            delivered = state.arrival_s is not None
            status = "delivered" if delivered else state.sender_status
            if not state.admitted and message.created_s > until_s:
                status = "not_created"
            output[identity] = {
                "flow_id": state.owner.flow_id,
                "status": status,
                "sender_status": state.sender_status,
                "arrival_s": state.arrival_s,
                "delivered_bytes": message.size_bytes if delivered else 0,
                "received_packets": state.received_count,
                "received_payload_bytes": state.received_payload_bytes,
                "acked_packets": state.acked_count,
                "packet_count": state.packet_count,
            }
            per_flow_bytes[state.owner.flow_id] += output[identity]["delivered_bytes"]
        total = sum(per_flow_bytes.values())
        squares = sum(value**2 for value in per_flow_bytes.values())
        # 未来 ACK 排队记录不属于本次观测；跨截止点占用区间裁到终点。
        events = [event for event in events if event["time_s"] <= until_s]
        for event in events:
            if "end_s" in event:
                event["end_s"] = min(event["end_s"], until_s)
        events.sort(key=lambda event: event["time_s"])
        return PacketNetworkResult(
            output,
            events,
            {
                "observed_until_s": until_s,
                "goodput_bps": total * 8 / until_s,
                "unique_delivered_bytes": total,
                "transmitted_wire_bits": wire_bits,
                "packet_attempts": attempts,
                "transmit_energy_j": energy_j,
                "sensing_energy_j": sensing_energy_j,
                "retune_resync_energy_j": retune_energy_j,
                "data_and_defense_energy_j": energy_j + sensing_energy_j + retune_energy_j,
                "channel_switches": switches,
                "per_flow_delivered_bytes": per_flow_bytes,
                "queue_peak_bytes": peak,
                "queue_remaining_bytes": occupied,
                "jain_fairness": total**2 / (len(flows) * squares) if squares else None,
                "data_channels": 1,
                "data_simultaneous_transmitters_max": 1,
                "ack_channel": "independent_serial_assumed_available",
            },
        )
