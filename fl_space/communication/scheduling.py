"""共享单信道的分组级轮询与字节公平 DRR 调度器。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class PacketScheduler:
    """在可发送流中选择队首分组；DRR 按实际载荷字节扣除 deficit。

    暂时等待 ACK 的流不丢失 deficit；空队列通过 reset 显式清零。
    DRR 同一轮可连续发多个小包，避免退化为按包轮询。
    """

    def __init__(self, flow_ids: Sequence[str], policy: str, quantum_bytes: int) -> None:
        if policy not in ("tdma_rr", "drr") or not flow_ids or len(set(flow_ids)) != len(flow_ids):
            raise ValueError("需要唯一流 ID 与 tdma_rr/drr 策略")
        if (
            isinstance(quantum_bytes, bool)
            or not isinstance(quantum_bytes, int)
            or quantum_bytes <= 0
        ):
            raise ValueError("quantum_bytes 必须为正整数")
        self.flows = tuple(flow_ids)
        self.policy = policy
        self.quantum_bytes = quantum_bytes
        self.deficit = dict.fromkeys(self.flows, 0)
        self.cursor = 0
        self.visiting = False

    def reset(self, flow_id: str) -> None:
        """队列真正排空时清除历史信用。"""
        self.deficit[flow_id] = 0
        if self.flows[self.cursor] == flow_id:
            self.visiting = False

    def choose(self, ready_bytes: Mapping[str, int]) -> str | None:
        """选择并扣费；调用方必须提供可尝试发送的队首载荷字节数。"""
        if any(flow not in self.deficit or size <= 0 for flow, size in ready_bytes.items()):
            raise ValueError("未知流或无效队首大小")
        if not ready_bytes:
            return None
        while True:
            flow = self.flows[self.cursor]
            if flow in ready_bytes:
                if self.policy == "tdma_rr":
                    self.cursor = (self.cursor + 1) % len(self.flows)
                    return flow
                if not self.visiting:
                    self.deficit[flow] += self.quantum_bytes
                    self.visiting = True
                if self.deficit[flow] >= ready_bytes[flow]:
                    self.deficit[flow] -= ready_bytes[flow]
                    return flow
            self.cursor = (self.cursor + 1) % len(self.flows)
            self.visiting = False
