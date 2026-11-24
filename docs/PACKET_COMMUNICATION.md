# 共享通信资源与分组队列：第二阶段

状态：本地已实现 TDMA-RR、DRR、分组队列和停止等待 ARQ。
与第一阶段的独立流体传输并存，由 `PacketNetwork` 显式启用。
第三阶段已接入可选接收端干扰与换频策略，见 [抗干扰说明](ANTI_JAMMING.md)。

## 运行对照实验

```powershell
.\venv\Scripts\python.exe -m fl_space.communication --scenario contention --output output/communication-contention.json
```

实验采用三条逻辑流共享相同 Kepler 星地接触轨迹，隔离资源竞争的影响；不将
相同轨迹冒充三颗真实独立轨道卫星的位置。每流传输 120,000 字节模型更新，
按 1200 字节分包，带 32 字节包头，时隙 0.1 秒。固定 seed 42，外生数据丢包
概率 0.15、ACK 丢失概率 0.05，ACK 超时 0.3 秒，最多尝试 5 次。

本次两种策略各完整交付 360,000 字节，374 次数据发送尝试，120 秒观察范围内
消息 goodput 为 24,000 bit/s。等长包和充足时间下结果相同是合理基线，不能
据此声称 DRR 提升吞吐。字节公平性的单独测试使用 100/300 字节队首包。

JSON 包含每种策略的配置、消息/流/端点、几何轨迹、发送占用区间、包接收、
ACK、超时、重复包、消息交付、队列峰值、有效吞吐和 Jain 公平性。

## 使用接口

```python
from fl_space.communication import (
    FlowMessage, GeometrySample, Message, PacketNetwork, PacketNetworkConfig, RFConfig,
)

messages = [
    FlowMessage("flow-a", "sat-a", "ground", Message("update-a", 12000)),
    FlowMessage("flow-b", "sat-b", "ground", Message("update-b", 12000, deadline_s=30)),
]
geometry = {
    flow: [GeometrySample(0, 30, distance_m=550000)]
    for flow in ("flow-a", "flow-b")
}
result = PacketNetwork(RFConfig(), PacketNetworkConfig(policy="drr")).run(
    messages, geometry, until_s=30,
)
print(result.to_dict())
```

默认时隙 1 秒，足以容纳默认捕获 0.2 秒；实验特意使用捕获 0.002 秒配合
0.1 秒时隙。参数必须允许整包在一次 grant 内发送完成，否则包将持续失败。

## 实现约束

- 唯一共享数据资源：全网一个 RF 数据信道、一个同时发送 grant。每个 grant
  使用完整配置带宽和功率，因此不会给每个竞争流复制完整链路容量。
  这也给出保守的端点并发上界 1；暂不支持多个独立波束或并行频段。
- TDMA-RR 在有可尝试队首的流之间按包轮询。DRR 按载荷字节扣 deficit，
  允许一个访问轮连续服务多个小包。每时隙最多一个包，余下时隙不复用。
  信道不可用仍可能占用 grant；这是不感知链路的基线调度假设。
- 每流 FIFO、停止等待：队首在等 ACK 时，该流后续包阻塞，其他流仍可发送。
  新消息按 `created_s` 入队。同一流端点保持一致，消息 ID 必须全局唯一。
- 容量采用整消息保留：发送端为已入队消息预留全部载荷存储，直到全部 ACK、
  超期或重试耗尽才释放。容量不足拒绝整条新消息，不偷偷截断；包头不计入
  此载荷存储预算，在线路占用中计入。
- 每个 grant 独立捕获，复用 `DynamicCommunication` 的 RF 服务积分。包未在
  grant 内完整发送，或发送过程中发生窗口中断，则本次包失败，下次从头发送。
  片段消耗的功率和信道服务量仍计费。该实现尚未复用跨 grant 的终端锁定状态。
- 每次实际发送后计 ACK 超时；发送失败也等到超时才能重试。完全无发送服务
  的 grant 不消耗重试次数。`max_attempts` 包含首次发送。
- ACK 为独立串行控制信道，显式 ACK 字节大小、速率、队列和反向传播；路径
  暂假定可用。控制信道不争用数据信道，不能解释为真实半双工协议。
- 接收端按消息 ID/包序号去重，重复包再次触发 ACK。完整模型只产生一次
  `delivered`。ACK 丢失可造成“接收已交付，但发送端重试耗尽”，因此输出
  `status` 与 `sender_status` 分开；未来 FL 只能消费真实接收交付事件。
- 数据到达和 ACK 是离散事件；未来事件不提前交付。期限等时的接收优先处理，
  数据在期限后到达则丢弃。发送端在 deadline 停止等待和后续尝试。
- 丢包通过 `(seed, message_id, packet_index, attempt, data/ack)` 的 SHA-256
  稳定键取样，不依赖调度事件调用顺序。它是可复现外生过程，非 SINR/BLER
  校准结果。消息 partial payload、完整消息有效载荷及在线字节分别统计。
- 功率统计仅含数据发送；未计算捕获电路、接收机和 ACK 发射能量。
  Jain 指标基于完整交付的每流字节量，所有流纳入，零交付时为 null。
- `run` 从模拟秒 0 开始一次性运行，完整返回轨迹；当前不支持快照续跑或
  实时推进。分组/超时事件在最终观察时刻之后不输出。

## 模块与下一阶段

`scheduling.py` 提供 `PacketScheduler`；`packet_network.py` 提供配置、流消息、
结果契约和事件引擎；`contention_demo.py` 提供可运行实验。
所有对象通过 `fl_space.communication` 公开导出，通信层不依赖 FL。

验证：

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_packet_network.py tests/test_dynamic_communication.py -q
```

当前已接入外部源接收端干扰预算和固定/随机/观测换频。多波束终端并发、
Slotted ALOHA、网页和 FL 聚合接入尚未实现。
