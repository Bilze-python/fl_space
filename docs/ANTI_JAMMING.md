# 抗干扰通信基线：第三阶段

已实现接收端同频干扰预算，并接入 `PacketNetwork` 分组队列和 ARQ。
研究背景见 [文献方案](DYNAMIC_COMMUNICATION_INTERFERENCE_PLAN_20261111.md)。
本实现工程参数为 assumption，不把论文性能百分比写入仿真结果。

## 本地运行

```powershell
.\venv\Scripts\python.exe -m fl_space.communication --scenario anti-jamming --output output/communication-antijamming.json
```

同一条 Kepler 星地轨迹传输 48,000 字节；候选中心频率 20、20.02、20.04 GHz，
每个候选带宽 10 MHz。只有一个可调谐的数据终端，全网每时隙最多一个数据
grant，增加候选信道不增加并发容量。窄带源在 0–100 秒覆盖第一个信道，
接收功率为 1e-8 W。本对照数据与 ACK 外生丢包概率均为 0。

| 策略 | 完整交付字节 | 完整消息到达时间 | 完成换频次数 |
| --- | --- | --- | --- |
| fixed 固定信道 | 0 | 未交付，100 秒超期 | 0 |
| random 随机换频 | 48,000 | 57.050425 秒 | 41 |
| observed 延迟观测避扰 | 48,000 | 40.019285 秒 | 1 |

这是固定窄带压力场景，不能外推为真实部署收益或所有场景下的策略排序。
全频段同时受强干扰时，三种策略都可能无法交付。

## 物理计算与输入

`InterferenceBurst` 表达一个源在指定接收端的已接收功率：

- 包含源 ID、接收端 ID、起止秒、中心频率 Hz、带宽 Hz、接收总功率 W、
  来源与 jammer/cochannel 类型。类型不改变功率公式。
- 接收功率已包含路径损耗、接收方向增益、遮挡与指向损失，不能将发射瓦数
  直接填入 received_power_w。`interference_from_link` 可通过既有 RF 预算，
  从干扰路径 GeometrySample 和 RFConfig 投影接收功率；rx_gain_dbi 必须是
  朝向干扰源的接收增益，不能默认复用期望信号的主瓣增益。
- 平坦矩形功率谱假设下，单源带内功率为接收总功率乘以重叠带宽/源带宽。
  按接收端筛选后，各源功率在线性 W 单位相加。
- `SINR = S / (k T B × noise_factor + Σ I)`，SNR 另行保留。动态速率使用
  效率折减 Shannon 上界与硬速率上限，SINR 低于门限则停传。
- 干扰开始/结束处切分几何区间，不跨整个时隙平均短脉冲。扫频、突发干扰
  可用相邻 Burst 表达；同源同接收端重叠时间段被拒绝，防止重复计功率。
  相位相干、接收机非线性和经校准的 MCS/BLER 尚未建模。

## 策略、因果性与成本

`AntiJammingConfig` 提供三种策略：

- fixed：保持 initial_channel，不感知也不主动换频。
- random：按 seed、流 ID、时隙索引的稳定 SHA-256 键选信道，允许选回当前
  信道；不读取干扰真值，也不按事件调用次数消耗随机序列。
- observed：只读取决策时间减 observation_delay_s 的接收功率向量，选择历史
  带内干扰最低信道。同分优先当前信道，无历史观测时保持当前信道。扫描
  读数为无噪声瞬时测量，不读取未来干扰。

observed 每次 grant 占 sensing_s；实际改变信道再占 retune_s + resync_s，
随后还需链路 acquisition_s。以上从 grant 可发送时间扣除，并按功率×实际
秒数计感知和换频/同步能量；期限或观察终点之后不计费。grant 太短时无法
完成换频，不更新锁定信道，下次重新尝试，不跨 grant 恢复同步进度。

默认观测延迟 0.2 秒、感知 0.01 秒、换频与重同步各 0.02 秒。演示将观测延迟
设为 1 秒。`channel_decision` 记录观测时间、历史功率、信道、开销与能量；
`link_budget` 记录实际区间 SINR 与功率。后者是诊断真值，不是策略输入。

收发端暂假定能共享信道决策并在 resync_s 后协调成功；宽带监测接收机和
独立 ACK 控制信道为显式假设。ACK 暂不受外部 RF 干扰；没有控制报文丢失、
反应式对手、功控或绕行。物理速率暂随 SINR 即时变化，真实反馈/调制切换
延迟需另行建模。

## Python 接入

```python
from fl_space.communication import (
    AntiJammingConfig, InterferenceBurst, InterferenceEnvironment,
    PacketNetwork, PacketNetworkConfig, RFConfig,
)

environment = InterferenceEnvironment((
    InterferenceBurst("jammer", "ground", 0, 60, 20e9, 10e6, 1e-8),
))
network = PacketNetwork(
    RFConfig(), PacketNetworkConfig(),
    anti_jamming=AntiJammingConfig(policy="observed"),
    interference=environment,
)
# result = network.run(messages, geometry_by_flow, until_s=100)
```

必须使用 RFConfig(mode="dynamic")；原 fixed 速率消融忽略信噪比，不能用于
抗干扰效果对照。这里 fixed 策略指固定信道，仍使用动态 SINR 物理预算。

输出保留全部配置、种子、源功率时间表、轨道与事件，并增加 sensing_energy_j、
retune_resync_energy_j、data_and_defense_energy_j、channel_switches。
能量只包括数据发射及上述防御开销，不含 ACK、接收电路或基带计算，不能
当作整星总能耗。

## 验证与边界

测试覆盖多源功率累加、频谱重叠、接收端隔离、遮挡路径投影、增加干扰不能
提高 SINR、脉冲边界、延迟观测因果性、随机可复现、实际换频成本与期限能量
上限、宽带干扰无逃逸、无干扰消融及 CLI 输出。

当前自主流仍串行发送，外部合法网络的同频影响可以 cochannel Burst 输入；
不能声称实现多个自主竞争网络。ALOHA、多波束并发、网页与 FL 接入仍待后续。
