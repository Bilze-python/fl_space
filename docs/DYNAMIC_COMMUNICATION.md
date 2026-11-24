# 动态通信层：第一阶段使用说明

状态：本地已实现单消息动态 RF 基线。第二阶段已另增共享信道分组/ARQ 引擎，见
[分组通信说明](PACKET_COMMUNICATION.md)。以下边界针对 `DynamicCommunication` 单消息接口。
后续抗干扰与训练器联动按
[研究方案](DYNAMIC_COMMUNICATION_INTERFERENCE_PLAN_20261111.md) 分阶段接入。

## 架构决策

选择独立的 `fl_space/communication/` 包，而非在 `PhysicsTimeModel` 内继续扩展
固定速率换算。前者能保留链路轨迹、窗口边界和交付事件，代价是需要单独传入
几何区间；后者接入训练更快，但无法可靠表示中断与跨窗口传输。

依赖方向为 `environment/orbit → communication → 将来的 FL 适配器`。
通信层不导入训练器、不加载 PyTorch、不访问网络。

| 文件 | 职责 |
| --- | --- |
| `models.py` | RF 配置、几何区间、链路预算、消息和传输事件的数据契约 |
| `geometry.py` | 开普勒轨道分钟/km 到秒/m 的适配、WGS84 遮挡、几何采样 |
| `rf_channel.py` | Friis 路径损耗、增益/指向损失、kTB 噪声、SNR 与动态速率 |
| `engine.py` | 分段服务积分、捕获/重连/切换、中断恢复、传播与交付期限 |
| `__main__.py` | 同轨迹固定/动态速率的本地对照实验与 JSON 导出 |

## 直接运行

仓库根目录 PowerShell：

```powershell
.\venv\Scripts\python.exe -m fl_space.communication --output output/communication-baseline.json
```

示例为开普勒赤道圆轨道与 WGS84 赤道地面站，1200 秒、5 秒几何步长，传输
20,000,000 字节（20 MB，非 MiB）。同轨迹对照只改变速率模型，均计入捕获、
传播和遮挡。输出包含完整配置、轨道根数、保真度、链路轨迹和消息事件，
可直接作为后续 API/网页回放的数据来源。首次运行实测输出：

| 模式 | 消息状态 | 到达时间（模拟秒） |
| --- | --- | --- |
| fixed | delivered | 16.201866 |
| dynamic | delivered | 88.395178 |

这些数值对应示例工程假设，不代表真实卫星性能或论文收益。默认频率 20 GHz、
带宽 10 MHz、发射功率 1 W、两端增益各 20 dBi、固定基线 10 Mbps。
初次捕获 0.2 秒；重连或改变 `link_id` 再付 0.2 秒捕获与 0.1 秒切换成本。

## Python 接口

```python
from fl_space.communication import (
    DynamicCommunication, GeometrySample, Message, RFConfig,
)

# 每个区间明确结束，缺失区间不会外推；中断后保留流体发送进度。
samples = [
    GeometrySample(0, 10, distance_m=550_000),
    GeometrySample(10, 20, distance_m=650_000, visible=False),
    GeometrySample(20, 40, distance_m=700_000),
]
result = DynamicCommunication(RFConfig()).transfer(
    Message("update-1", size_bytes=5_000_000, deadline_s=40), samples,
)
print(result.to_dict())
```

已有轨道用 `orbit_position_provider(orbit)` 适配，再传入
`sample_geometry(tx_provider, rx_provider, boundaries_s)`。
两个 provider 必须使用相同 epoch 的 ECEF 米制坐标。星地最低仰角/已知接触窗口
可通过 `visibility(time_s)` 提供，与 WGS84 遮挡取交集；原始 `ISLWindow` 的
最小/最大距离不够重建时变速率，须使用轨道位置重新采样。

结果状态：

- `pending`：截至仿真终点仍未发完，不能聚合。
- `in_flight`：已发完但尚未全部到达，不能聚合。
- `delivered`：全部片段在仿真终点和期限之前到达，允许未来 FL 适配器消费。
- `expired`：期限到达且尚未完整交付，不能聚合。

`arrival_s` 是完整发送后计算的预计到达时刻，`in_flight` 时它可能在未来，
不得仅根据该字段非空就触发聚合。`delivered_bytes` 仅在实际交付状态下为消息大小。
每次 `transfer` 是一次完整的独立仿真查询，不是有状态推进 API；不能拼接多次
查询的事件作为连续执行，也不能把独立调用解释为共享资源并发传输。

## 物理与协议边界

- RF 采用自由空间远场损耗与常数理想跟踪天线增益；额外指向损失可逐区间输入。
  尚未模拟姿态控制、天气衰落、残余多普勒或光通信。
- 动态速率是 `min(max_rate, efficiency × B × log2(1+SNR))` 的一阶假设，
  低于显式 SNR 门限停传。它没有经过 MCS/BLER 实测校准，不是保证可达吞吐。
- 固定基线保留几何可见性但忽略 SNR 门限，用于单独比较固定速率假设。
- 带宽 Hz、速率 bit/s；损耗 dB、增益 dBi、功率 dBW、发射功率 W、时间 s、
  距离 m。`noise_temperature_k` 指噪声因子施加前的等效输入温度。
- 几何在区间左端点采样并零阶保持；窗口根未精确求解，可见性边界误差受采样
  步长影响。已知窗口边界应插入 `boundaries_s`，正式实验需缩小步长验证收敛。
- 各发送片段使用本区间距离/c 估算传播时延，全部片段到达后才交付。
  不追踪光行时内接收机运动与遮挡变化；WGS84 判定专用于地球。
- 首版为可靠单消息流体模型。跨窗口的进度保留是显式假设，不是已实现的
  分组缓存、ACK 或重传。当前能耗只统计实际发射功率×发送时长，不含接收、
  基带或捕获能耗。
- 当前无其他发射源，SNR 等于 SINR。真实同频功率累加、感知/避扰、TDMA/DRR/
  ALOHA、多流队列与硬件并发预算尚待后续阶段实现。原训练入口和网页尚未切换到此层。

## 验证与下一步

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_dynamic_communication.py -q
.\venv\Scripts\python.exe -m ruff check fl_space/communication tests/test_dynamic_communication.py
```

测试覆盖距离翻倍损耗增加 6.0206 dB、GEO 传播量级、地球遮挡、无链路、
动态速率积分、窗口中断与重连、捕获跨采样区间、链路切换、未来到达不交付、
先前片段的较长传播、截止期、序列化、无效参数及现有 KeplerOrbit 最小集成。

下一阶段先增加共享资源调度与真正的分组/队列，明确发送端和接收端 ID、信道、
波束、终端预算；在调度出的同时发送集合上计算接收端同频干扰。最后通过训练器
适配器只将实际 `delivered` 的更新送入 FL 聚合，避免固定时长估算与实际交付重复计时。
