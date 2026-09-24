# SpaceFL 架构优化方向

**状态**：v0.1 设计落地
**日期**：2026-09-16
**目标**：按“学习任务”为最高级单元，组织轨道、基础设施、学习任务和调度。

## 设计结论

导师建议可以落成一个清晰的分层模型：

```text
LearningTaskScenario
├── DatasetSpec              # 数据类型、规模、分布、分片
├── ModelSpec                # 模型架构、参数量、模型/更新大小
├── SpaceNodeSpec[]          # 卫星设计 + 六根数轨道
├── InfrastructureSpec       # 地面站 + 星地/星间链路
└── SchedulerSpec            # 接触感知、能量感知、FedLEO 等调度策略
```

新代码入口：

- `fl_space.study`：学习任务顶层场景；
- `fl_space.mission.ClassicalOrbitalElements`：二体轨道六根数；
- `fl_space.mission.CommunicationSystem`：补充频段、带宽、最大距离、终端数量和来源；
- `fl_space.integrations.spacex_api.orbital_elements_from_starlink`：从 SpaceX API Starlink 目录字段提取轨道；
- `fl_space.integrations.spacex_api.public_starlink_optical_link_profile`：公开资料驱动的 Starlink 光链路候选配置。

## 四个关键要素

### 1. 轨道

轨道决定卫星相对速度、接触窗口、星间可见性和训练轮同步机会。新增的
`ClassicalOrbitalElements` 使用二体六根数：

- `semi_major_axis_km`
- `eccentricity`
- `inclination_deg`
- `raan_deg`
- `arg_perigee_deg`
- `anomaly_deg`

当前旧接口 `OrbitalTarget` 仍保留，用于发射任务和一阶圆轨道估算。后续传播器
可以逐步优先接收六根数，再降级到 `OrbitalTarget`。

### 2. 基础设施

基础设施分为地面站和通信链路：

- 地面站：经纬度、最小仰角、角色；
- 星地链路：频段、中心频率、信道带宽、数据速率、最大距离；
- 星间链路：视线、指向、并发链路、最大距离、峰值/持续速率；
- 光链路：额外标注天气敏感性、捕获时间和来源。

原则：链路参数必须带来源。官方未公开的参数使用 `assumption`，不得写成 SpaceX
官方规格。

### 3. 学习任务

学习任务是最高级配置对象，而不是训练脚本里的零散参数。一个任务至少包含：

- 数据集：类型、分布、样本量、大小、分片；
- 模型：架构、框架、参数量、模型大小、每轮更新大小；
- 参与节点：每颗卫星的物理设计和轨道；
- 基础设施：地面站和链路；
- 调度策略：轮时长、参与比例、最大客户端数量。

这样用户以后自定义功能时，只需要扩展某一层对象，不必直接改动训练循环。

### 4. 调度

调度决定学习效率和效果。建议逐步支持：

- `contact_aware`：优先选择当前或即将接触地面站的卫星；
- `energy_aware`：考虑功率余量、电池和蚀期；
- `fedleo`：对齐既有 FedLEO 调度逻辑；
- `round_robin`：作为可解释基线；
- `custom`：用户插件扩展。

调度层读取轨道、链路和学习任务参数，但不改写物理公式。

## 后续施工顺序

1. 将 `LearningTaskScenario` 接入 YAML/Pydantic schema；
2. 增加从 `LearningTaskScenario` 到现有 `OrbitSimulator` 和 FL 配置的适配器；
3. 将链路可见性、带宽和模型更新大小接入每轮训练耗时；
4. 在 web 控制台用“学习任务 -> 节点/轨道 -> 基础设施 -> 调度”四级视图展示；
5. 把发射部署任务作为可选前置阶段，输出卫星节点与初始轨道；
6. 将经营模式、合约和资源系统做成独立插件，只调用研究层服务。

## 边界

本框架是研究和教学用，不是飞行认证软件。二体、Kepler、SGP4 和更高保真结果
必须带保真度标签，不能混在同一个实验结论里。
