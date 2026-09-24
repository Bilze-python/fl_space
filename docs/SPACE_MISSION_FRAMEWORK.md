# SpaceFL 航天任务扩展框架

**状态**：领域模型 v0.3（学习任务顶层、六根数、链路参数、卫星分系统、火箭分级）
**定位**：研究、教学和开放式模拟，不是飞行认证软件
**保真度标签**：`first_order`（一阶/低阶模型）

## 用户需求与仓库文档的边界

本次需求来自用户对未来方向的描述：

- 从地面发射火箭并模拟卫星部署；
- 支持玩家/研究者自定义卫星外观和物理参数；
- 形成类似开放经营模拟的可扩展任务框架；
- 为开源用户提供基础研究功能、扩展入口和工程规范。

仓库中的 `docs/`、`CODING_STANDARDS.md`、`WORKLOG.md` 和
`.codebuddy/skills/` 是现状说明、历史记录或协作参考，不自动等同于新的产品需求。
其中已有的 SpaceFL 联邦学习能力保持不变；本扩展层只在其上增加任务领域对象。

## 新增领域层

```text
fl_space/
├── study/
│   ├── models.py       # 学习任务：数据、模型、卫星轨道、基础设施、调度
│   └── __init__.py     # 学习任务公开导出
├── mission/
│   ├── models.py       # 火箭级、运载火箭、卫星设计、目标轨道、发射窗口、部署计划、任务场景
│   ├── subsystems.py   # 电源、通信、姿态控制、推进分系统
│   ├── physics.py      # SI 单位的一阶轨道/火箭/大气/链路估算
│   ├── budget.py       # 唯一的质量与速度预算口径
│   ├── constraints.py  # 可注册的任务约束与严重程度模型
│   ├── simulation.py   # 倒计时→上升→入轨→部署→完成/失败
│   └── registry.py     # 自定义卫星设计注册表
└── integrations/
    └── spacex_api.py   # SpaceX API 只读目录适配器
```

依赖方向为：

```text
physics  ←  subsystems        ←  models  ←  registry
physics  ←  budget            ←  constraints  ←  models
                                   ↓
                          models  →  simulation
                                   ↓
                    现有 orbit / simulator / FL / web
```

`study` 层是新的顶层组合入口，负责把学习任务、轨道、基础设施和调度组织在
一起；`mission` 层继续负责发射、部署和卫星工程对象。任务层可以调用现有
`environment` 和 `orbit`，但现有 FL 和轨道核心不需要依赖
SpaceX API。这样用户可以离线构造自己的火箭和卫星，不被外部网络服务锁定。
`budget` / `constraints` 只在类型检查时引用 `models`（`TYPE_CHECKING`），
运行时没有循环导入。

## 质量口径（全层一致）

```text
卫星总质量 = 平台干质量 + 分系统结构质量 + 卫星推进剂(含推进系统推进剂)
火箭起飞质量 = 各级湿质量 + 整流罩 + 有效载荷
```

`SatelliteDesign.dry_mass_kg` 只描述平台结构。电源、通信、姿态控制和推进
分系统通过各自的 `mass_kg` 暴露结构质量，推进剂单独计入总质量。因此
`total_mass_kg`、`dry_total_mass_kg` 和 `subsystem_mass_kg` 三个属性可以
互相校验，不会出现"推进剂被算两次"或"分系统质量被静默忽略"。

推进剂不允许同时在 `propellant_mass_kg` 和 `propulsion_system` 中声明，
构造时会直接报错。

## 当前可用能力

```python
from fl_space.mission import (
    DeploymentPlan,
    DeploymentStep,
    LaunchSite,
    LaunchVehicle,
    LaunchWindow,
    MissionScenario,
    MissionSimulator,
    OrbitalTarget,
    PowerSystem,
    PropulsionSystem,
    RocketStage,
    SatelliteDesign,
    VisualAppearance,
)

satellite = SatelliteDesign(
    name="my-satellite",
    dry_mass_kg=80,
    body_dimensions_m=(1.2, 0.8, 0.6),
    cross_section_area_m2=0.8,
    power_system=PowerSystem(solar_array_power_w=420, average_load_w=180, mass_kg=18),
    propulsion_system=PropulsionSystem(
        propellant_type="hydrazine",
        propellant_mass_kg=14,
        dry_mass_kg=7,
        specific_impulse_s=230,
    ),
    visual=VisualAppearance(primitive="custom_mesh", asset_uri="assets/my-satellite.glb"),
)

scenario = MissionScenario(
    launch_vehicle=LaunchVehicle(
        stages=[
            RocketStage(
                name="booster",
                dry_mass_kg=25_000,
                propellant_mass_kg=180_000,
                specific_impulse_s=300,
            ),
            RocketStage(
                name="upper", dry_mass_kg=4_000, propellant_mass_kg=35_000, specific_impulse_s=350
            ),
        ],
        payload_capacity_kg=2_000,
        fairing_inner_diameter_m=4.6,
        fairing_inner_height_m=8.0,
    ),
    payloads=[satellite],
    target_orbit=OrbitalTarget(altitude_km=550, inclination_deg=53),
    launch_window=LaunchWindow(
        open_utc="2026-03-01T02:00:00+00:00",
        close_utc="2026-03-01T04:00:00+00:00",
    ),
    launch_epoch_utc="2026-03-01T02:30:00+00:00",
    deployment_plan=DeploymentPlan(
        steps=[DeploymentStep(satellite_index=1, delay_s=90, separation_delta_v_mps=0.25)]
    ),
)

result = MissionSimulator().simulate(scenario.with_default_constraints())
print(result.phase.value, result.delta_v_margin_mps)
for constraint in result.constraint_results:
    print(constraint.kind, constraint.satisfied, constraint.message)
```

成功结果中的 `deployed_satellites[*].orbit` 是现有 `KeplerOrbit`，可以继续用于
可见性、接触矩阵、ISL 和 SpaceFL 训练时序模拟。

## 领域模型要点

### 学习任务（`fl_space.study.LearningTaskScenario`）

导师建议已落成新的顶层模型：

| 关注点 | 对象 |
| --- | --- |
| 数据 | `DatasetSpec`（类型、分布、样本量、大小、分片） |
| 模型 | `ModelSpec`（任务类型、框架、参数量、模型/更新大小） |
| 卫星/轨道 | `SpaceNodeSpec`（`SatelliteDesign` + `ClassicalOrbitalElements`） |
| 基础设施 | `InfrastructureSpec`（地面站 + 星地/星间链路） |
| 调度 | `SchedulerSpec`（接触感知、能量感知、FedLEO、自定义策略） |

学习任务是研究场景的最高级单元。发射部署任务可以作为前置阶段，生成参与学习
的卫星节点和初始轨道；经营规则和 UI 模式可以读取学习任务对象，但不改写物理层。

### 卫星（`SatelliteDesign`）

| 关注点 | 字段/分系统 |
| --- | --- |
| 质量 | `dry_mass_kg`、`propellant_mass_kg`、`subsystem_mass_kg`、`total_mass_kg` |
| 尺寸 | 物理包络 `body_dimensions_m`、发射收拢 `stowed_dimensions_m` |
| 功率 | `PowerSystem`（太阳翼、电池、平均/峰值负载、蚀期占比） |
| 通信 | `CommunicationSystem`（频段、频率、增益、上下行速率、可选星间链路） |
| 姿态控制 | `AttitudeControlSystem`（模式、指向精度、机动角速度、角动量容量） |
| 推进剂 | `PropulsionSystem`（推进剂类型与质量、推力、比冲、最小冲量） |
| 外观 | `VisualAppearance`（图元、尺寸、资产、颜色） |

`body_dimensions_m`（物理）与 `VisualAppearance.dimensions_m`（渲染）是
两套独立参数：更换渲染模型不会隐式改变质量、阻力和功率，反之亦然。

`CommunicationSystem` 还支持 `channel_bandwidth_mhz`、`terminal_count`、
`max_range_km`、`acquisition_time_s`、`pointing_accuracy_deg` 和 `source`，
用于星地/星间链路初评。无法从公开资料核实的工程参数必须写进
`SatelliteDesign.assumptions` 或链路 `source`，
不要把宣传峰值当作持续可用能力。

### 轨道（`ClassicalOrbitalElements` / `OrbitalTarget`）

`ClassicalOrbitalElements` 使用二体轨道六根数描述航天器轨道：

- 半长轴；
- 偏心率；
- 倾角；
- 升交点赤经；
- 近地点幅角；
- 真近点角或平近点角。

`OrbitalTarget` 作为发射任务的一阶目标轨道继续保留。新研究场景优先使用六根数，
需要兼容旧流程时调用 `to_orbital_target()`。

### 火箭（`LaunchVehicle` / `RocketStage`）

- `stages` 从一级到末级，`stage_count` 为级数；
- 每级声明 `dry_mass_kg`（结构质量）、`propellant_mass_kg`、
  `specific_impulse_s`（真空比冲）、`thrust_kn`、`engine_count`、`propellant_kind`；
- `structure_mass_fraction` / `propellant_mass_fraction` 便于和公开数据对照；
- `burn_time_s` 与推力/比冲/推进剂是四组独立输入，`burn_time_deviation_ratio`
  偏离 1 说明参数不自洽——模型只报告，不自动改写参数；
- 整流罩 `fairing_mass_kg` 计入末级需推动的质量，`fairing_usable_volume_m3`
  只用于容积约束，不参与速度预算。

### 任务（`MissionScenario`）

- 载荷：`payloads`、`payload_mass_kg`、`payload_volume_m3`；
- 目标轨道：`OrbitalTarget`（高度、倾角、偏心率、RAAN 等）；
- 发射窗口：`LaunchWindow`（ISO 8601 且必须带时区）+ `launch_epoch_utc`；
- 部署顺序：`DeploymentPlan` / `DeploymentStep`（载荷序号、相对间隔、分离冲量、
  目标槽位），未声明时按载荷列表顺序、统一使用 `deployment_delay_s`；
- 任务约束：`constraints`，用 `with_default_constraints()` 或
  `default_constraints(scenario)` 挂上推荐集合；
- 损失模型：`gravity_loss_mps`、`drag_loss_mps`、`steering_loss_mps` 显式给出。

### 速度预算（`mission.budget`）

`compute_launch_budget(scenario)` 返回冻结的 `LaunchBudget`，把预算拆成
轨道速度、自转收益、平面变化代价、损失和两级速度增量。模拟器、约束和
报告脚本都消费同一个对象，因此不存在"两套公式"的分叉风险。

### 约束（`mission.constraints`）

| 约束 | 默认严重程度 | 判定内容 |
| --- | --- | --- |
| `PayloadMassConstraint` | ERROR | 载荷 + 整流罩 ≤ 运载能力 |
| `PayloadVolumeConstraint` | ERROR | 等效装载容积 ≤ 整流罩可用容积 |
| `DeltaVMarginConstraint` | ERROR | 速度增量余量 ≥ 绝对值和比例要求 |
| `LaunchAzimuthConstraint` | ERROR | 目标方位角落在航区允许范围 |
| `LaunchWindowConstraint` | ERROR | 发射时刻落在声明的窗口内 |
| `DeploymentOrderConstraint` | WARNING | 部署顺序/间隔已声明且可执行 |
| `PowerBudgetConstraint` | WARNING | 轨道平均功率余量 ≥ 要求 |
| `PropellantReserveConstraint` | ERROR | 卫星可用速度增量 ≥ 预留需求 |

规则：

- 约束只读取场景并调用 `physics` / `budget`，不写回场景、不改写物理公式；
- `preflight_gate = True` 的约束在点火前判定（失败时 `elapsed_time_s = 0`），
  `False` 的约束在上升段结束时判定（例如 `DeltaVMarginConstraint`）；
- 内置门槛（运载能力、速度预算）始终执行；场景声明同名约束时以内场为准，
  不会重复评估；
- 插件和经营规则通过 `register_constraint` 注册新类型，配合
  `constraint_to_dict` / `constraint_from_dict` 参与序列化。

## 物理模型边界

当前版本采用可解释的一阶模型：

- 圆轨道速度：`v = sqrt(mu / r)`，周期：`T = 2π sqrt(r³/μ)`；
- 火箭速度预算：分级齐奥科夫斯基公式；
- 发射场自转收益：按纬度估算；发射方位角：`sin(A) = cos(i)/cos(φ)`；
- 平面变化代价：`Δv = 2v sin(Δi/2)`；共面圆轨道间霍曼转移两次点火；
- 重力、阻力和转向损失：场景参数显式给出；
- 大气：指数密度模型；阻力：`D = 0.5 * rho * v² * Cd * A`；
- 链路：自由空间路径损耗、抛物面天线增益和接收功率一阶预算；
- 分系统：轨道平均功率、蚀期能耗、齐奥科夫斯基可用速度增量、下传时间；
- 任务流程：离散事件，不模拟连续姿态和制导。

当前模型明确**不代表**：

- 真实 SpaceX 运载火箭性能或任何具体型号的验收数据；
- 六自由度姿态、制导、导航与控制；
- 真实发射窗口、天气、航区和安全约束；
- 结构载荷、热流、发动机瞬态或级间分离细节；
- 真实 Starlink 卫星内部设计。

后续高保真后端应保持 `MissionResult` 和事件协议稳定，再替换计算器，而不是在
现有一阶函数中悄悄混入不可验证的“修正系数”。

## SpaceX API 的使用原则

随附的 `SpaceX-API-master` 是 `r-spacex/SpaceX-API` 的开源目录服务代码，
服务文档提供 launches、rockets、payloads、starlink 等资源及字段定义。

本项目只通过 `fl_space.integrations.spacex_api.SpaceXApiClient` 做只读访问：

- API 数据用于任务目录、历史记录、载荷质量和轨道元数据的研究输入；
- 外部记录不能直接视为实时状态或飞行认证数据；
- `payload` 记录没有统一外形、迎风面积和功率字段，适配器会使用明确的研究默认值；
- `starlink` 记录可通过 `orbital_elements_from_starlink()` 提取 Space-Track
  OMM 字段并转换为二体六根数；
- Starlink 光链路公开资料可通过 `public_starlink_optical_link_profile()` 作为
  assumption 配置，不代表官方工程规格；
- 重要实验应保存 API 快照、访问时间、资源版本和转换后的配置。

## 验证入口

| 内容 | 入口 |
| --- | --- |
| 领域模型与一阶物理测试 | `tests/test_mission_framework.py` |
| 可运行的端到端示例 | `examples/mission_deployment_demo.py` |
| 格式与静态检查 | `ruff check` + `ruff format`（唯一口径，见 `pyproject.toml`） |

`tests/test_mission_framework.py` 覆盖四类要求：序列化往返、物理量级、
非法输入与边界、最小集成（预算 + 约束门槛 + 部署事件流）。

## 后续扩展边界

建议按以下顺序扩展：

1. 增加热控、数据存储和寿命模块，并把分系统质量与公开数据进行对照；
2. 为发射任务增加时间线检查点、失败原因分类和重试语义；
3. 增加级间分离、姿态角和有效载荷分配的显式模型；
4. 将目标轨道转换为现有多卫星星座与 ISL 模拟；
5. 用高保真传播器替换 `mission.physics` 的特定计算函数；
6. 以插件形式提供经营规则、任务模式、资源市场和自定义 UI。

游戏规则应调用物理/任务服务并记录自己的状态，不要把资金、声望或剧情条件
写进轨道传播器。
