# SpaceFL 项目记忆

## 项目概述
- **项目**: 太空联邦学习 (SpaceFL) 开源研究框架
- **基础工作区**: `D:\project_wang\fl_space_reproduction1` (原论文复现项目)
- **新框架工作区**: `d:\fl_space` (模块化重构)
- **目标**: 将太空FL做成可配置、可扩展的开源研究框架

## 开发计划 (7步)
1. ✅ 环境模拟设置 — 已完成模块化重构 (Kepler + Skyfield双后端, ruff规范化)
2. ✅ 卫星设置（多星簇、自定义注册、可视化）— 2026-05-29
3. ✅ 太空联邦的算法选择 — FedAvg/FedProx/FedBuff (2026-06-01)
4. 数据集
5. 地面基站设置
6. 保存节点（checkpoint机制）
7. 信息展示增强

## Step 1 完成内容 (d:\fl_space)

### 四层模块化架构
- **environment/**: CelestialBody（地球/火星/月球/木星/土星/金星/自定义 + JPL精确参数）、AtmosphereModel（无/恒定/指数）、GroundStation/GroundStationNetwork、coordinate_utils
- **orbit/**: KeplerOrbit（圆形/椭圆/任意倾角）、ConstellationConfig（Walker/星簇/均匀分布）、VisibilityEngine（单星/多星批量）、**SkyfieldOrbitBackend**（SGP4高精度后端）
- **simulator/**: OrbitSimulator（主模拟器，双后端kepler/skyfield）、ContactMatrix（兼容模式+完整模式）
- **config/**: 预设配置（天体/星座/地面站/实验）、配置加载器

### 双后端支持
- `backend="kepler"` (默认): 轻量开普勒二体力学，无外部依赖，计算快
- `backend="skyfield"`: 高精度SGP4传播 + JPL DE421星历，需 `pip install skyfield`
  - 使用 sgp4.Satrec 底层API（避免合成TLE格式问题）
  - 实现eci-to-geodetic坐标转换
  - Topocentric仰角/方位角计算
  - 支持TLE/SGP4 + Kepler要素双输入

### 第三方库调研结论
- **Skyfield** (MIT, 纯Python): ✅ 最佳选择 — 地面站可见性 + JPL星历 + SGP4
- **AstroLib** (C++): ❌ 不可直接集成 — C++静态库，仅Windows+VS，但算法可参考
- **hapsira/poliastro** (MIT, Python): ⚠️ 缺地面站可见性模块
- **Orekit** (Java+Py): ❌ 依赖过重

### 关键特性
- 可自定义行星（默认地球），内置6个天体预设 + JPL精确参数
- 完整接触模式记录所有可见地面站（满足需求：每个卫星与哪些基站传输了信息）
- 兼容原 orbit_sim_v2.py 接口
- 50+ 项自动化测试全部通过（含Skyfield后端）

### 代码规范化 (2026-05-29)
- **工具**: Ruff 0.15.15 (linting + formatting)，配置在 `pyproject.toml` `[tool.ruff]`
- **命令**: `ruff check fl_space/` (检查), `ruff check --fix` (修复), `ruff format fl_space/` (格式化)
- **忽略规则**: RUF002 (中文全角标点), RUF003 (中文注释全角), N806 (轨道力学符号M/E/N/GM)
- **命名例外**: GM, M, M0, E, dE, cos_E, sin_E, N (标准轨道力学符号)
- **行宽**: 100 (科研代码适度放宽)
- 所有 `__init__.py` 必须包含 `__all__` 列表和显式 `as` 重导出

### 技术约定
- Python 3.9+, numpy + 可选 skyfield
- 用 `verbose=False` 控制大规模星座的输出
- sgp4init的no_kozai参数单位为 rad/min（不是rev/day）
- 类型注解用 Python 3.9+ 风格: `list[X]` (非 `List[X]`), `X | None` (非 `Optional[X]`)

## Step 2 完成内容 (2026-05-29)

### 卫星配置系统 (`fl_space/orbit/satellite_config.py`)
- **MultiClusterConfig**: 多星簇星座配置，支持多个不同高度/倾角的星簇
- **ClusterSpec**: 单星簇规格（卫星数、高度、倾角、分布策略、RAAN偏移）
- **SatelliteSpec**: 单星精细配置（完整轨道六要素 + 通信参数）
- 内置预设：`polar_only()`, `starlink_like()`, `mixed_orbit()`, `demo_default()`
- 支持 `to_dict()`/`from_dict()` 序列化

### 卫星注册表 (`fl_space/orbit/satellite_registry.py`)
- `SatelliteRegistry` 全局单例 + `registry` 实例
- 三种注册方式：装饰器 `@registry.register()`、函数 `registry.register_func()`、字典 `registry.register_from_dict()`
- 内置4种卫星类型：polar_6, starlink_20, mixed_10, demo
- 支持 `list_types()` 查看所有已注册类型

### 轨道可视化 (`fl_space/viz/`)
- `OrbitVisualizer`: 2D地图投影（简化大陆轮廓）+ 星下点轨迹 + 地面站
- `plot_dashboard()`: 综合仪表盘（地图 + 统计 + 接触矩阵热力图）
- `quick_plot()`: 一键传入模拟器生成完整可视化
- 深色主题（#0d1117 背景），6种对比色系
- 纯 matplotlib 实现，无 cartopy 依赖

### OrbitSimulator 增强
- 新增 `orbits` 参数：支持直接传入预构建轨道列表（优先级最高）

### Demo 输出 (examples/output/)
- demo1: 极轨4+LEO壳6 双星簇 → 接触率21.7%
- demo2: 自定义注册卫星类型 → 接触率18.5%
- demo3: Walker壳+自定义单星(SSO+赤道) → 接触率27.3%
- demo4: 4种星座对比图
- demo5: 单星星下点轨迹+接触高亮

## 代码规范文档 (2026-06-01)
- 创建了 `CODING_STANDARDS.md` — 完整的代码规范与开发指南
- 包含11个章节：架构概述、命名、类型注解、文档字符串、导入、数据模式、错误处理、Ruff配置、测试、AI辅助开发准则、检查清单
- 特别包含 AI 辅助开发准则（第10章），涵盖提示词模板、AI代码检查流程、严禁事项
- 附录提供常见模式速查和常见错误示例

## 师兄项目优势整合 (2026-06-06) — ISL + Flower + Pydantic + CesiumJS

### ISL 星间链路模块 (`fl_space/isl/`)
- **`los.py`**: WGS84椭球遮挡判定（从 autoFly_Stk earth_los.py 移植），纯数学计算零依赖
- **`base.py`**: ISLCalculator(ABC) + ISLConfig + ISLWindow + NoISLCalculator + 工厂函数 create_isl_calculator()
- **`intra_cluster.py`**: 簇内 LOS 窗口计算（从 autoFly_Stk intra_cluster.py 改写），适配 KeplerOrbit ECEF
- 支持用户自定义 ISL 计算器：`--isl path/to/custom.py:ClassName`

### 集成到 SpaceFL 核心
- `KeplerOrbit.ecef_at_time()`: 新增 ECEF 坐标方法
- `OrbitSimulator`: 新增 isl_config、compute_isl()、isl_windows、isl_stats、isl_active_at()、isl_peers_at()
- `OrbitSimulator`: 新增 get_sat_ecef()、get_all_ecef_at_timeslot()
- `FLConfig`: 新增 isl_enabled/isl_calculator/isl_atmosphere_buffer_km/isl_step_seconds
- CLI: `fls experiment --isl wgs84 --isl-buffer 80`
- `standard_experiment.py`: 支持 isl_enabled 等参数

### Flower 框架集成 (`fl_space/integrations/flower/`)
- **`adapter.py`**: FlowerAdapter — 从 SpaceFL OrbitSimulator 构造，提供 AccessSchedule 兼容 API
- 不重算轨道/不依赖 CSV — 直接消费 ContactMatrix + ISL Windows
- 支持 clients_visible_at / stations_visible_for / next_contact / intra_peers_at 等查询

### Pydantic 强类型配置 (`fl_space/config/`)
- **`schemas.py`**: SpaceFLScenario + WalkerSpec + GroundStation + IntraClusterSpec + FLExperimentSpec（可选依赖 pydantic）
- **`yaml_loader.py`**: YAML 配置加载器，纯 PyYAML 实现（无需 pydantic），输出 sim_kwargs + fl_config 字典

### CesiumJS 3D 可视化 (`web/`)
- **`index.html`**: CesiumJS 3D 地球 + 卫星轨迹/GS-SAT链路/ISL链路实时渲染
- **`server.py`**: FastAPI 服务器，从 OrbitSimulator 生成前端 JSON 数据
- CLI: `fls serve --sats 10 --gs 5 --isl wgs84`

### 模块化可替换设计
- ISL: `--isl disabled|wgs84|path/to/custom.py:ClassName`
- Flower: `FlowerAdapter.from_simulator(sim)` 可选消费
- Pydantic: `from fl_space.config.schemas import ...` 可选导入
- CesiumJS: `fls serve` 独立子命令
- **向后兼容**: 所有新功能默认关闭，不影响现有实验

### Ruff 零告警 + 烟雾测试通过
- 所有新文件 ruff check + ruff format 通过
- 烟雾测试验证：ISL WGS84 遮挡正确、ECEF 计算准确、FL 实验兼容性完好

## CLI 三层架构重构 (2026-06-06)
- **tune** 调参指令：lr/rounds/epochs/batch/mu/seed/dataset/scale/early-stop/workers/non-iid/alpha/device/buffer-size
- **mount** 挂载指令：algo/isl/isl-buffer/isl-step/time-model/time-model-args/backend/body/distribution/staleness/sats/stations/sim-hours/timeslot-min/altitude/inclination/config
- **run** 运行指令：simulate/train/experiment/quick-test/list/export/serve/show
- **Session 持久化**：tune/mount 修改保存到 `.fls_session.json`，run 消费当前 session
- **参数优先级**：CLI --覆盖 > session 值 > 默认值
- **`--` 替换为空格**：tune/mount 使用纯空格分隔（如 `fls tune lr 0.001`），run 保留 `--` 覆盖
- Tab 补全：argcomplete 集成 + 内置 PowerShell 脚本（`fls completion install/ps1`）
- 自定义 help 输出：分类彩色布局，`fls help` 或 `fls -h`
- 完全向后兼容：原有 `_cmd_simulate` 等函数保留逻辑，新架构复用

## AI 修改代码强制性规范 (2026-06-05)
- **每次修改代码后，Agent 必须自动运行 ruff 检查并修复**：
  1. 修改代码完成后，立即运行 `ruff check --fix <修改的文件>` 
  2. 然后运行 `ruff format <修改的文件>` 确保格式一致
  3. 如有无法自动修复的问题（ruff 仍报错），手动修复后再运行上述命令
  4. 目标：修改后的代码通过 ruff 零告警
- **pyproject.toml 已配置忽略规则**: RUF001（中文全角标点）、RUF002（中文全角标点）、E501（行过长由 formatter 处理）等
- **锁定版本**: Ruff 0.15.15，行宽 100

## Step 3 完成内容 — FL算法模块 (2026-06-01)

### 架构设计 — 四组件解耦
每种FL算法分解为四个可独立替换的组件：
1. **ClientSelector** — 每轮选哪些客户端参与（RandomSelector / AsyncSelector）
2. **LocalTrainer** — 本地训练 epoch 次数 + 逻辑（FixedEpochTrainer / ProximalTrainer / AsyncTrainer）
3. **Aggregator** — 何时聚合 + 如何聚合（SyncWeightedAggregator / BufferAggregator）
4. **Evaluator** — 模型评估（StandardEvaluator）

### 三种算法实现
- **FedAvg**: RandomSelector + FixedEpochTrainer + SyncWeightedAggregator（同步加权平均）
- **FedProx**: 复用 FedAvg 的 selector/aggregator/evaluator，仅替换 trainer（增加 proximal term μ·||w-w_global||²）
- **FedBuff**: AsyncSelector + AsyncTrainer + BufferAggregator（FIFO缓冲区K，异步聚合，支持staleness降权）

### 关键文件 (fl_space/fl/)
- `core.py` — 四个ABC + 数据结构（ClientState/ClientUpdate/FLRoundResult）
- `fedavg.py` / `fedprox.py` / `fedbuff.py` — 算法实现 + create_*_components() 工厂
- `models.py` — MLP/SimpleCNN + get_model()/register_model() 注册机制
- `scheduler.py` — 通信调度器（完全独立于FL算法，读取模拟器接触矩阵）
- `server.py` — FLServer 编排器 + FLConfig（支持run_sync/run_async双模式）
- `runner.py` — FLRunner（IID/non-IID Dirichlet数据分配 + 模型创建 + 训练执行）
- `config.py` — 3算法×3规模×3数据集 = 6个组合预设

### 遵循导师建议
- 调度器与算法完全解耦：scheduler 仅处理"何时可通信"，FL算法不涉及通信判断
- 四组件接口清晰：每个组件有明确的输入/输出定义
- 细粒度模块：组件可独立测试、替换、复用
- ruff check + ruff format 通过，Python 3.9+ 类型注解

## 虚拟时间模型 (2026-06-02)

### 新增文件
- `fl_space/fl/time_model.py` — TimeModel(ABC) + SlotTimeModel + PhysicsTimeModel + TimeBreakdown

### 三种模式
1. **SlotTimeModel** (`--time-model slot`): timeslot级粗粒度，可配置 slots_per_epoch/slots_per_mb_down/up
2. **PhysicsTimeModel** (`--time-model physics`): FLOPs/带宽驱动的秒级物理精度，硬件感知
3. **自定义**: `--time-model path/to/file.py:ClassName` 动态导入

### 改动文件
- `core.py`: FLRoundResult 新增 timeslot_start 和 time_breakdown 字段
- `server.py`: FLConfig 新增 time_model/time_model_kwargs；FLServer 集成 TimeModel；run_sync 生成 TimeBreakdown
- `runner.py`: 传递 TimeModel，训练完成摘要显示时间模型信息
- `cli.py`: train 子命令新增 --time-model 和 --time-model-args 参数
- `__init__.py`: 导出 TimeModel/TimeBreakdown/SlotTimeModel/PhysicsTimeModel

### 测试结果
- 全部通过：slot(train=0) 22min / slot(train=3) 37min / slot(全成本) 46min / physics 1h13min
- 每轮输出显示时间分解（下载/训练/上传耗时）

## 性能优化 + 实验系统 (2026-06-04)

### 性能优化
1. **并行客户端训练** (`server.py`): `_train_clients_parallel()` 使用 ThreadPoolExecutor，`num_train_workers` 控制并行度
2. **DataLoader workers** (`runner.py`): `num_workers` 传入 DataLoader，GPU自动 `pin_memory=True`
3. **早停机制** (`server.py`): `early_stop_acc` 触发自动停止

### 新增文件
- `fl_space/utils/__init__.py` + `fl_space/utils/viz.py` — 可视化工具（热力图、准确率对比、时间分解、GS地图、报告器）
- `examples/run_spacefl_experiment.py` — 完整太空FL实验脚本

### 实验设计
- **异构轨道**: 10卫星，高度 350-800 km 均匀分布，同轨道面（倾角53°），不同周期自然时间差
- **地面站对比**: 1(Beijing) / 3(+Svalbard/Santiago) / 5(+Singapore/Washington)
- **FedProx 同步**: 300轮，早停阈值90%，标准FL基线对比（无轨道约束）
- **输出**: JSON报告 + 接触热力图 + 准确率曲线 + 时间分解 + GS地图 + GS对比图

### CLI
- 新增 `fl-space experiment` 子命令，完整参数化（--sats, --gs, --rounds, --device, --train-workers, --data-workers, --altitudes 等）

### 烟雾测试验证
- 3卫星/1GS/5轮完整流水线正常
- 基线FL 5轮达93.57%，SpaceFL(1GS)受轨道约束仅6%接触率（预期行为）

## FEMNIST writer-level 数据集集成 (2026-07-22)
- **新增**: `fl_space/fl/femnist_loader.py` — FEMNIST writer-level 数据加载器（LEAF benchmark）
- **数据源**: HuggingFace `flwrlabs/femnist`（814K 样本, 3597 writers, 62类, 28×28 灰度）
  - 字段: `image` (PIL), `writer_id` (str), `character` (int 0-61)
  - 使用 hf-mirror.com 镜像加速国内下载
- **切分方式**: 按 writer 80/20 train/test（LEAF 默认），1 writer = 1 client
- **缓存**: `data/femnist_cache.pkl` (664MB), pickle 序列化，自动构建/复用
- **集成点**:
  - `config.py` DATASET_PRESETS: `femnist` → MLP(784→128→64→62)
  - `runner.py` prepare_data: femnist 分支 + `_partition_femnist_writers()` 方法
  - `cli.py` tune dataset: 白名单加入 femnist
  - `control_panel.py` 调参面板 + FL训练菜单: 加入 femnist 选项
- **异构特性**: 每个 writer 有 17-23 个不同类别（类别均衡但 feature 漂移大），比 MNIST shard 更接近论文语义
- **测试**: 6项全部通过 — 加载(189s/5writers)、模型创建(112K params)、CLI集成、非法数据集拒绝、字符映射、DATASET_PRESETS

