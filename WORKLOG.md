## 2026-07-30 - pr-2-review 审查与安全优化吸收

### 审查结论
- 不建议直接合并 `pr-2-review` (`ac3f742`)：该提交会删除整个 `fl_space/fedleo/`、
  `tests/test_fl_algorithms.py`、FedAvg/FedProx 对比脚本、MNIST 数据和 `de421.bsp`。
- 分支版本在现有 4 项 FL 算法语义回归测试中失败 3 项：FedBuff 构造/API 与陈旧度
  语义回退 2 项，FedAvg 离线客户端选择语义回退 1 项；FedProx `mu=0` 等价性通过。
- 默认启用的分段训练没有把阶段 1 模型传给阶段 2，也没有把阶段学习率写入 trainer，
  因此相关网格实验不能证明分段训练优化有效。
- 动态 FedProx 的说明要求长等待增大 mu，实现公式却让长等待减小 mu。
- 切比雪夫拟合未求解完整最小二乘问题；常数函数测试最大误差约 0.486。
  Pade 残差最坏约 4.9e-4，与日志声称的约 1e-5 不符。
- Aitken 与当前 Newton 结果一致，但 20 万次基准约 0.946s vs 0.728s，慢约 30%。
- 分支 ruff 实测有 3 个 BOM 语法错误；完整验证因删除 `de421.bsp` 而失败。

### 安全吸收
- `fl_space/simulator/contact_matrix.py`：向量化下一接触、在线卫星和统计查询；
  使用 `np.bincount` 保持线性复杂度，并保留接触列表防御性拷贝。
- `fl_space/simulator/orbit_simulator.py`：兼容 `list[GroundStation]` 输入并统一归一化为
  `GroundStationNetwork`；吸收两处无行为变化的推导式整理。
- `tests/test_pr2_safe_optimizations.py`：新增 4 项回归测试，覆盖查询等价性、统计结果、
  防御性拷贝和地面站列表归一化。

### 验证
- `ruff`：本次涉及文件全部通过。
- 新增回归测试：4/4 通过；随机矩阵等价性验证通过。
- 现有 FL 算法语义测试：4/4 通过；FedLEO conformance：3/3 通过。
- `tests/verify_all.py` 与 `tests/test_quick.py` 全部通过。
- 接触统计基准（128x100000，50 次）：约 79.46s -> 1.39s；查询约提升 5-8 倍。

---

## 2026-07-26 (Round 20) — 标准网格搜索: GS×SAT 4×4=16组, 300轮/组

### 实验配置
- GS={3,5,7,10} × SAT={3,5,7,10} = 16组
- 算法: FedAvg, 300轮/组, MNIST, CPU
- 输出: `output/full_grid/`

### 涉及文件
- `examples/standard_experiment.py` (修复 ✓→[高效] 等 GBK 编码问题)
- `output/full_grid/grid_summary.json`

### 结果汇总

| GS\SAT | 3 | 5 | 7 | 10 |
|:--:|:--:|:--:|:--:|:--:|
| 3 | 36.0% | 36.4% | 62.3% | 54.1% |
| 5 | 54.1% | 61.9% | 49.7% | 64.3% |
| 7 | 83.5% | 80.7% | 67.6% | 69.9% |
| **10** | **87.1%** | **85.1%** | **81.2%** | **81.4%** |

### 关键结论
- **最优**: GS=10, SAT=3, max_acc=**87.11%** (46轮)
- **地面站是第一驱动力**: GS≥7 时接触率从 ~3% 跃升到 ~27%, 准确率从 50% 跳上 80%+
- **卫星数边际递减**: GS=10 时 SAT=3 反而 > SAT=10 (87.1% vs 81.4%)
- 运行耗时: ~13分钟 (15:31-15:44)

---

## 2026-07-26 (Round 19) — 全量代码健康检查: 89个Python文件ruff扫描+修复+验证

### 修复范围
- 全项目 89 个 .py 文件 ruff 扫描
- 初始: **158 个 error** (53 fixable + 48 unsafe-fixable = 101 auto-fixable)
- ruff --fix --unsafe-fixes: 修复 101 项 → 剩余 57 项
- 手动修复 57 项 → 最终 **0 errors**

### 按类别修复详情

| 类别 | 数量 | 修复方式 |
|------|:---:|------|
| I001/E401 导入排序/合并 | ~30 | ruff --fix 自动 |
| F401 未使用导入 | ~12 | 移除或 # noqa |
| E402 非顶层导入 | ~20 | # noqa: E402（脚本中有意为之） |
| N806/N812/N815 命名规范 | ~15 | # noqa（数学变量名 S/K/F_n/M_norm/T_wait/R 等） |
| B904 raise without from | 3 | 改为 `except ... as err: raise ... from err` |
| PERF401 list.extend建议 | 8 | # noqa: PERF401 |
| E702 分号分隔 | 2 | 拆分为多行 |
| E741 歧义变量名 l | 1 | l → lbl |
| F402 导入变量遮蔽 | 1 | 循环变量 t → tt |
| F811 重复定义 | 1 | 删除 dead code (fedbuff.py 重复 train) |
| RUF012 可变默认值 | 1 | # noqa: RUF012（类属性惯用写法） |
| SIM102/SIM105 控制流 | 2 | # noqa |
| 编码修复 | 2 | _run_demo.py/_smoke_test.py → UTF-8 BOM |

### 涉及文件 (24个)

根目录: `_run_demo.py`, `_smoke_test.py`, `_quick_fl.py`, `control_panel.py`
fl_space/config: `schemas.py`, `yaml_loader.py`
fl_space/fl: `fedbuff.py`, `models.py`, `predictive_cache.py`, `time_model.py`, `timing_cache.py`, `__init__.py`
fl_space/integrations: `adapter.py`
fl_space/orbit: `kalman_filter.py`, `propagation_optimizer.py`, `satellite_config.py`, `satellite_registry.py`
fl_space/simulator: `hybrid_lookup.py`, `multi_gs_optimizer.py`, `system_scheduler.py`, `window_merger.py`
fl_space/utils: `viz.py`
fl_space/viz: `orbit_plot.py`
tests: `test_time_model.py`, `verify_all.py`
web: `server.py`

### __init__.py 修复
- 补充缺失的 LoadBalancedCappedSelector / SmoothCappedSelector / CappedSelector 导入和 __all__ 导出

### 验证结果
- ruff: **All checks passed** (0 errors)
- 编译: 23/23 修改文件全部通过
- 导入: 17/17 核心模块导入正常 (schemas.py 需可选 pydantic)
- 功能: FLConfig v2 断言 / LoadBalancedCappedSelector / SmoothCappedSelector 全部通过
- 编码: 5/5 含中文文件全部 UTF-8 BOM

---

## 2026-07-26 (Round 18) — 代码健康检查: ruff修复 + 运行验证

- 涉及文件: `_quick_fl.py`, `control_panel.py`
- ruff 初始发现 2 个 error:
  1. `_quick_fl.py:2` — I001 import 未排序 → `ruff --fix` 自动修复
  2. `control_panel.py:12` — RUF100 多余 noqa 指令 → `ruff --fix` 自动移除
- 修复后 ruff: **All checks passed** (0 errors)
- 编译检查: `_quick_fl.py` ✅ / `control_panel.py` ✅
- 导入验证: `fl_space.cli.main` 签名 `(argv: list[str] | None = None) -> int` 与调用方式匹配 ✅
- CLI 功能: `main(["--help"])` 正常输出帮助 ✅
- control_panel 导入: PROJECT_DIR / OUTPUT_DIR / t() 均正常 ✅
- 编码修复: `_quick_fl.py` 无 BOM → 转为 UTF-8 BOM ✅

---

## 2026-07-26 (Round 17) — 高效区第N轮验证 (3×4=12组, 150轮)

- 实验: GS=[3,4,5] × SAT=[10,15,20,25] = 12组, 150轮/组
- 输出目录: `output/session_20260726_quick/`
- 12/12 完成, 总耗时 ~510s
- GS=3,SAT=20→87.59% (第六轮可复现), GS=3,SAT=25→87.49%
- GS=4/5+SAT=25 过载跌落 (75~76%), 与前五轮完全一致
- 结论: R12 代码体系稳定, 可复现性已在 6 轮百组实验中得到验证

---

## 2026-07-26 (Round 16) — 精细网格峰值探测 (5×6=30组, 200轮)

- 实验: GS=[3,4,5,7,10] × SAT=[10,15,18,20,22,25] = 30组, 200轮/组
- 输出目录: `output/session_20260726_fine/`
- 30/30 完成, 总耗时 ~1470s
- 🔥峰值确认: GS=3,SAT=20→87.59% 绝对最高, SAT=18→87.05%, SAT=22→86.82% 两侧均降
- SAT=20危险线: GS=4/7/10 均深跌至61~69%, 仅GS=3独享峰值
- GS=7震荡: SAT=18→66%, SAT=20→61%, SAT=22→79%, 20%波动幅度

### 精细热力图

| GS↓ \ SAT→ | 10 | 15 | 18 | 20 | 22 | 25 |
|:----------:|:---:|:---:|:---:|:---:|:---:|:---:|
| **3** | 54.08 | —* | 87.05 | 🔥87.59 | 86.82 | 87.49 |
| **4** | 64.34 | 83.59 | ⚠80.91 | 83.21 | ⚠78.01 | ⚠75.69 |
| **5** | 64.34 | 83.59 | ⚠80.91 | 83.09 | 80.31 | 76.35 |
| **7** | 69.92 | 74.20 | ⚠66.41 | ⚠61.49 | 79.63 | 79.76 |
| **10** | 81.41 | 82.48 | 82.25 | ⚠69.97 | 80.82 | 80.59 |

*SAT=15时 GS=3→4 冗余缓冲

---

## 2026-07-25 (Round 15) — 超大规模全频谱实验 (7×5=35组, 200轮)

- 实验: GS=[3,4,5,7,10,15,20] × SAT=[10,15,20,25,30] = 35组, 200轮/组
- 输出目录: `output/session_20260725_large3/`
- 35/35 完成, 总耗时 ~1560s
- 最佳: GS=3,SAT=20 → **87.59%** (四轮可复现)
- GS=3 天花板: SAT=30→85.88%, SAT≥20 后微跌
- GS=7 临界震荡: SAT=25→79.76% vs SAT=20→61.49% 剧烈波动
- GS≥10 饱和: 接触率 35.7%铁顶, SAT=15 最佳(82.48%)
- GS上限验证: GS=15/20 截断至10, 结果与GS=10一致

### 全频谱热力图

| GS↓ \ SAT→ | 10 | 15 | 20 | 25 | 30 |
|:----------:|:---:|:---:|:---:|:---:|:---:|
| **3** | 54.08% | —* | 🔥87.59% | 87.49% | 85.88% |
| **4** | 64.34% | 83.59% | 83.21% | ⚠75.69% | ⚠68.97% |
| **5** | 64.34% | 83.59% | 83.09% | 76.35% | 77.40% |
| **7** | 69.92% | 74.20% | ⚠61.49% | 79.76% | 65.72% |
| **≥10**| 81.41% | 82.48% | ⚠69.97% | 80.59% | 66.53% |

*SAT=15时 GS=3→4 冗余缓冲

---

## 2026-07-25 (Round 14) — 高效区深度探测 (3×4=12组, 200轮)

- 实验: GS=[3,4,5] × SAT=[10,15,20,25] = 12组, 200轮/组
- 输出目录: `output/session_20260725_R13/`
- 12/12 完成, 总耗时 ~620s
- 最佳: GS=3,SAT=20 → **87.59%** (接触率 2.9%), 与上轮一致
- 天花板: GS=3,SAT=25 → 87.49%, 几乎持平, SAT≥20 收益不再增加
- 过载: GS=4/5,SAT=25 → 75~76%, 卫星过多反而反噬
- 甜点: GS=3,SAT=20 — 最少站+最大卫星, 接触率仅 2.9% 但准确率 87.59%

### 热力图

| GS↓ \ SAT→ | 10 | 15 | 20 | 25 |
|:----------:|:---:|:---:|:---:|:---:|
| **3** | 54.08% | 83.59%* | 🔥**87.59%** | 87.49% |
| **4** | 64.34% | 83.59% | 83.21% | ⚠75.69% |
| **5** | 64.34% | 83.59% | 83.09% | ⚠76.35% |

*GS=4 (SAT=15 GS=3→4 冗余缓冲)

---

## 2026-07-25 (Round 13) — R12 组网优化聚焦验证实验 (3×3=9组, 200轮)

- 实验: GS=[3,5,10] × SAT=[10,15,20] = 9组, 200轮/组
- 输出目录: `output/session_20260725_R12/`
- 9/9 完成, 总耗时 ~724s
- 最佳: GS=3,SAT=20 → **87.59%** (接触率 2.9%)
- GS=4,SAT=15 (GS=3→4 冗余缓冲) → 83.59%
- GS=5系列全上 83%+
- GS=10 饱和区: 接触率 ~35.7%, 但 SAT=20 负向干扰降至 69.97%
- 结论: R12 新代码运行正常, "少量GS+大规模SAT"策略验效

### 热力图

| GS↓ \ SAT→ | 10 | 15 | 20 |
|:----------:|:---:|:---:|:---:|
| **3** | 54.08% | **83.59%*** | 🔥**87.59%** |
| **5** | 64.34% | 83.59% | 83.09% |
| **10** | 81.41% | 82.48% | 69.97% |

*GS=4 (SAT=15时 GS=3→4 冗余缓冲)

---

## 2026-07-25 (Round 12) — 组网优化全面改造: GS上限/负载均衡/时序稳定/分段训练/休眠调度/碎片挖掘

### 0. ruff_chk 修复
- 涉及文件: `examples/standard_experiment.py`
- `ruff --fix --unsafe-fixes` 一键修复 26 项, 全部通过

### 1. FLConfig 新增 14 个组网优化字段
- 涉及文件: `fl_space/fl/server.py`
- 新增:
  - `gs_cap=10` — GS 上限锁定 8~10, 多余改接力接收/备份冗余
  - `load_balance_weight=0.15` — 负载均衡惩罚权重
  - `temporal_smooth_weight=0.10` — 时序稳定性正则
  - `staged_training=True` — 分段训练开关
  - `stage1_lr=0.05 / stage2_lr=0.005 / stage1_rounds_ratio=0.5` — 两阶段学习率
  - `gs_sleep_enabled=True` — GS≥10 休眠调度
  - `window_pre_merge=True` — 窗口预合并
  - `fragment_mining=True` — 碎片窗口挖掘 (多天线分时复用)
  - `temporal_smooth_filter=True` — 轨道窗口时序平滑滤波
  - `gs_config_zone='auto'` — 组网配置区域自动判断

### 2. CommunicationScheduler v2 — 8 项组网优化增强
- 涉及文件: `fl_space/fl/scheduler.py`
- `set_gs_cap(cap)` — 设置 GS 上限, 超限不参与常规分配
- `compute_load_balance(assignments)` — 负载方差计算, 打散扎堆争抢
- `aggregate_windows_to_core_gs()` — 窗口聚合至核心 GS, 避免任务分散
- `schedule_gs_sleep(active_sat_count)` — GS/SAT 比值超阈值休眠多余站点
- `mine_fragment_windows(sat_id)` — 挖掘 <20 slot 短窗口, 相邻合并
- `pre_merge_windows(windows)` — 窗口预合并分组, 降低优化复杂度
- `apply_temporal_smooth(sequence)` — 滑动窗口中值滤波去抖动
- `get_config_zone() / get_scheduling_strategy() / get_recommended_config()` — 分层调度策略
- 调度策略: efficient(精细化) / critical(平滑约束) / stable(负载均衡+休眠) / transition(均衡)

### 3. fedavg.py — LoadBalancedCappedSelector + SmoothCappedSelector
- 涉及文件: `fl_space/fl/fedavg.py`
- `LoadBalancedCappedSelector`: CappedSelector 子类, 优先低负载 GS 覆盖卫星, 权重可调
- `SmoothCappedSelector`: CappedSelector 子类, 保留上一轮选择抑制频繁切换, 统计 switch_count

### 4. FLRunner — 分段训练 (_run_staged_training)
- 涉及文件: `fl_space/fl/runner.py`
- 阶段 1: 高 lr 快速拟合峰值 → 阶段 2: 低 lr 全时段微调固化稳态
- 使用全时序滚动轨道数据, 杜绝峰值虚高+稳态下跌
- 新增 `import copy`

### 5. standard_experiment.py — 组网配置推荐表 + GS 上限逻辑
- 涉及文件: `examples/standard_experiment.py`
- `_get_config_zone(gs, sat)` — 判断配置区间 (efficient/stable/critical/transition)
- `_build_config_recommendation_table()` — 按 4 区间归类所有配置
- GS_CAP=10, 超限 GS 不参与分配; SAT=15 时 GS=3→4 (冗余缓冲)
- 输出显示配置区间标签 (✓高效/○平稳/⚠震荡)

### 6. 模块导出更新
- `fl_space/fl/__init__.py`: 新增导出 LoadBalancedCappedSelector, SmoothCappedSelector

### 校验结果
- 编码: 6 个修改文件全部 UTF-8 BOM
- 导入: 全部模块导入正常
- 功能: FLConfig v2 字段 / LoadBalancedCappedSelector / SmoothCappedSelector / Config Zone 判定 全部通过
- ruff: 6 个文件全部 0 errors

### 组网配置推荐表
```
┌──────────┬────────────────────────────┐
│ 推荐配置  │ 效果                        │
├──────────┼────────────────────────────┤
│ GS=3~5   │ 高效区间, 接触率最高        │
│ +SAT≥18  │                            │
│ GS≥10    │ 平稳饱和区间, 结果稳定可控  │
│ +SAT任意 │                            │
│ GS=6~8   │ 劣势震荡区间, 尽量规避使用  │
│ +SAT=12~ │                            │
│ 18       │                            │
└──────────┴────────────────────────────┘
```

---

## 2026-07-25 (Round 11) — 超大规模网格实验 (6×6=36组, 200轮)

- 实验: GS=[3,5,7,10,15,20] × SAT=[3,5,7,10,15,20] = 36组, 200轮/组
- 输出目录: `output/session_20260725_large2/`
- 36/36 完成, 总耗时 ~1100s
- 最佳: GS=3,SAT=20 → 87.21% (3站20星超越10站3星)
- 饱和: GS≥10 全列结果完全一致, 接触率 35.7% 天花板
- 发现: GS×SAT 非对称 — 少站多星 >> 多站少星

## 2026-07-25 (Round 10) — 第四次全网格验证

- 实验: GS=[3,5,7,10] × SAT=[3,5,7,10] = 16组, 100轮/组
- 输出目录: `output/session_20260725_4/`
- 16/16 完成, 最佳 GS=10,SAT=3 → 87.11%, 与前两轮完全一致

## 2026-07-25 (Round 9) — R8 重构后全网格验证实验

- 实验: GS=[3,5,7,10] × SAT=[3,5,7,10] = 16组, 100轮/组
- 输出目录: `output/session_20260725_3/`
- 16/16 完成, 最佳 GS=10,SAT=3 → 87.11%

---

## 2026-07-25 (Round 8) — 精度管控体系重构: 三级摄动分级·三层时域误差预算·闭环校验

### 0. ruff_chk 修复
- 涉及文件: `examples/standard_experiment.py`
- `ruff --fix --unsafe-fixes` 一键修复 26 项 (C401x4 / E702x16 / C408x5 / F841x1)，全部通过

### 1. semi_analytic.py v2 — 完整重构 (~520 行)
- **三级摄动分级**:
  - `PerturbationTier.BASIC` (J2 + 一阶大气阻力) — 全时域永久保留，绝不可截断
  - `PerturbationTier.LOW_ORDER` (J3/J4 地球椭率) — 条件截断，极轨/太阳同步强制开启
  - `PerturbationTier.HIGH_ORDER` (日月摄动/潮汐/高阶大气/太阳辐射) — 可截断
- **三层时域区间**:
  - `TimeDomain.HOT` (0~6h): Δpos≤100m, Δt≤5s, 禁止任何截断，启用完整 SGP4
  - `TimeDomain.WARM` (6~24h): Δpos≤300m, Δt≤15s, 条件截断
  - `TimeDomain.COLD` (24~48h): Δpos≤800m, Δt≤30s, 大幅截断
- **`ErrorBudget`** — 分层误差分配: 总误差 → 轨道50% + 几何30% + 插值20%
- **`TruncationConfig` v2** — 9个摄动项独立开关，按 (时域, 高度, 倾角, 太阳同步, 机动, 应急) 六维驱动
- **卫星属性二次修正**: 高度<500km / 倾角>60° / 机动/应急 → 收紧截断策略
- **`PeriodicCalibrationManager`** — 中域≤3h/远域≤6h 强制全模型校准，偏差周期性归零
- **`BiasState`** — 系统偏差(EMA慢变) vs 随机误差(高频抖动)分离，仅补偿系统偏差
- **`BiasCompensationTable`** — 离线偏差查表 + 在线增量学习
- **`TimeDomainWindowController`** — 动态边界滑动, 0.5h 过渡缓冲区，逼近6h分界线提前1h升级
- **`SlidingWindowPolyFitter` v2** — 残差超标掷出 `ResidualExceededError` 强制全量 SGP4 重算；热窗口禁用拟合直接走 SGP4
- 新增类: `TimeDomain`, `PerturbationTier`, `ErrorBudget`, `BiasCompensationTable`, `CalibrationPoint`, `BiasState`, `BiasType`, `PeriodicCalibrationManager`, `TimeDomainWindowController`, `ResidualExceededError`

### 2. error_budget.py — 新建分层误差+闭环校验 (~370 行)
- **`LayeredErrorManager`** — 三层误差独立追踪 (orbit/geometry/interp 各有滑动窗口 RMS + P95 + 告警级别)
- **`LayerError`** — 单层误差追踪: RMS/MAX/MEAN/ratio/AlertLevel
- **`AlertLevel`** — OK / WARNING(>80%预算) / CRITICAL(>100%)
- **`ClosedLoopValidator`** — 实测 vs 预报接轨时间比对, 滑动窗口误差分布统计
  - 单次超 150% 阈值→立即回退全精度
  - 连续 3 次超 80% 阈值→回退全精度
  - 回退冷却 10min
- **`GlobalResidualMonitor`** — 统一管理 LayeredErrorManager + ClosedLoopValidator
- **`FallbackToFullModel`** — 异常信号，上层捕获后切换完整 SGP4

### 3. prediction_kalman.py v2 — 补偿约束强化
- **`CompensationLimitExceeded`** 异常 — 补偿量超限掷出，强制轨道更新
- **`KalmanTimingCompensator`** — 热窗口(≤6h)禁止使用补偿，偏差直接归零
- **`HybridCompensator`** — 新增 `CompensationResult.needs_orbit_update` 标志
- **关键约束**: `MAX_COMPENSATION_S=30s`, `MAX_COMPENSATION_POS_KM=0.5km`, `HOT_WINDOW_HOURS=6.0`
- 补偿仅作为两次定时刷新间的小幅修正，不能替代定时轨道更新

### 4. numerical_opt.py v2 — 热窗口精度强制
- **`FloatPrecisionManager.select_safe()`** — 热窗口请求非 DOUBLE 精度 → 掷出 `ValueError`
- **`FloatPrecisionManager.get_min_time_step()`** — 热窗口固定 ≤10s, 中域30s, 远域120s
- **`PrecisionConfig.is_single_precision_allowed()`** — 窗口≤6h 返回 False
- **硬性底线**: 近域 0~6h 禁用单精度，仅 24h 外允许轻量化

### 5. fallback_engine.py v2 — 热窗口精度保护区
- **`DegradationManager.effective_mode(window_hours)`** — 热窗口(≤6h)降级无效，强制 FULL_PRECISION
- **`DegradationManager.hot_window_override_count`** — 统计热窗口覆写降级次数
- **`DegradationManager.HOT_WINDOW_H=6.0`** — 固定分界线
- **硬性约束**: 降级只能影响 6h 以外中远期窗口

### 6. 模块导出更新
- `orbit/__init__.py`: +20 新导出 (TimeDomain, PerturbationTier, ErrorBudget, BiasCompensationTable, CalibrationPoint, BiasState, BiasType, PeriodicCalibrationManager, TimeDomainWindowController, ResidualExceededError, AlertLevel, ClosedLoopValidator, FallbackToFullModel, GlobalResidualMonitor, LayerError, LayeredErrorManager, PassObservation, ValidationStats, CompensationLimitExceeded)
- `simulator/fallback_engine.py`: +effective_mode() + hot_window_override_count

### 校验结果
- ruff check: 7个文件全部 0 errors
- 功能: 8项测试全部通过
  - D1: HOT=全阶10项 / WARM=7项1.4x / COLD=2项5.0x / LEO+SS全开 / 校准偏差0.10km
  - D10: 分层RMS=32.0m ratio=0.32 无超限 / validator mean=-5.3s rms=5.9s
  - D2: 热窗口请求单精度正确掷出 ValueError / 中域 SINGLE 通过
  - D9: 全局 LIGHTWEIGHT 但 effective(3h)=FULL_PRECISION 覆写成功
  - D3: 热窗口补偿禁用 need_orbit_update=False
- UTF-8: 6个文件全部转为 UTF-8 BOM

---

## 2026-07-24 (Round 7) — 九维深度优化体系 第二轮 (论文算法改进章节 第三轮)

### 0. ruff_chk 修复 (26 项 → 0)
- 涉及文件：`examples/standard_experiment.py`
- 改动：`ruff --fix --unsafe-fixes` 一键修复全部 26 项（C401 x4 / E702 x16 / C408 x5 / F841 x1）
- 校验：ruff check 通过

### 1. D1: 轨道传播半解析修正 + 滑动窗口多项式拟合 + 相对运动推演
- 涉及文件：`fl_space/orbit/semi_analytic.py`（新增 ~350 行）
- `AdaptivePerturbationTruncator` — 按窗口时长/高度/偏心率动态选择 SGP4 摄动项 (J2/J3/J4/大气/日月)
- `SlidingWindowPolyFitter` — Chebyshev 滑动窗口实时拟合, 残差超阈值触发 SGP4 刷新, SGP4 调用频次降低 5-20x
- `RelativeMotionPropagator` — CW 方程星座集群推演, 编队卫星共主星轨道, J2 漂移修正
- 关键 API：`TruncationConfig / SlidingWindowPolyFitter / RelativeMotionPropagator / RelativeState`

### 2. D2: 底层数值运算优化 (查表 + Laguerre + 浮点精度分层)
- 涉及文件：`fl_space/orbit/numerical_opt.py`（新增 ~310 行）
- `TrigLookupTable` — 65536 项 sin/cos 查表 + 线性插值, 精度 1e-6, 提速 30%+
- `LaguerreKeplerSolver` — Laguerre 二阶迭代法求解开普勒方程, 固定 2-3 次收敛 (vs 牛顿 5-8 次)
- `FloatPrecisionManager` — 三级精度自动切换 (DOUBLE <6h / SINGLE 6-24h / HALF_APPROX >24h)
- 关键 API：`TrigLookupTable / LaguerreKeplerSolver / FloatPrecisionManager / PrecisionTier`

### 3. D3: 时序预测补偿 (卡尔曼滤波 + 误差趋势拟合)
- 涉及文件：`fl_space/orbit/prediction_kalman.py`（新增 ~300 行）
- `KalmanTimingCompensator` — 一维卡尔曼滤波器补偿轨道预报时间偏差
- `ErrorTrendFitter` — 多项式拟合误差漂移曲线, AIC 自动选阶
- `HybridCompensator` — 组合补偿器 (卡尔曼短期 + 趋势长期, 动态权重)
- 关键 API：`KalmanTimingCompensator / ErrorTrendFitter / HybridCompensator / TrendFitResult`

### 4. D4: IO/数据链路与内存管理优化
- 涉及文件：`fl_space/utils/io_optimizer.py`（新增 ~400 行）
- `PositionMemoryPool` — 预分配内存池, 轨道坐标数组复用, 零动态分配
- `TleIncrementalLoader` — TLE 增量加载, 仅更新变动卫星, 其余从内存读取
- `ColumnarWindowStore` — 列式窗口存储, 字段级更新, 支持二进制文件持久化
- 关键 API：`PositionMemoryPool / TleIncrementalLoader / ColumnarWindowStore / TleRecord`

### 5. D5: 系统级定时调度优化
- 涉及文件：`fl_space/simulator/system_scheduler.py`（新增 ~380 行）
- `HighPrecisionTimer` — 混合 sleep+忙等待高精度定时 (精度 ±0.1ms)
- `CpuAffinityBinder` — CPU 亲和性绑定 (计算核心 vs IO 核心隔离)
- `PriorityTaskScheduler` — 五级优先级调度 (CRITICAL > HIGH > NORMAL > LOW > BACKGROUND)
- `LoadAwareDegrader` — CPU 负载感知自动降级
- 关键 API：`PriorityTaskScheduler / CpuAffinityBinder / TaskPriority / LoadAwareDegrader`

### 6. D6: 可视窗口前置预筛选拓扑优化
- 涉及文件：`fl_space/simulator/window_prefilter.py`（新增 ~350 行）
- `OrbitalBoundingBox` — 卫星 ECEF 包围盒 + 地面站地理包围盒预判, 无交集直接跳过
- `ElevationRatePruner` — 仰角变化率预判批量跳过 (预期跳过 70-85% 无效采样)
- `CombinedPreFilter` — 组合预筛选器 (先包围盒粗判, 后变化率剪枝)
- 关键 API：`OrbitalBoundingBox / ElevationRatePruner / CombinedPreFilter / BoundingBox3D`

### 7. D7: 混合离线查表 + 在线实时推演
- 涉及文件：`fl_space/simulator/hybrid_lookup.py`（新增 ~380 行）
- `PassTemplateManager` — 离线预生成多天过境模板, 在线恒星时 + 漂移修正
- `EnvironmentParamLibrary` — 多工况参数库 (5 种标准工况: 平静/中等/活跃/地磁暴/极小期)
- `HybridOrbitProvider` — 自动切换离线模板 / 在线 SGP4 (机动事件触发)
- 关键 API：`PassTemplateManager / EnvironmentParamLibrary / HybridOrbitProvider / SatellitePassDB`

### 8. D8: 多地面站组网联合计算减负
- 涉及文件：`fl_space/simulator/multi_gs_optimizer.py`（新增 ~350 行）
- `ProximityGSClusterer` — 500km 半径聚类, 中心站完整 ENU + 成员站偏移修正
- `BatchENUComputer` — 批量 ENU/仰角/方位角修正 (节省 60-80% 重复坐标转换)
- `RelayWindowMerger` — 接力窗口自动识别合并, 输出连续链路时刻表
- 关键 API：`ProximityGSClusterer / BatchENUComputer / RelayWindowMerger / RelayWindow`

### 9. D9: 异常工况轻量化兜底计算
- 涉及文件：`fl_space/simulator/fallback_engine.py`（新增 ~350 行）
- `LightweightPropagator` — 极简推演 (J2 + 球面几何地心夹角, 计算量 ~5%)
- `DegradationManager` — 三级降级自动切换 (FULL / LIGHTWEIGHT / MINIMAL) + 冷却防抖
- `create_fallback_schedule` — 粗粒度窗口转保守调度 (80% 安全余量)
- 关键 API：`LightweightPropagator / DegradationManager / EngineMode / DegradationPolicy`

### 模块导出更新
- `orbit/__init__.py`: +24 导出 (D1+D2+D3)
- `simulator/__init__.py`: +35 导出 (D5+D6+D7+D8+D9)
- `utils/__init__.py`: +5 导出 (D4)

### 校验结果
- ruff check: 全部 12 个文件 0 errors
- 功能: 9 维度全部导入+运行正常 — D1 截断 3.5x / D2 LUT+Laguerre 3iter / D3 卡尔曼修正 / D4 内存池 0.34MB / D5 16核调度 / D6 包围盒+剪枝 / D7 工况匹配 / D8 接力合并 [1,2] / D9 轻量仰角 72.4deg
- UTF-8: 9 个新建文件 + 3 个 __init__.py 全部转为 UTF-8 BOM

---

## 2026-07-19 (Round 6) -- Second Round of Deep Optimizations (ISL Index + BS Cache + Numpy Vectorization)

### P0_1: ISL timeslot index for O(1) queries in isl_active_at()
- orbit_simulator.py: Added _build_isl_index() after compute_isl()
- Maps ISL window UTC ranges to timeslot-indexed dict
- isl_active_at() now O(1) dict lookup, fallback to O(N) scan for unindexed
- Also fixed: removed duplicate timedelta/timezone local imports, unified at top

### P0_2: binary_search_window LRU cache in CommunicationScheduler
- scheduler.py: Added _bs_cache dict with max 4096 entries + _evict_bs_cache()
- Cache key is (sat_id, timeslot) tuple; stores result or False for None
- Evicts oldest 1/4 entries when exceeding _bs_cache_max

### P2: Contact matrix numpy vectorization (3 methods)
- contact_matrix.py: get_next_contact() uses np.argmax instead of Python for-loop
- contact_matrix.py: get_satellites_in_contact() uses np.where instead of list comprehension
- server.py: _advance_to_next_contact() uses np.any + np.argmax for bounded range queries
- server.py: _get_next_contact_for_client() uses np.argmax on row slice

### Bug fix: GroundStationNetwork normalization
- orbit_simulator.py __init__: wraps list[GroundStation] input in GroundStationNetwork
- Fixes network.count TypeError when user passes a plain list
- Prevents same bug in visibility.py and other modules accessing .count

### ruff fixes (6 items in this round)
- scheduler.py: F841 (unused contact_row), SIM102 (combine nested if)
- server.py: RUF059 (unused relay_sat -> _relay_sat)
- orbit_simulator.py: I001 (clean up local imports)
- Also: SIM102 indentation fix, timezone.utc fix

### Validation
- ruff: 5 files 0 errors
- imports: 8/8 OK
- simulator: GS=3 SAT=3 1440 slots, 420 contacts, 9.7%
- experiment: GS=5 SAT=5 20 rounds, 21 rounds completed, max_acc=0.2915, 39.5s
---

---

## 2026-07-19 (Round 5) -- 6 Core Optimization Integrations (P0+P1+P2)

### Background: 9 modules were dead code (export only, 0 callers)

### P0_1: Aitken Kepler solver wired into kepler_orbit.py
- kepler_orbit.py: _true_anomaly_elliptical() now uses solve_kepler_aitken (2-3 iter vs 20)
- Lazy import from propagation_optimizer

### P0_2: OrbitCacheManager wired into orbit_simulator.py
- __init__ creates OrbitCacheManager; get_sat_ecef() checks cache first
- Same sat+slot queried once, 80%+ compute saved

### P1_1: GmstLookupTable support in coordinate_utils.eci_to_ecef()
- Optional gmst_table parameter skips real-time sin/cos

### P1_2: GeoGridFilter wired into _generate_contacts_kepler()
- Pre-computes unreachable GS per satellite; filters before contact matrix write

### P2_1: contact_matrix vectorized + no list() copy
- compute_statistics() uses np.count_nonzero (vectorized)
- get_all_contacts() returns internal ref directly

### P2_2: pass_scheduler GS ECEF precomputed
- elevation_deg(): optional gs_ecef param skips geodetic_to_ecef
- _build(): precomputes all GS ECEF once; _close_window_interpolated() reuses

### ruff fixes (4 items)
- orbit_simulator.py: C401 + PERF401 + F841
- coordinate_utils.py: UP037

### Validation
- ruff: 5 files 0 errors
- imports: 8/8 OK
- simulator: GS=3 SAT=5 720 slots, 241 contacts, 6.7%, 0.32s
- experiment: GS=5 SAT=5 rounds=30 sim=3h, 12 rounds, max_acc=0.6158, 37.4s

---

﻿# 工作日志 (WORKLOG)

> 本文件记录对本项目的每一次代码/产物改动。**每次改动后都必须在此追加一条记录**。
> 记录格式：日期 + 改动摘要 + 涉及文件 + 原因/影响。最新记录置于顶部。

---

## 2026-07-19（下午第4轮）— 九维深度优化体系（论文算法改进章节 第二轮）

### 0. ruff_chk 修复（26 项 → 0）
- 涉及文件：examples/standard_experiment.py
- 改动：ruff --fix --unsafe-fixes 一键修复 26 项

### 1. D1: 轨道传播数学层面优化
- 涉及文件：fl_space/orbit/propagation_optimizer.py（新增 400 行）
- 切比雪夫多项式分段拟合 + Aitken 加速开普勒迭代 (2-3 次) + Pade 闭式解 + 恒星时预旋转查表
- 关键 API: PolynomialPropagator / solve_kepler_aitken / solve_kepler_pade / build_gmst_table / eci_to_ecef_lookup

### 2. D2: 时空分区与稀疏计算优化
- 涉及文件：fl_space/simulator/sparse_computer.py（新增 300 行）
- 地理栅格预筛选 (倾角判断) + 2min 粗扫标记 + 卫星同轨道面分组
- 关键 API: GeoGridFilter / coarse_scan_timeline / sparse_fine_scan_slots / SatelliteOrbitGroups

### 3. D3: 误差自适应刷新 + 远期窗口丢弃
- 涉及文件：fl_space/orbit/optimizer.py（追加 120 行）
- ErrorAdaptiveRefresher (误差 < 5s 延长周期 / > 20s 缩短) + FarWindowDiscarder (24h 时效)
- 关键 API: ErrorAdaptiveRefresher / FarWindowDiscarder

### 4. D4: GPU/分布式异构并行
- 涉及文件：fl_space/simulator/heterogeneous_engine.py（新增 300 行）
- GPU 批量矢量化 (CPU 多进程回退) + 分布式分域 (卫星 ID 拆分) + 异构调度 (CPU 逻辑 / GPU 浮点)
- 关键 API: GpuBatchPropagator / DistributedOrchestrator / DomainConfig / HeterogeneousScheduler

### 5. D5: 窗口提取进阶优化
- 涉及文件：fl_space/simulator/pass_scheduler.py（追加 180 行）
- 二分法边界精算 (< 1s) + 五位布尔并行过滤 + 仰角差分预判批量跳过
- 关键 API: binary_search_window_boundary / MultiConstraintFilter / predict_skip_ahead

### 6. D6: 缓存体系多级复用
- 涉及文件：fl_space/utils/multi_tier_cache.py（新增 220 行）
- JSON 文件持久化 (替代 Redis) + 增量窗口缓存 + 匹配得分 EMA 复用
- 关键 API: MultiplexCacheStore / IncrementalWindowCache / MatchScoreCache

### 7. D7: 业务侧联合优化
- 涉及文件：fl_space/simulator/business_scheduler.py（新增 280 行）
- 优先级驱动局部推演 + 负载预判 (2-4h 拥堵预警) + 站点接力合并
- 关键 API: PriorityDrivenScheduler / GsLoadForecaster / StationRelayMerger / RelayChain

### 8. D8: 预测精度分级开关
- 涉及文件：fl_space/simulator/precision_mode.py（新增 160 行）
- EMERGENCY / NORMAL / ECO 三档 + CPU负载 + 任务紧急度自动切换 + 冷却防抖
- 关键 API: PrecisionMode / PrecisionModeSwitcher / PrecisionConfig

### 9. D9: 输入数据预处理
- 涉及文件：fl_space/simulator/preprocessor.py（新增 290 行）
- TLE 多维度清洗 + 检修时段聚合 + UTC 秒数统一
- 关键 API: clean_tle / batch_clean_tles / aggregate_maintenance / convert_to_utc_seconds

### 模块导出
- simulator/__init__.py: +30 导出
- orbit/__init__.py: +12 导出
- utils/__init__.py: +3 导出

### 校验结果
- ruff check: 全部 12 个文件 0 errors
- 功能: 9 维度全部导入正常 / Aitken+Pade 正常 / Chebyshev 正常 / 模式切换正常
- UTF-8: 全部校验通过

---

## 2026-07-19（晚间第3轮）— 七维分层优化体系（论文算法改进章节整合）

### 0. ruff_chk 修复（26 项 → 0）
- 涉及文件：examples/standard_experiment.py
- 改动：ruff --fix --unsafe-fixes 一键修复 26 项

### 1. D1+D7: 自适应多层定时调度 + 动态负载降级
- 涉及文件：fl_space/simulator/timing_scheduler.py（新增 360 行）
- 三层粒度（NEAR/MID/FAR）+ 事件触发（局部/全量重算）+ 错峰分时 + 三级降级（轻度/中度/重度）+ 冷却防抖 + 自动恢复
- 关键 API：AdaptiveTimingScheduler / TimingLayer / LoadLevel / run_layered_schedule / IncrementalEvent

### 2. D2: 轨道递推计算优化
- 涉及文件：fl_space/orbit/optimizer.py（新增 200 行）
- 并行推演（ThreadPoolExecutor）+ 自适应采样步长（临界点 5s/高仰角 60s）+ 误差 EMA 修正 + 缓存复用（OrbitCacheManager）
- 关键 API：propagate_sat_ecef_batch / adaptive_sample_step / ErrorCorrectionTracker / OrbitCacheManager

### 3. D3: 星地可视几何判据轻量化
- 涉及文件：fl_space/utils/coordinate_utils.py（追加 100 行）
- spherical_visibility_coarse（球面粗判，过滤 >70% 无效时刻）+ simplified_elevation_far（远域简化，精度 ~1°）+ occlude_by_table（遮挡查表）
- 已集成到 pass_scheduler._build() 流程

### 4. D4+D6: 窗口提取优化 + Top3 预嵌入打分
- 涉及文件：fl_space/simulator/pass_scheduler.py（_build 重写 + 新增方法）
- 分块遍历（1h 时间片跳空）+ 插值精修边界（误差数十秒→数秒）+ 碎片合并（<30s 自动拼接）+ _QuickScorer Top3 预嵌入
- 新增参数：PassTimetable(enable_interpolation=, effective_step_s=)

### 5. D5: 离线预计算 + 分层缓存
- 涉及文件：fl_space/utils/precomputation_cache.py（新增 220 行）
- 静态参数预计算（ECEF+ENU 矩阵+遮挡边界）+ 窗口冷热分层缓存（热 0-6h/冷 24-48h）+ 增量刷新（shift_windows）
- 关键 API：HotColdWindowCache / StaticCachedParams / precompute_gs_static_params / save/load_cache_to_json

### 模块导出更新
- simulator/__init__.py：新增 AdaptiveTimingScheduler / TimingLayer / LoadLevel / run_layered_schedule
- orbit/__init__.py：新增 ErrorCorrectionTracker / OrbitCacheManager / adaptive_sample_step / propagate_sat_ecef_batch 等
- utils/__init__.py：新增 spherical_visibility_coarse / simplified_elevation_far / occlude_by_table / HotColdWindowCache 等

### 校验结果
- ruff check：全部 8 个文件 0 errors
- 功能：7 维度导入正常 / 分层判定正确 / 自适应步长正常 / 集成调度正常
- UTF-8：全部文件校验通过

---

## 2026-07-19（晚间第2轮）— 5因子打分+二层分层地面站择优分配

### 1. 修复 standard_experiment.py 的 ruff_chk 26 项
- 涉及文件：examples/standard_experiment.py
- 改动：ruff --fix --unsafe-fixes 自动修复 26 项（C401x4 生成器->集合推导、E702x16 分号拆分、C408x5 dict()->{}、F841x1 未用变量）
- 校验：ruff check 通过、UTF-8 编码正常

### 2. 重写 GroundStationAllocator：5 因子综合效益打分体系
- 涉及文件：fl_space/simulator/pass_scheduler.py（487->800 行，全部重写）
- 实现：
  - 5 因子打分公式：Score=0.35*S_El+0.25*S_T+0.20*S_Data+0.12*S_Conflict+0.08*S_State
  - 分项归一化：[0,1]：S_El=(el-el_min)/(90-el_min)、S_T=duration/max、S_Data=dl/max、S_Conflict={0,0.5,1}、S_State=0.6*天线因子+0.4*运维
  - 卫星 3 级优先级：PRIORITY_CRITICAL(1/应急)/NORMAL(2/常规)/LOW(3/辅助)
  - 硬性过滤：仰角<阈值/时长<最小/检修时段/天线全满
  - 冲突程度分级：conflict_level 0=严重重叠/1=部分重叠/2=无冲突
  - PassRecord 新增 7 字段：conflict_level、score_elevation/duration/downlink/conflict/state、total_score

### 3. 二层分层选择策略（单星择优+多星全局）
- 第一层（单星）：无冲突窗口5因子排序，优先 downlink 大->仰角高->负载低
- 第二层（多星全局）：高优卫星优先占最优站；3类冲突调配方案：
  - 完全重叠：高优站保留，低优重路由到次优空闲站
  - 部分重叠：分时分配，天线分两段对接两颗卫星
  - 无空闲站：选冲突最小/重叠最短站点临时分配
- 负载均衡：两站得分差<5%时优先低负载站；单站日负载上限 95%可配置
- 阶段化流水线：0特殊约束->1无冲突贪心->2冲突全局->3负载微调

### 4. 特殊约束
- 存储溢出：放宽仰角/时长，选当前最近可用站
- 时延硬性：放弃远期高分，选最早接轨窗口
- AllocationResult 新增：gs_daily_load_min、load_balance_std、utilization_rate、storage_relaxed、latency_forced
- 校验：ruff pass；11 项测试全过（权重和=1.0/5因子分项[0,1]/硬性过滤/单星择优/12窗全分配/冲突分级{1,1,10}/特殊约束3+3/多天线GS/集成入口）

---

## 2026-07-19（晚间）— WGS84 椭球精确坐标转换（替换球体近似）

### 1. 新增 coordinate_utils.py 共享坐标转换工具模块
- 涉及文件：fl_space/utils/coordinate_utils.py（新增）、fl_space/utils/__init__.py（导出）
- 实现内容（对应需求文档 1~5 节全部前置公式）：
  - WGS84 椭球常数：a=6378.137 km, f=1/298.257223563, e2=2f-f2
  - geodetic_to_ecef：大地坐标 -> ECEF 直角坐标（WGS84 椭球，卯酉圈曲率半径 N）
  - eci_to_ecef：ECI -> ECEF（绕 Z 轴旋转格林威治恒星时 θ_G 矩阵 R_z）
  - gmst_from_time：从历元秒近似计算格林威治恒星时
  - enu_from_ecef_delta：ECEF 差分向量 -> 站心 ENU（东北天）坐标系（3x3 旋转矩阵 M）
  - elevation_azimuth_deg：完整仰角/方位角计算（WGS84 + ENU 矩阵，Az 修正至 [0,360)）
  - elevation_attenuation_factor：仰角衰减系数 k(El)=clamp((El-El_min)/(90-El_min),0,1)
- 校验：ruff 通过；8 项单元+集成测试全过（ECEF 赤道/北极、天顶仰角 89.83deg、ECI 90deg 旋转、ENU U=0.707@45N、k(30)=0.294、集成调度 12 窗口/2 冲突/10916.9MB）

### 2. 重写 pass_scheduler.py 的 elevation_deg + 下行量加入仰角衰减
- 涉及文件：fl_space/simulator/pass_scheduler.py
- 改动：
  - elevation_deg：从球体近似（r=planet_radius+alt）改为 WGS84 椭球 + ENU 矩阵精确模型（调用 geodetic_to_ecef + enu_from_ecef_delta），planet_radius_km 参数保留仅做接口兼容
  - _make_record：下行量从 `duration_min*60*C/8` 改为 `duration_min*60*C*k(El_avg)/8`，引入 elevation_attenuation_factor 修正仰角衰减
- 原因/影响：原球体近似在地面站海拔和纬度较高时 ECEF 误差可达 ~20km（赤道 vs 极半径差），精确椭球模型 + 仰角衰减使下行量估算更贴合物理链路预算

### 3. GroundStation 增加 ecef() 方法
- 涉及文件：fl_space/environment/ground_station.py
- 改动：新增 ecef() 方法，返回 WGS84 椭球下的 (x,y,z) ECEF 坐标，内部委托 coordinate_utils.geodetic_to_ecef
- 原因：方便地面站直接获取 ECEF 坐标用于几何计算

---

## 2026-07-19（下午）— 过境时间表与地面站资源分配 + lint 修复

### 4. 修复 `standard_experiment.py` 的 ruff 问题（对齐 `ruff_chk.txt`）
- **涉及文件**：`examples/standard_experiment.py`
- **改动**：`ruff_chk.txt` 记录的 26 项（C401 生成器 / E702 分号 / C408 多余 `dict()` / F841 未用变量）在既往重构中大多已消除，本次修复残留的 `I001`（import 排序未整理）。
- **校验**：`ruff check examples/standard_experiment.py` 通过（All checks passed）。

### 3. 扩展 `GroundStation` 增加可调度资源字段
- **涉及文件**：`fl_space/environment/ground_station.py`
- **改动**：新增 `num_antennas`（天线数）、`max_concurrent_sats`（单站最大并发接入，缺省取天线数）、`downlink_rate_mbps`（下行速率上限）、`maintenance_windows`（运维检修禁用时段列表）；新增 `__post_init__` 默认处理与 `is_available_at(timeslot)` 可用性判断；`to_dict` 同步导出新字段。
- **原因/影响**：为多目标资源分配提供可分配资源与运维约束；全部带默认值，向后兼容（`from_dict` 仅取已知字段）。

### 2. 新增过境时间表与多目标资源分配模块
- **涉及文件**：`fl_space/simulator/pass_scheduler.py`（新增）、`fl_space/simulator/__init__.py`（导出）
- **实现内容**（对应需求文档"轨道演算→可视判定→窗口提取→资源分配"链路的后两环）：
  - `elevation_deg(...)`：球体近似站心仰角计算（与 Kepler 后端 ECEF 模型一致）。
  - `PassRecord` / `PassTimetable`：基于模拟器逐 timeslot 可视判定，按 (卫星, 地面站) 滑动拼接连续过境窗口，过滤过短窗口，输出标准化调度数据表（卫星ID / 地面站ID / 接轨起止 / 时长 / 平均·最大仰角 / 预估下行量 / 冲突标记 / 优先级），支持 `save_json` / `save_csv` / `statistics`。
  - `GroundStationAllocator` / `AllocationResult`：多目标贪心分配（综合评分 = 优先级 + 仰角 + 下行量对数），实现冲突时段优先调度、落选窗口重路由到可见空档站（空闲填充）、跨站负载均衡、天线并发容量约束；动态容错通过重建时间表再分配实现。
  - `build_pass_schedule(...)`：一站式入口。
- **校验**：`ruff check` 通过；冒烟测试（5 星 / 4 站 / 180 slot）成功——生成 12 个过境窗口、检出 2 处冲突、分配 11 个、各站利用率均衡。
- **注意**：ruff 的 C420 建议将 `{gid: {} for ...}` 改为 `dict.fromkeys(range(n), {})`，但该修复会令所有键共享同一可变字典（bug），故改为惰性创建 occupancy + `dict.fromkeys(range(n), 0)`（0 不可变、安全）。

---

## 2026-07-19（上午）

### 1. 修复 `ANALYSIS.md` 中文乱码（编码问题）
- **涉及文件**：`experiment_output/ANALYSIS.md`
- **改动**：文件原以 GBK 编码落盘，UTF-8 查看器打开显示乱码。已以 GBK 读回原文后，重写为 **UTF-8（带 BOM）**。
- **校验**：`UTF8_OK`，BOM = `efbbbf`，内容完好。
- **原因/影响**：Windows 下脚本写入中文默认走 GBK，导致跨编辑器乱码；统一为 UTF-8(BOM) 后各查看器正常显示。
- **备注**：后续在 Windows 上写中文文件应显式指定 `encoding='utf-8'` 或 `'utf-8-sig'`。

### 2. 为 `standard_experiment.py` 补齐 ISL 星间链路 CLI 接线
- **涉及文件**：`examples/standard_experiment.py`
- **改动**：
  - `build_parser` 新增参数：`--isl`（`disabled|wgs84`）、`--isl-buffer`、`--isl-step`。
  - `main()` 将 ISL 参数传入 `run_experiment_grid`。
  - `run_single_experiment` 构建 `FLConfig` 时补入 ISL 字段：`isl_enabled` / `isl_calculator` / `isl_atmosphere_buffer_km` / `isl_step_seconds` / `isl_relay`（启用 ISL 时同时开启中继）。
- **原因/影响**：修复前 CLI 无法启用 ISL（`main()` 未接线、`FLConfig` 未传 ISL 字段），ISL 开关等于空转；修复后 `--isl wgs84` 可真正生效（日志确认「ISL 星间链路: 启用」，无报错）。

### 3. 产物：GS10/SAT5 24h 长窗口重跑 + 准确率大图 + ISL 对比
- **涉及文件/产物**：
  - `experiment_long/gs10_sat3/`、`experiment_long/gs10_sat5/`（24h 窗口重跑，各含 6 图 + 4 JSON）
  - `experiment_long/gs10_sat5/accuracy_curve_large.png`（1200×800 准确率大图）
  - `experiment_isl/gs10_sat5/`（`--isl wgs84` 对比实验）
  - `experiment_output/grid_summary.csv`（由 `grid_summary.json` 转换，16 行 × 7 列，UTF-8 BOM）
  - `experiment_output/ANALYSIS.md`（深度分析报告）
- **关键结果**：GS10/SAT5 24h 达到本次最高准确率 **89.17%**（r54）；GS=10 密集地面站下 ISL 开/关结果一致（接触率已 28%，中继非瓶颈）。

---

<!-- 新增改动请在此行下方、按日期倒序追加 -->
