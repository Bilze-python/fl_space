# SpaceFL 可视化智能实验平台 — 搭建设计方案

**版本**: v0.1（设计稿）
**日期**: 2026-08-10
**目标**: 在现有 SpaceFL 框架（CLI + Cesium 3D Web 面板）基础上，构建一个「可视化 + 实时运行 + AI 辅助 + 可存档 + 可扩展」的一体化实验平台。

---

## 0. 需求总览

| 需求 | 现状 | 目标能力 |
|------|------|---------|
| ① 网页调参 | CLI `tune/mount` 改 `.fls_session.json` | Web 表单可视化调参，与终端双向同步 |
| ② 地图源接入 | CesiumJS + NaturalEarthII 离线底图 | 支持 Cesium Ion Token 接入真实影像/地形底图，可插拔多地图源 |
| ③ AI 辅助 | 无 | 本地 AI 助手：代码修改（带 diff 预览/回滚）、实验解读、文献问答 |
| ④ 文档阅读 | 无 | 内置文档阅读器（Markdown / PDF / 代码浏览） |
| ⑤ 保存参数与结果 | `output/` JSON + PNG | 实验配置快照 + 结果归档 + 对比分析 |
| ⑥ 暂停/保存/载入实验 | 无（一次性运行） | 类文字游戏存档：中途暂停、存档、续跑 |
| ⑦ 文献库 | `文献/` 平铺目录 | 结构化文献库：分类、标签、搜索、在线阅读 |
| ⑧ 算法导入工作台 | 手动改代码 | 拖拽导入新算法/优化，自动分析 + 冲突检测 + 注册 |

---

## 1. 总体架构

```
┌──────────────────────────── 浏览器 (前端 SPA) ────────────────────────────┐
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ │
│  │ 控制台   │ │ 3D 视图   │ │ 实验管理  │ │ 文献/文档 │ │ AI 助手   │ │ 算法   │ │
│  │ (调参/   │ │ (Cesium) │ │ (运行/   │ │ (阅读/   │ │ (对话/   │ │ 工作台 │ │
│  │ 实时日志) │ │ 多地图源  │ │ 存档/续跑)│ │ 搜索)    │ │ 代码/解读)│ │ (导入/ │ │
│  │          │ │          │ │          │ │          │ │          │ │ 冲突)  │ │
│  └─────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────┘ │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ HTTP REST + WebSocket (实时事件流)
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                        FastAPI 后端 (web/ 扩展)                              │
│  ┌──────────┐ ┌────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐  │
│  │ session  │ │ 实验编排器   │ │ 存档引擎     │ │ AI 服务     │ │ 算法工作台  │  │
│  │ 服务      │ │ Experiment │ │ Checkpoint  │ │ (Provider   │ │ Workbench  │  │
│  │ (读/写    │ │ Manager    │ │ Manager     │ │  抽象层)    │ │ (AST 分析/ │  │
│  │ .fls_session│ (启动/暂停/ │ │ (pickle+JSON)│ │             │ │  冲突检测) │  │
│  │ .json)    │ │ 恢复)      │ │             │ │             │ │            │  │
│  └──────────┘ └────────────┘ └─────────────┘ └─────────────┘ └────────────┘  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  适配层: CLI 桥 (复用 fls run 逻辑) · Pydantic 校验 · 预设服务           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                     SpaceFL 核心 (fl_space/)                                │
│  environment · orbit · simulator · isl · fl(algorithms) · fedleo · config   │
└──────────────────────────────────────────────────────────────────────────────┘
```

**设计原则**：
1. **零侵入核心**：所有新能力放在 `web/` + 新增 `web_platform/` 包，不改动 `fl_space/` 核心算法，保证原有 CLI/测试不回退。
2. **后端复用 CLI 逻辑**：实验运行直接调用 `fl_space.cli` 的函数（如 `cmd_run_fedleo` 背后的 `run_fedleo_vs_baseline`），而非重新实现，保证"终端能跑的网页也能跑"。
3. **存档/恢复 = 状态序列化**：在 `FLRunner` 轮次循环外部做检查点（checkpoint），不侵入算法内部。
4. **AI 只读感知 + 受控写入**：AI 修改代码一律生成 diff → 用户确认 → 自动备份原文件 → 应用 → 可一键回滚。

---

## 2. 技术选型

| 层 | 选型 | 理由 |
|----|------|------|
| 后端框架 | **FastAPI**（现有 `web/server.py` 已用）+ WebSocket | 与现有代码一致，原生支持 async 与 WS |
| 前端 | **Vue 3 (CDN) + 原生 JS**，无需构建工具 | 项目无 Node 环境，避免引入构建链；后续可升级 Vite |
| 3D 可视化 | **CesiumJS 1.120**（现有）+ **Ion Token 动态配置** | 已集成，仅需替换底图源 |
| 图表 | **ECharts / Plotly**（现有输出用 Plotly HTML） | 准确率曲线、热力图、对比图 |
| 实时通信 | **WebSocket**（uvicorn[standard] 已装 `websockets`） | 训练进度/日志/内存占用实时推送 |
| 存档 | `pickle`（模型/状态）+ JSON（元数据） | 简单可靠，模型权重序列化直接复用 torch |
| 任务执行 | `asyncio` + 线程池（`run_in_executor`） | FL 训练为 CPU 密集，线程池执行 + 事件回传 |
| AI 接入 | **OpenAI 兼容 API / Ollama 本地**，Provider 抽象 | 支持离线（Ollama）与云端两种模式 |
| 文档渲染 | 前端 `marked`(MD) + `pdf.js`(PDF)；后端解析目录 | 零后端依赖 |
| 配置 | Pydantic v2（现有 `config/schemas.py`） | 已具备强类型校验，直接复用 |

---

## 3. 模块与 API 设计

### 3.1 目录规划（新增）

```
fl_space/
├── web/
│   ├── server.py              # 现有: 3D 数据 API（保留）
│   ├── index.html             # 现有: Cesium 页（改为工作台入口）
│   └── (新增) ...
web_platform/                  # 新增后端包（pip 无需安装，通过 sys.path 或包注册）
├── __init__.py
├── main.py                    # FastAPI 汇总路由（挂载全部子路由）
├── session_api.py             # ① 调参服务: 读写 .fls_session.json
├── experiment_api.py          # ② 实验管理: 启动/暂停/恢复/结果
├── checkpoint.py              # ③ 存档引擎
├── ai/
│   ├── provider.py            # AI Provider 抽象（OpenAI/Ollama）
│   ├── code_assistant.py      # 代码修改: diff 生成/应用/回滚
│   ├── experiment_analyst.py  # 实验解读
│   └── docs_qa.py             # 文献/文档问答（简单检索）
├── workbench.py               # ④ 算法导入: AST 分析 + 冲突检测 + 注册
├── library_api.py             # ⑤ 文献库/文档
└── presets_api.py             # 预设列表/校验
dashboard/                     # 前端页面（静态文件，由 FastAPI 托管）
├── index.html                 # 工作台 SPA
├── app.js / app.css
└── views/                     # 各功能页（若需分页）
data/
├── experiments/               # 实验目录（配置+日志+检查点+结果）
└── library_index.json         # 文献索引
```

### 3.2 ① 调参服务（需求①）— session_api.py

**核心思路**：`fls tune`/`fls mount` 本质是改 `.fls_session.json`。Web 端直接操作同一文件，天然双向同步。

```
GET  /api/session                 → 完整 session（tune+mount 分组、每项带 schema 元数据）
PUT  /api/session/tune/{key}      → 改超参（校验规则复用 cli.py 的 rules）
PUT  /api/session/mount/{key}     → 改挂载项（枚举校验: algo/isl/backend/...）
GET  /api/session/schema          → 参数元数据（类型/范围/枚举/描述），前端自动渲染表单
POST /api/session/reset           → 恢复默认
GET  /api/presets                 → 算法/规模/数据集/实验预设列表（复用 fl.config）
```

**参数 schema 元数据**（前端表单驱动源，从 CLI 校验规则 + 预设自动生成）：

```json
{
  "tune": {
    "lr":        {"type": "float", "min": 0, "step": 0.001, "desc": "学习率"},
    "rounds":    {"type": "int",   "min": 1, "step": 1,     "desc": "训练轮次"},
    "dataset":   {"type": "enum",  "values": ["mnist","fashion_mnist","cifar10","femnist"], "desc": "数据集"},
    "non_iid":   {"type": "bool",  "desc": "non-IID 数据切分"},
    ...
  },
  "mount": {
    "algo":      {"type": "enum", "values": ["fedavg","fedprox","fedbuff"], "desc": "FL 算法"},
    "isl":       {"type": "enum", "values": ["disabled","wgs84"], "desc": "ISL 计算器"},
    "time_model":{"type": "enum", "values": ["slot","physics"], "desc": "时间模型"},
    "backend":   {"type": "enum", "values": ["kepler","skyfield"], "desc": "轨道后端"},
    ...
  }
}
```

**页面**：左侧参数分类导航（训练超参 / 算法组件 / 轨道环境 / ISL / 时间模型），右侧表单；「保存到终端 session」「从终端刷新」两个按钮实现双向同步；所有修改即时写盘，终端 `fls run` 直接用新参数。

### 3.3 ② 实验管理（需求⑤⑥）— experiment_api.py + checkpoint.py

**状态机**：`PENDING → RUNNING → PAUSED → (RESUMED) → DONE / FAILED / ABORTED`

```
POST /api/experiments                       → 新建实验（name + 配置快照 JSON）
POST /api/experiments/{id}/run              → 启动（后台线程，立即返回）
POST /api/experiments/{id}/pause            → 暂停（请求中断 → 存档）
POST /api/experiments/{id}/resume           → 从检查点续跑
POST /api/experiments/{id}/abort            → 终止
GET  /api/experiments/{id}                  → 详情（状态/进度/配置/检查点列表）
GET  /api/experiments/{id}/checkpoints      → 检查点列表（含时间/轮次/指标）
POST /api/experiments/{id}/save-checkpoint  → 手动存档
POST /api/experiments/{id}/load             → 载入指定检查点并续跑
GET  /api/experiments/{id}/results          → 结果（history JSON + 图表数据）
GET  /api/experiments/{id}/outputs          → 产物文件列表（HTML/PNG/JSON）
DELETE /api/experiments/{id}                → 删除
GET  /api/experiments                       → 列表（卡片视图，可对比）
GET  /api/experiments/compare?ids=a,b       → 多实验准确率对比曲线

WS   /api/ws/experiments/{id}               → 实时事件流
```

**WebSocket 事件类型**：
```json
{"type": "log",       "ts": "...", "message": "Round 5/20 | acc=0.6224"}
{"type": "round",     "round": 5,  "total": 20, "accuracy": 0.6224, "loss": 0.9, "timeslot": 14}
{"type": "metric",    "key": "contact_rate", "value": 0.211}
{"type": "status",    "status": "PAUSED", "reason": "user_request"}
{"type": "checkpoint","path": "...", "round": 7}
{"type": "done",      "output_dir": "output/exp_001"}
{"type": "error",     "message": "..."}
```

**存档机制（核心设计）**：

```
experiments/{id}/
├── config.json            # 实验配置快照（不可变，含算法/数据/轨道/seed）
├── run.log                # 全量终端输出流
├── checkpoint_round_5.pkl # pickle: FLRunner.server 状态（模型权重/轮次/历史/调度器/时间模型）
├── checkpoint_round_5.json# 元数据: 轮次/时间/当前指标/依赖版本
├── results.json           # 最终 history + 汇总指标
└── outputs/               # 可视化产物（复用现有 output/ 逻辑）
```

- **暂停实现**：设置 `runner.request_pause=True`，`FLRunner` 在**每轮结束的检查点回调**处响应——先落盘检查点再抛 `PauseRequested` 异常，外层捕获后标记 `PAUSED`。无需改算法内部代码，只在外层加回调钩子。
- **恢复实现**：反序列化 checkpoint → 重建 `FLServer`/`FLRunner` → 从 `start_round` 继续，`history` 完整续接。
- **断点保护**：任何异常（断电/崩溃）至少保留最近检查点，恢复后提示"从第 N 轮续跑"。
- **类文字游戏体验**：实验卡片右上角「存档」按钮 + 自动检查点（每 N 轮）+ 可命名存档点（如"第 5 轮 tuned-mu=0.05"），恢复界面显示存档时间线。

### 3.4 ③ 地图源接入（需求②）

**Cesium Ion Token 配置**：用户提供的 Ion Token 配置到 `web_platform/config.json`（或环境变量 `CESIUM_ION_TOKEN`），前端启动时从 `/api/settings` 拉取，动态设置 `Cesium.Ion.defaultAccessToken`。

```
GET /api/settings                 → {cesium_ion_token, ai_provider, ai_model, ...}
```

**可插拔底图源**（前端下拉切换，无需后端）：

| 底图源 | 接入方式 |
|--------|---------|
| Cesium Ion 影像（World Imagery） | `Cesium.IonImageryProvider.fromAssetId(2)` — 需 Ion Token |
| Cesium Ion 地形（World Terrain） | `createWorldTerrainAsync()` — 需 Ion Token |
| NaturalEarthII（现有离线） | `TileMapServiceImageryProvider` 保留为回退 |
| OSM / ArcGIS / 自建 WMTS | `UrlTemplateImageryProvider` / `ArcGisMapServerImageryProvider` |

**页面交互**：3D 视图保留现有「时隙滑块 + 播放/暂停 + 加载」控件，新增：地图源下拉、卫星轨迹线开关、ISL 链路开关、地面站标签、轨道面着色、点击卫星查看状态卡片（当前可见站/最近接触/训练进度映射）。

### 3.5 ④ AI 辅助（需求③）— ai/

**Provider 抽象**（支持离线/在线切换）：

```python
class AIProvider(Protocol):
    async def chat(self, messages, stream=False) -> ...: ...
    async def complete(self, system, user) -> str: ...

class OllamaProvider(AIProvider):   # http://localhost:11434, 模型如 qwen2.5-coder
class OpenAICompatProvider(AIProvider):  # OpenAI/DeepSeek/通义等兼容 API
```

```
POST /api/ai/chat          → 对话流（SSE/WS），上下文含当前 session 参数 + 实验状态
POST /api/ai/analyze       → 解读实验：传入 results.json + 配置，返回结论/异常提示/优化建议
POST /api/ai/ask-doc       → 基于选中文档/文献的问答
POST /api/ai/code/request  → 提出修改请求（如"给 scheduler.py 加带宽上限"）
POST /api/ai/code/diff     → 返回修改方案 diff（不落盘）
POST /api/ai/code/apply    → 确认应用（自动备份 → 写盘 → 记录变更日志）
POST /api/ai/code/revert   → 按变更记录回滚
GET  /api/ai/change-log    → 变更历史（文件/时间/摘要/可回滚）
```

**代码修改安全流程（关键设计）**：
1. AI 生成修改方案 + 完整 diff（基于当前文件内容）。
2. 后端**静态预检**：`ast.parse` 语法正确性 + 导入是否存在 + 是否触碰 `fl_space/` 受保护文件（禁止直接改核心，除非用户勾选"高级模式"）。
3. 前端展示 diff 面板（绿/红高亮），用户审阅。
4. 应用时：`cp file file.bak.<timestamp>` → 写入 → 追加 `change_log.json`。
5. 任何一步失败自动回滚；「一键回滚」从备份恢复。
6. AI 建议基于**当前真实代码**（后端注入相关文件内容片段），不做记忆性幻觉。

### 3.6 ⑤ 文档与文献库（需求④⑦）— library_api.py

```
GET  /api/library                 → 文献列表（元数据 + 过滤条件）
GET  /api/library/{id}            → 详情（标签/摘要/PDF 路径）
POST /api/library/scan            → 扫描 文献/ 目录重建索引（提取 PDF 标题/文件名）
POST /api/library/{id}/tags       → 打标签
GET  /api/library/search?q=...    → 全文/文件名/标签搜索
GET  /api/docs/tree               → docs/ 与 fl_space/ 代码文档目录树
GET  /api/docs/content?path=...   → 读取 Markdown/代码文件（限制大小）
GET  /api/docs/pdf?path=...       → 返回 PDF 流（前端 pdf.js 渲染）
```

**文献索引设计**：
```json
{
  "id": "[22]_Zhai2024_FedLEO_中文翻译.md",
  "title": "FedLEO: 卸载辅助的去中心化联邦学习 (Zhai 2024)",
  "type": "md" | "pdf",
  "path": "文献/[22]_Zhai2024_FedLEO_中文翻译.md",
  "tags": ["fedleo", "decentralized", "LEO"],
  "source_paper": "FedLEO (arXiv 2024)",
  "related_algorithms": ["fedleo"],
  "added_at": "2026-08-10"
}
```
- `related_algorithms` 字段与算法注册表联动：在工作台查看某算法时，自动列出相关文献。
- 阅读器分栏：左侧目录树（docs/ + 文献/），右侧渲染（Markdown 用 `marked`，PDF 用 `pdf.js`，代码用语法高亮）。
- 支持「AI 阅读」：选中段落 → 右键 → 让 AI 解释/翻译/总结。

### 3.7 ⑥ 算法导入工作台（需求⑧）— workbench.py

**目标**：用户导入新的 FL 算法或优化（单个 `.py` 或打包 ZIP），平台自动分析其结构、检测与现有代码的冲突，验证通过后注册为可选算法。

```
POST /api/workbench/upload          → 上传文件/ZIP（multipart）
POST /api/workbench/{job}/analyze   → 异步分析（AST + 依赖扫描 + 接口匹配）
GET  /api/workbench/{job}/report    → 分析报告（结构/接口/冲突/风险 评分）
POST /api/workbench/{job}/activate  → 激活注册（写入算法注册表 + 预设）
POST /api/workbench/{job}/test-run  → 隔离冒烟测试（小数据集跑 1-2 轮）
GET  /api/workbench/algorithms      → 已注册算法列表（内置+导入）
DELETE /api/workbench/algorithms/{name} → 停用导入算法
```

**分析流程（静态 + 动态两级）**：

1. **结构识别（AST 分析）**：
   - 识别算法类入口（如 `class MyAlgo(...)` 是否继承/匹配 `LocalTrainer`/`Aggregator` 协议）
   - 提取参数清单（`__init__` 签名）→ 自动生成调参表单
   - 扫描 import：第三方包（记录进依赖清单，缺失则警告）
2. **冲突检测（静态）**：
   - 符号冲突：导入的模块名/类名/函数名是否与 `fl_space.fl.*`、`fl_space.fedleo.*` 现有符号冲突
   - 路径冲突：文件落点是否覆盖已有文件（只允许写入 `plugins/` 目录）
   - 受保护文件检查：禁止修改 `fl_space/` 核心（除非高级模式）
3. **语义匹配（动态）**：
   - 将导入类接入最小 FL 编排器，跑隔离冒烟测试（2 卫星 × 1 轮 × 极小数据）
   - 验证接口签名兼容（如 `train()` 返回 `ClientUpdate` 结构）
   - 若失败，报告具体异常栈，标红「未通过」并给出修复提示
4. **注册**：通过后写入 `plugins/registry.json`（插件注册表），CLI `fls mount algo` 枚举自动包含新算法；Web 工作台出现该算法的调参表单。

**插件目录隔离**：所有导入算法放 `plugins/`，通过 `plugins/loader.py` 动态 `importlib` 加载，**不修改** `fl_space/fl/algorithms/` 原始目录——保证官方代码与插件互不干扰，且卸载插件即删除目录+注册条目。

---

## 4. 前端页面结构（dashboard/index.html）

```
┌────────────────────────────────────────────────────────────────┐
│ 顶栏: [🛰 SpaceFL 工作台] [调参台] [3D视图] [实验] [文献] [AI助手] [算法工作台] │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  页面 1「调参台」:                                              │
│    左: 参数分类树(训练/算法/轨道/ISL/时间)  右: 动态表单        │
│    [保存到终端 session] [从终端刷新] [预设快速填充]              │
│                                                                │
│  页面 2「3D 视图」(现有 Cesium 页升级):                         │
│    底图源下拉 · 卫星/链路/地面站开关 · 时隙时间轴 · 点击交互      │
│    右侧联动面板: 当前实验实时状态(来自 WS)                      │
│                                                                │
│  页面 3「实验管理」:                                            │
│    实验卡片网格(状态徽章/进度条/操作按钮)                       │
│    运行日志实时流 · 检查点时间线(可载入) · 结果图表对比           │
│                                                                │
│  页面 4「文献/文档」:                                           │
│    左: 目录树(文献/ + docs/ + 代码)  右: 阅读器(MD/PDF/代码)    │
│    标签过滤 · 全文搜索 · 选中段落"AI 解释"                      │
│                                                                │
│  页面 5「AI 助手」:                                             │
│    左: 会话历史  右: 对话区(可切换 代码模式/解读模式/文档模式)   │
│    diff 审阅面板(应用/回滚) · 变更日志                          │
│                                                                │
│  页面 6「算法工作台」:                                          │
│    拖拽上传区 · 分析报告卡片(结构/接口/冲突/评分)               │
│    激活/冒烟测试按钮 · 已注册算法列表                          │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. 数据模型汇总

```python
# experiment
class Experiment(BaseModel):
    id: str                       # exp_<timestamp>_<rand>
    name: str
    status: Literal["PENDING","RUNNING","PAUSED","DONE","FAILED","ABORTED"]
    config: dict                  # 完整配置快照（tune+mount）
    created_at: str
    current_round: int = 0
    total_rounds: int = 0
    checkpoint_dir: str
    result_summary: dict | None   # final accuracy, elapsed, contact_rate...

class CheckpointMeta(BaseModel):
    path: str
    round: int
    created_at: str
    label: str | None             # 用户命名存档
    metrics: dict                 # 当前 accuracy/loss

# workbench
class AlgorithmAnalysis(BaseModel):
    name: str
    entry_point: str              # 模块路径
    params: list[ParamSpec]       # 自动生成调参表单
    imports: list[str]
    conflicts: list[ConflictIssue]  # severity: info/warning/error
    interface_matches: bool
    smoke_test: SmokeResult | None
    score: float                  # 0-100

class ChangeRecord(BaseModel):
    id: str
    file: str
    ts: str
    summary: str
    backup_path: str              # 可回滚
    applied: bool
```

---

## 6. 实施计划（分阶段）

| 阶段 | 内容 | 交付物 | 预估 |
|------|------|--------|------|
| **P1 基础平台** | 后端 `main.py` 挂载 + `session_api` + 前端工作台壳（调参台）+ WS 事件管道 | 网页可调参、终端联动、实时日志 | 核心 |
| **P2 实验管理+存档** | `experiment_api` + `checkpoint.py` + 暂停/恢复/检查点 UI + 结果对比 | 类文字游戏存档体验 | 核心 |
| **P3 3D 视图升级** | Cesium Ion Token 接入、多地图源、ISL/轨迹/点击交互、与运行实验联动 | 真实底图 3D 面板 | 核心 |
| **P4 文献+文档** | `library_api` + 目录树 + MD/PDF 阅读器 + 标签搜索 | 文献阅读中心 | 扩展 |
| **P5 AI 助手** | Provider + 对话 + 代码修改安全流程 + 实验解读 + 文献问答 | 本地 AI 协同 | 扩展 |
| **P6 算法工作台** | AST 分析 + 冲突检测 + 插件注册 + 冒烟测试 | 可扩展生态 | 扩展 |

> P1-P3 构成「可视化 + 实时运行」主闭环（需求①②⑤⑥）；P4-P6 为增强能力（需求③④⑦⑧）。建议按序推进，每阶段可独立验收。

---

## 7. 风险与对策

| 风险 | 对策 |
|------|------|
| 训练线程与 WS 并发冲突 | 实验运行使用独立线程 + `asyncio.Queue` 桥接，事件带 `experiment_id` 路由 |
| pickle 存档跨版本不兼容 | 存档 JSON 记录 Python/torch/依赖版本；载入时版本检查并警告 |
| AI 修改破坏代码 | 只读核心保护 + diff 审阅 + 自动备份 + 一键回滚 + 变更日志（详见 3.5） |
| Cesium Token 泄露 | Token 存服务端配置，经 `/api/settings` 下发前端；README 提示勿提交 |
| 导入算法恶意/冲突 | 插件目录隔离 + AST 预检 + 受保护文件名单 + 冒烟测试沙箱 |
| 前端无构建环境 | 使用 CDN Vue 3 + 原生 JS，零构建；如后续复杂可迁移 Vite |

---

## 8. 与现有系统的对接清单（实现时的关键钩子）

1. **调参**：复用 `cli.py` 的 `load_session/save_session/_tune_set` 校验规则 → 提取为可复用模块或直接调用。
2. **运行**：`cmd_run_train`/`cmd_run_fedleo` 底层函数（`run_fedleo_vs_baseline` 等）改为**可编程调用**（返回 runner/server 引用而非只打印），供实验管理器嵌入。
3. **检查点**：在 `FLRunner.run()` 外层包一层「轮次回调」——检查现有 `runner.py` 是否已有每轮 hook，若无则新增一个可选 `on_round_end` 回调参数（向后兼容，默认 None）。
4. **算法注册**：参考现有 `register_model` 模式，建立 `plugins/` 加载器与 `registry.json`。
5. **3D 数据**：直接复用 `web/server.py` 的 `build_orbit_data()`；新增「运行中实验映射」把训练状态叠加到轨道视图。
6. **文献**：`文献/` 目录现有 README_文献阅读指南.md 可作为索引格式参考。
