# SpaceFL 科研智能平台 v1 — 接口契约与模块规格

> 目标：把现有 SpaceFL 可视化实验平台升级为面向科研的软件：内置期刊文献知识库、
> 主动抓取前沿资讯、以自然语言驱动实验/批量实验、并由研究智能体分析实验计划与结果。
>
> 本文件是 v1 的「接口契约」，前后端与各模块以本文件为准。

## 1. 总体结构

```
web/
  index.html            # 新版外壳（顶部导航 + Hero + 9 个视图）
  assets/app.js         # 既有视图逻辑（总览/配置/星簇/学习任务/档案/轨道）
  assets/research.js    # 新视图逻辑（研究助手 / 文献库 / 前沿情报 / 研究目标）
  assets/app.css        # 既有组件主题（太空官方风格）
  assets/research.css   # 新组件样式（依赖 app.css 的 CSS 变量）
  server.py             # FastAPI：既有路由 + /api/research/* 路由注册
fl_space/research/
  __init__.py
  config.py             # 研究目标 / 主题 / AI 供应商配置持久化
  providers.py          # LLM 供应商抽象（OpenAI 兼容 / Responses / Anthropic / DeepSeek）
  literature.py         # 文献扫描、PDF 文本抽取、分块、BM25 全文检索、RAG 上下文
  intel.py              # arXiv / RSS 前沿资讯抓取、缓存、入库
  planner.py            # 自然语言 → 结构化实验计划（含离线规则兜底）
  analyzer.py           # 实验结果分析 / 计划评审 / 文献精读（含离线兜底）
  service.py            # ResearchService：编排以上模块，供 web/server.py 调用
scripts/run_research_plan.py   # 后台批量实验执行器（子进程）
```

## 2. CSS 变量契约（app.css / research.css 共用）

```css
--bg: #000;            /* 页面底色 */
--panel: #0a0a0a;      /* 面板底色 */
--panel-2: #101010;
--line: #262626;       /* 1px 分割线 */
--text: #fff;
--muted: #9b9b9b;
--accent: #fff;        /* 主强调（SpaceX 式纯白）*/
--green / --cyan / --amber / --red   /* 语义色，保持克制 */
--space-blue: #5e8fb5; /* 少量科技蓝点缀 */
```

## 3. 后端 API 契约

所有接口前缀 `/api/research`，返回 JSON。错误使用 FastAPI HTTPException。

### 3.1 概览与配置

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/research/overview` | 研究目标、文献统计、情报统计、最近实验、AI 状态 |
| GET | `/api/research/config` | 完整配置 |
| PUT | `/api/research/config` | 保存 `{goal?, topics?, ai?:{provider,model,base_url,api_key,wire_api,source}}` |

`GET /api/research/config` 响应：

```json
{
  "goal": "研究星地链路受限条件下的联邦学习聚合与数据卸载策略",
  "topics": ["太空联邦学习", "LEO 星座", "数据卸载", "通信调度"],
  "ai": {"configured": true, "provider": "DeepSeek", "model": "deepseek-chat",
         "base_url": "https://api.deepseek.com", "wire_api": "chat", "source": "environment"},
  "literature": {"root": "D:\\fl_space-master\\文献", "indexed": 12, "parsed": 10},
  "updated_at": "2026-11-05T12:00:00+00:00"
}
```

### 3.2 文献库

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/research/library` | `{items, stats}` 全部索引条目 |
| POST | `/api/research/library/scan` | 重建索引（增量），返回 stats |
| GET | `/api/research/library/search?q=&limit=` | 全文检索，每条带 `snippets` |
| GET | `/api/research/library/{doc_id}` | 单篇元数据 + `pages: [{page, text}]` |
| POST | `/api/research/library/{doc_id}/digest` | AI 精读，返回 digest |
| POST | `/api/research/library/ask` | `{question}` → `{answer, citations, mode}` |

`LiteratureItem`：

```json
{"id": "lib_ab12cd34ef", "title": "FedLEO: An Offloading-Assisted ...",
 "authors": ["Zhiwei Zhai"], "year": 2024, "venue": "IEEE Transactions on Mobile Computing",
 "type": "pdf", "source": "默认文献", "size": 1234567, "pages": 20, "chunks": 42,
 "abstract": "...", "keywords": ["federated learning", "LEO"], "tags": ["fedleo", "satellite"],
 "extracted": true, "extract_note": "pypdf", "modified_at": "2026-10-01T00:00:00+00:00"}
```

`Digest`：

```json
{"doc_id": "...", "mode": "offline|llm", "provider": "DeepSeek",
 "summary": "...", "contributions": ["..."], "method": "...", "findings": ["..."],
 "limitations": ["..."], "quotes": [{"page": 3, "text": "..."}],
 "relevance": {"score": 82, "reason": "...", "goal": "..."}}
```

### 3.3 前沿资讯

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/research/intel?refresh=0` | 缓存优先；`refresh=1` 强制拉取 |
| POST | `/api/research/intel/refresh` | 强制刷新 |
| POST | `/api/research/intel/ingest` | `{item_id}` → 下载/归档到文献目录并重建索引 |

`IntelFeed`：

```json
{"updated_at": "...", "refresh_interval_sec": 1800,
 "sources": [{"name": "arXiv 太空联邦学习", "status": "ok", "count": 12}],
 "items": [{"id": "arxiv_2511.14889", "source": "arXiv", "kind": "paper",
            "title": "...", "url": "https://arxiv.org/abs/2511.14889",
            "published": "2026-11-01T00:00:00+00:00", "summary": "...",
            "authors": ["..."], "tags": ["federated learning", "satellite"],
            "pdf_url": "https://arxiv.org/pdf/2511.14889", "ingested": false}]}
```

### 3.4 研究智能体

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/research/agent/plan` | `{instruction, goal?}` → `{reply, steps, plan, citations, mode}` |
| POST | `/api/research/agent/run` | `{plan}` → `{job_id, status}` 启动后台批量实验 |
| GET | `/api/research/agent/jobs` | 任务列表 |
| GET | `/api/research/agent/jobs/{job_id}` | 任务详情（进度、逐条件结果、日志尾部） |
| POST | `/api/research/agent/analyze` | `{job_id? experiment_id? question?}` → `{content, metrics, citations, mode}` |
| POST | `/api/research/agent/chat` | `{message, experiment_id?}` → `{reply, steps, citations, mode}` |

`Plan`：

```json
{"id": "plan_20261105_120000", "title": "...", "goal": "...", "instruction": "用户原话",
 "backend": "fedleo", "hypothesis": "...",
 "independent_variables": [{"name": "sats_per_plane", "label": "每轨面卫星数", "values": [2, 3]}],
 "conditions": [{"id": "c1", "name": "3 轨面 × 2 星",
                 "params": {"planes": 3, "sats_per_plane": 2, "gs": 3, "rounds": 3,
                            "seed": 42, "offload_every": 2, "max_samples_per_client": 120}}],
 "controls": ["相同随机种子", "相同本地 epoch", "相同数据划分策略"],
 "metrics": ["final_accuracy", "peak_accuracy", "total_offloaded", "total_delay_slots"],
 "expected_duration_sec": 300, "risks": ["..."],
 "literature": [{"id": "lib_...", "title": "...", "why": "提供了卸载‑时延指标定义"}],
 "created_at": "..."}
```

`AnalysisResult`：`{"content": "markdown", "metrics": {...}, "citations": [{"index":1,"doc_id":"lib_...","title":"...","quote":"..."}], "mode": "offline|llm", "provider": "..."}`

### 3.5 后台实验任务

`ResearchJob`：

```json
{"id": "rjob_ab12cd", "plan_id": "plan_...", "plan_title": "...",
 "status": "PENDING|RUNNING|DONE|FAILED|CANCELLED", "progress": 0.33,
 "message": "正在运行 条件 2/3：3 轨面 × 3 星", "started_at": "...", "finished_at": null,
 "conditions": [{"id": "c1", "name": "...", "status": "DONE",
                 "final_accuracy": 0.83, "peak_accuracy": 0.84,
                 "elapsed_sec": 96.2, "output": "research_results/rjob_ab12cd/c1"}],
 "result_path": "research_results/rjob_ab12cd", "log_tail": "..."}
```

执行器 `scripts/run_research_plan.py`：

```
python scripts/run_research_plan.py --plan <plan.json> --output research_results/<job_id>
```

- 每个 condition 调用 `fl_space/fedleo/experiment.py` 的 `run_fedleo_vs_baseline(...)`；
- 每完成一个条件写 `progress.json`（`{progress, current, done, conditions, logs}`）；
- 全部完成写 `research_result.json`（plan + conditions + 汇总指标）；
- 失败时写 `error` 字段并以非零码退出。

## 4. 离线兜底原则

比赛现场可能无外网 / 无大模型 Key，因此所有 AI 能力必须有两套实现：

1. **LLM 模式**：配置了 API Key 时，调用兼容 OpenAI 的 Chat Completions / Responses。
2. **离线规则模式**：无 Key 或调用失败时，使用确定性规则生成：
   - 规划器：中文关键词 + 数字单位解析 → 计划模板；
   - 分析器：基于曲线统计（首末/峰值/收敛轮次/稳定性/卸载量）生成结构化报告；
   - 文献精读：抽取摘要、贡献句、局限句、与目标的相关度；
   - 知识库问答：BM25 检索片段 + 引用拼接。
   返回体中 `mode` 字段明确标注 `offline` 或 `llm`，界面必须可见。

## 5. 研究目标（goal）的贯穿

`goal` 在以下位置被使用：

- 计划生成：作为假设与变量选择的上下文；
- 文献检索：与指令拼接后做 RAG 召回；
- 结果分析：输出「目标对齐度」清单；
- 情报抓取：作为 arXiv/RSS 主题词来源；
- 概览页：展示当前研究主线。

## 6. v1 实现状态（2026-11-05）

| 模块 | 文件 | 状态 |
|------|------|------|
| 研究配置 | `fl_space/research/config.py` | ✅ 研究目标 / 订阅主题持久化 |
| 双语检索扩展 | `fl_space/research/bilingual.py` | ✅ 中文查询 → 英文领域词，提升英文论文召回 |
| 供应商抽象 | `fl_space/research/providers.py` | ✅ chat / responses / anthropic，Key 脱敏与失败降级 |
| 文献知识库 | `fl_space/research/literature.py` | ✅ PDF 抽取、增量索引、BM25、页码引用 |
| 前沿情报 | `fl_space/research/intel.py` | ✅ arXiv + RSS 抓取、TTL 缓存、归档入库 |
| 实验规划 | `fl_space/research/planner.py` | ✅ 自然语言解析 + LLM 生成 + 结构校验 |
| 分析器 | `fl_space/research/analyzer.py` | ✅ 计划评审 / 结果解读 / 文献精读 / 情报简报 |
| 编排层 | `fl_space/research/service.py` | ✅ 后台任务、进度同步、缓存与相关度重排 |
| API | `web/research_routes.py` | ✅ 22 个 `/api/research/*` 路由 |
| 执行器 | `scripts/run_research_plan.py` | ✅ 逐条件执行、原子写进度、失败隔离、dry-run |
| 前端 | `web/index.html` / `research.js` / `research.css` / `app.css` | ✅ 研究助手 / 文献库 / 前沿情报 + 太空官方风格主题 |
| 测试 | `tests/test_research_*.py` | ✅ 133 项离线测试；全仓 192 项测试通过 |

### v1 已知边界

1. 批量执行器当前只实现 `backend = "fedleo"`（FedLEO 卸载开/关对照，内含 FedAvg 基线）；
   FedProx / FedBuff 批量对照在规划阶段会写入 `notes` 说明，请在「实验配置」页单跑。
2. 文献全文检索为 BM25 + 领域词典扩展，未使用向量模型；跨语言（中文问、英文论文）依赖双语词典。
3. PDF 解析依赖 `pypdf`（`pip install -e ".[web]"`）；缺失时只索引元数据并标记 `extracted=false`。
4. 大模型输出统一经过结构与范围校验（条件 ≤6、轮次 ≤30、卫星总数 ≤24 等），越界自动收缩并写入 `notes`。
5. 单个批量条件固定成本约 60–90 秒（数据准备 + 训练），预计耗时会在计划卡片中展示。

