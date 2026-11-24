# 科研决策模型与 LLM 协作接口

状态：已实现独立决策接口、Jev 适配器、离线规则、有界只读证据工作流、
现有文献问答 API 的显式入口和离线回归测试。尚未给原有问答全部启用，
未接入实验执行、计划修正或前端证据面板。本次没有调用计费 Jev 请求。

## 架构决策（已采用）

选择独立 `DecisionProvider`，而不是将全部行动选择塞进 LLM 的自由文本。
独立接口使概率校验、供应商替换、全局开关和规则基线可以单独测试，代价是
增加一次潜在网络调用。第一版仅适配文献证据整理，不让新模型获得执行权限。

```text
用户问题
  → 代码限定候选动作 retrieve / compose / stop
  → 决策模型建议（失败则切换规则）
  → 本地检索 → 指定文档/页原文子串核验 → 证据账本
  → LLM 选择 evidence_id 与原文摘录（可选）
  → schema / ID / 原文一致性校验
  → 代码生成带引用的摘录答复 + verification + trace
```

决策建议不是实验授权，也不是事实验证结果。Jev 只能在当前代码给出的候选
集合中选择；没有证据时不会获得 compose；没有工具允许训练、修改配置或
发消息。即使模型给出“run_experiment”，也会拒绝并使用离线规则。

## 模块与契约

- `fl_space/research/decision.py`：DecisionState、DecisionAdvice、DecisionProvider、
  OfflineDecisionProvider、JevDecisionProvider。新供应商继承 advise(state)，
  返回 DecisionAdvice；调用方必须 validate(state)。公开数据类支持字典往返。
- `fl_space/research/evidence_workflow.py`：有界编排、EvidenceItem、LLM 写作适配。
- `ResearchService.research_evidence`：接入现有 LiteratureIndex 与已配置 LLM，
  可用构造参数 decision_provider 注入模型，便于本地供应商或测试替换。

决策输入仅为 evidence_count、rejected_count、retrievals、steps_remaining、
candidates。不会发给 Jev 用户问题、文献正文、实验原始数据、文件路径或密钥。
这意味着第一版适合决定工作流下一步，不适合判断论文论证强弱。

决策概率集合必须与候选集合相同，值必须有限、位于 0–1，和在 1±0.02 内。
trace 中标为 uncalibrated_action_preference_not_truth，不解释为科研结论正确率。

## 使用方式

### 本地离线运行

```powershell
.\venv\Scripts\python.exe scripts/run_evidence_research.py --question "卫星通信中的同频干扰如何影响SINR" --output output/research-evidence.json
```

此入口默认离线决策且不调用 LLM；重建本地文献索引缓存，再保存 JSON。
加 `--llm` 才允许当前 LLM 供应商选择摘录，加 `--decision jev` 才允许尝试 Jev。
两者可单独启用，也可组合；尊重 SPACEFL_OFFLINE_ONLY，永不自动打开全局 Jev。

### 现有文献问答 API

```http
POST /api/research/library/ask
Content-Type: application/json

{"question":"卫星通信中的同频干扰如何影响SINR", "evidence_workflow":true}
```

不提供 evidence_workflow 时保持现有问答行为。该字段必须是 JSON 布尔值，
网页不能传 endpoint、API key 或供应商名称来选择外部服务。

本地服务端如需使用 Jev，显式设置 `SPACEFL_RESEARCH_DECISION=jev`；仍须当前
Windows 用户的 `JEV_ENABLED` 已打开且配置 JEV_API_KEY。开关每次实时读取，
不自动调用 jev.ps1 on。不在源码、配置、日志或输出保存密钥。
公网 demo 始终使用离线决策，即使服务端设置了该环境变量；LLM 仍受现有
公网开关控制。原有 HTTP 权限与限流路径不变。

Jev 适配器使用已安装本机集成的 HTTPS 协议：
`https://api.defapi.org/api/v1/decisions`，模型 `typesafe/jev-1.13`。
这是辅助决策供应商，会向 defapi.org 发送最小状态并产生费用。默认请求超时
10 秒、可配置上限 30 秒、响应上限 64 KiB；无重定向到自定义 endpoint 的配置。
一次失败后本次工作流全部转离线，不自动重复请求未知计费结果。

## 可验证范围与限额

- 最多 5 步、2 次检索、1 次 LLM 写作。每步至多一个决策请求，不存在无限反思。
  两次查询由代码固定构造。LLM 超时最多 15 秒（供应商内部协议兼容回退可能
  发生额外请求，因此不是整个工作流墙钟时间 SLA）。自定义适配器应自行实现
  传输超时；纯 Python 接口不能强杀任意阻塞插件。
- 证据最多 12 条，总正文 12,000 字符，单条最多 6000 字符。writer 最多选择
  5 条，最终输出必须是已校验原文的逐字子串，不能新增数字、页码或解释。
- 原文先按 doc_id/page 查索引，规范化空白后做子串匹配；错误页码、不存在的
  引文和空引文均拒绝。索引快照与 PDF 提取质量仍限制来源核验的可靠性。
- verification.citation=pass 只代表出处匹配；numeric=source_copy_only 代表
  数字来自摘录，未判断单位、口径或原论文的计算是否正确。
- semantic_support=not_assessed。coverage 是本次输出摘录的出处覆盖率，
  不是问题回答完整度，不是事实正确率。没有来源时 coverage=0 并停止生成结论。
- LLM 越界或失败时只返回已核验摘录；外部异常统一脱敏，不回显响应正文。
- 原文包含恶意指令时只作为引用数据，工具集合不会改变。摘录本身仍可能包含
  这些文本，因此后续消费者也必须维持“检索内容是数据”的边界。

现有聊天、计划评审、档案解读不会因此被宣称全部完成幻觉治理。本阶段的
claims 是 source_excerpt，不是自动验证过的 scientific_claim。

## 验证与后续实验

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_research_decision.py tests/test_research_service.py tests/test_public_llm.py -q
```

覆盖来源核验、虚假页码/数字、未知动作、无证据停止、注入文本、全局关闭不
调用网络、Jev 协议/概率、超时不重试、凭据不泄露、现有服务与 HTTP 显式入口。
测试的在线供应商均为替身，没有真实费用。它们是工程回归，不是已标注的
50 条科研事实评测集，也没有证明 Jev 降低了幻觉率。

后续可用相同文献快照和任务集比较 offline 与 Jev 决策，记录无效检索次数、
最终出处覆盖、拒答正确性、语义支持错误、延迟和真实账单。仅在测得增益后
扩大启用范围。计划语义校验、档案指标计算应继续由确定性代码单独实现。
