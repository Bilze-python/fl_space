# Agent 代码与原文工具

更新：tool_mode 已接入 [参谋/指挥/审计三角色编排](RESEARCH_ROLES.md)。LLM 现在
提出候选，由独立决策接口选动作；旧的直接选择工具函数仅保留兼容使用。

状态：本地已实现 7 个只读工具、模型选择/参数校验、有界调用循环、HTTP 入口与
网页“代码与原文”模式。未部署公网。新能力帮助定位代码、核对实现和读取来源，
尚不自动修改代码或执行测试。

## 已注册工具

| 工具 | 输入 | 输出与用途 |
| --- | --- | --- |
| code.list_files | 路径字面量过滤、limit | 受管目录文件列表，定位模块 |
| code.search | query、path_filter、limit | 字面量匹配的文件/行号/内容，无正则执行 |
| code.read_lines | path、start_line、line_count | 带行号摘录、文件 SHA-256、截断标志 |
| code.python_outline | Python 相对路径 | AST 类/函数/导入及起止行，不运行代码 |
| code.check_syntax | Python 相对路径 | 语法有效性/错误行列，不导入或执行模块 |
| literature.search | query、limit | 本地文献 doc_id/page/quote 候选摘要 |
| literature.read_page | doc_id、page | 索引原文页与截断标志，不接受任意磁盘路径 |

公开入口 `ResearchTools` 与 `ToolSpec` 从 `fl_space.research` 导出。
工具目录提供名称、说明、参数类型、范围、默认值；未知字段/类型错误拒绝。
注册表由服务端代码构造，模型不能动态注册函数。

## 网页与 API

本地网页进入研究助手，选择“代码与原文”，输入函数名（例如 evaluate_link）
或具体问题。结果显示代码摘录和可展开的真实调用记录，源码按文本转义，
不将文件中的 HTML/JavaScript 作为页面执行。

```http
GET /api/research/agent/tools

POST /api/research/agent/chat
Content-Type: application/json

{"message":"evaluate_link", "tool_mode":true}
```

tool_mode 为严格布尔值；不传时仍是原有科研聊天。这里没有任意工具直调 HTTP
端点，用户/模型只能通过任务工作流进入允许工具。现有公网聊天限流路径复用。
公网 demo 的目录及执行器都移除 code.*，只能读取隔离的演示文献，界面模式
不授予服务器源码访问权。

## 本地离线试用

```powershell
.\venv\Scripts\python.exe scripts/run_evidence_research.py --tools --question "evaluate_link" --output output/research-tools.json
```

默认不调用在线模型：代码字面量搜索→读取首个命中附近40行；没有代码命中则
检索文献并读取第一页命中。离线策略不是自然语言规划器，优先输入明确符号。
加 `--llm` 后，已配置 LLM 可在目录中选择文件列举、搜索、AST、语法或原文页，
读取最近两次工具结果继续选择。LLM 仍受 SPACEFL_OFFLINE_ONLY/公网 LLM 开关控制。
该模式当前由 LLM 选择具体工具，Jev 的 retrieve/compose/stop 决策流保持独立，
并未把 Jev 宣称为代码理解或工具参数生成模型。

## 调用与读取边界

- 最多5次真实工具调用；相同工具/参数组合不重复执行；无效模型提议转离线。
  工具候选和参数由代码校验，不接受 shell、任意 Python、命令参数或外部URL。
- 仅允许 fl_space、web、tests、scripts、docs 下的指定文本后缀；不读取根目录
  .env、AI 配置、.git、隐藏文件或 secrets/credentials 目录。真实路径检查防止
  ../、绝对路径、符号链接或 junction 逃逸。不是整个项目任意文件浏览器。
- 单文件512 KiB；目录遍历上限2000文件并设2秒扫描检查点。达到扫描限额时
  标为 truncated，搜索结果不代表全仓库穷尽。
- 每次读取最多160行/10,000正文字符，单行最多1000字符；文献页最多12,000
  字符；搜索最多50条。AST 最多60个定义/40条导入，返回截断标志。
- 耗时为实际工具调用的 monotonic 计时，trace 包含 call_id、已验证参数、
  工具结果/错误码和 latency_ms。不是模型内部思考链。工具异常不回显原始
  本机异常信息。当前没有单独 token/费用遥测或持久任务审计库。
- Python 语法检查仅证明可解析，不证明类型正确、测试通过或程序无漏洞。
  文件 SHA-256 只标识本次读取快照，不能证明代码实现或科研结论正确。
- 输出直接组织工具原文，不允许 LLM 自由改写成“测试已通过”。所有内容仍
  标 semantic_support=not_assessed、code_executed=false、code_modified=false。
- 使用在线 LLM 时，选择工具所需的代码/文献结果会发送至当前 LLM 供应商。
  这与仅接收计数的 Jev 接口不同；离线 CLI 不发送这些内容。

此阶段选择“受管只读工具”而非开放终端：足够支持找实现、读函数、理解模块
结构，也能让调用结果可回查。代价是暂不能自动验证运行行为或应用修复。
下一阶段如加入测试执行/补丁，应单独限制测试命令、运行目录、时长和资源，
将补丁作为可审核产物，并以实际测试结果更新状态。

## 回归验证

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_research_tools.py tests/test_research_decision.py tests/test_research_service.py tests/test_public_llm.py -q
```

覆盖工具目录、schema、源码行号、AST、语法、不执行模块、路径逃逸、读取限额、
模型非法工具回退、去重、文献搜索读页、服务端公网权限和HTTP模式；符号链接
测试在Windows缺少创建权限时明确跳过，不当作已通过。
