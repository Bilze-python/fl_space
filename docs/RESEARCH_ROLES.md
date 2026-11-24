# 参谋、指挥与审计：科研 Agent 三角色协作

状态：本地已接入研究助手“代码与原文”模式。默认仍为离线规则；LLM 和 Jev
分别遵守已有服务端开关，尚未公网部署，未发起真实计费 Jev 请求。

## 职责与调用顺序

1. LLM 参谋理解语义，结合工具目录与最近结果，提出最多4个候选动作。
2. verifier 校验候选 schema、工具权限与参数。违规提案拒绝并回退规则。
3. 代码删除重复/超预算候选，无证据时移除 compose，加入 stop/ask/refuse。
4. 决策模型指挥从合法候选ID中选择，不直接接触执行器；拒绝未知ID/非法概率。
5. 执行器再次检查参数、路径与硬预算，执行已选工具，记录原始结果。
6. 指挥选择 compose 后，LLM 参谋编排带 source_id 的原文摘录。
7. 审计检查字段、来源ID、逐字来源匹配、数字出处与摘录证据覆盖，拒绝编造。

模块：`commander.py` 编排，`decision.py` 供应商接口，`verifier.py` 审计，
`tools.py` 工具执行。ResearchBudget 与 run_commanded_research 可从包入口导入。

## 指挥控制与硬边界

指挥可选 tool、compose、stop、ask、refuse；ask 返回 needs_input 和澄清提示，
refuse 返回 refused；两者都会停止本次任务，不自动给其他会话发送消息。
当前澄清文案由代码生成，尚未实现多轮任务恢复。

默认最多5个工具调用、4次逻辑 LLM 调用、5次决策调用；LLM 生成候选和写作
都消耗 LLM 配额，工具失败同样消耗工具配额。指挥可以提前结束或选择不同成本
动作，但不能增加上限。预算单位是逻辑调用配额，不是货币、token或供应商账单；
底层协议兼容重试、系统负载和单次超时不等于整个任务有严格墙钟时间保证。
一次决策失败后本次任务改用离线指挥，不重复尝试外部服务。

Jev 收到的状态新增：候选ID、动作种类、受控工具名称、调用成本、剩余预算。
不发送用户问题、候选参数、源码/文献原文或路径；因此可做流程选择，但无法
仅凭这些元数据区分两个同工具不同参数候选的语义优劣。这是首版能力边界。
概率仍是未校准动作偏好，不代表结论正确率。全局关闭状态不会被程序开启。

## 参谋语言与审计范围

第一版允许参谋选择、排序逐字来源摘录，并由代码生成答复格式，不开放无来源
自由改写。这样数字和引用可严格核对，但尚不能验证单位含义、实验公平性或
论文真实有效性。semantic_support 始终 not_assessed。coverage 只表示输出摘录
的来源覆盖，不能解释为问题回答完整度。审计失败后返回代码选择的来源摘录。

原 evidence_workflow=true 文献摘录流程保留兼容；新三角色循环已用于 tool_mode=true，
未给旧计划生成、实验执行或普通聊天全面切换编排。旧 run_tool_research 作为
兼容工具函数保留，不代表它本身已采用指挥架构。

## 使用与观察

网页选择“代码与原文”，查看“参谋候选 / 指挥选择 / 审计与预算”折叠面板。
HTTP 仍为 POST /api/research/agent/chat，JSON 传 tool_mode=true。

```powershell
.\venv\Scripts\python.exe scripts/run_evidence_research.py --tools --question "evaluate_link" --output output/research-tools.json
```

默认离线；加 --llm 才允许 CLI 调用已配置 LLM。Jev 的本地服务端选择仍需
SPACEFL_RESEARCH_DECISION=jev 和已打开的用户级 JEV_ENABLED；PUBLIC_DEMO 与
SPACEFL_OFFLINE_ONLY 继续覆盖在线决策配置。

结果包含 status、roles、command_trace、tool_trace、claims、citations、verification
和 budget。command_trace 记录候选、选择、参谋回退、审计结果及预算前后值；
这属于可观察工作记录，不是模型内部思考链。

回归验证覆盖指挥否决、选择第二候选、停/问/拒、预算、非法动作、模型编造
数字、脱离来源、协议兼容、工具隔离及原有流程。没有证明实际 Jev 比规则
基线更有效，后续须用带标签科研任务集进行影子与配对评估。
