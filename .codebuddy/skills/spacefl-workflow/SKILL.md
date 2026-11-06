---
name: spacefl-workflow
description: 在 SpaceFL 仓库中修改或构建训练、数据集、卫星仿真、局部星簇、科研平台和实验功能，并验证可运行结果。
---

# SpaceFL 项目工作流

此 skill 用于 `D:\fl_space-master`。用户要求修复、扩展、编写或构建项目时，直接在仓库中完成相应代码、配置、界面和测试；用户只问原理或方案时先回答问题。不要把研究助手的文字建议当作已经完成的代码修改。

## 找到正确入口

- 网页入口与 API：`web/index.html`、`web/assets/app.js`、`web/assets/app.css`、`web/server.py`。
- 联邦训练：`fl_space/cli.py` 的 `run train`，以及 `fl_space/fl/runner.py`、`server.py`、`config.py`。
- 数据集导入：`web/dataset_store.py`；训练器当前支持 MNIST、Fashion-MNIST、CIFAR-10、FEMNIST 和 `ImageFolder` 单标签分类。
- 用户本机算力：`web/local_compute.py` 为无 PyTorch 的内存中转队列；`web/local_worker.py` 主动出站连接平台，按配对控制权运行固定训练 CLI；`web/assets/local-compute.js` 提供网页配对、队列和摘要下载。使用说明：`docs/LOCAL_COMPUTE.md`，Windows 启动：`start-local-compute.ps1`。
- 本机连接和课题组口令是独立权限。不得让网页指定任意本机路径或命令；平台和本机都校验 Session，用本地数据集逻辑 ID 映射路径。原始数据、完整输出保留本机，平台只接收元数据和指标。中转记录只在单进程内存中，重启后需重新配对；不可宣称跨重启恢复或已上线。
- 局部星簇解析实验：`fl_space/local_cluster/`；网页接口在 `web/server.py`，可视化在 `web/assets/app.js`。
- 轨道、接触与链路：`fl_space/orbit/`、`fl_space/simulator/`、`fl_space/isl/`。
- 学习任务与航天任务：`fl_space/study/`、`fl_space/mission/`。
- 公网 Render 部署：免费演示使用 `render.yaml`、`Dockerfile`；课题组训练使用 `render-training.yaml`、`Dockerfile.training` 和 `web/training_runtime.py`。详见 `docs/RENDER_TRAINING.md`。
- 公网训练必须同时保留 `SPACEFL_PUBLIC_DEMO=1` 和 `SPACEFL_PUBLIC_TRAINING=1`：访客只读，口令登录成员可上传并训练；其他配置、文件与任意执行接口继续受只读限制。
- 公网研究助手默认仍是离线规则引擎。只有明确配置 `SPACEFL_PUBLIC_LLM=1`、`SPACEFL_OFFLINE_ONLY=0` 和服务端 `DEEPSEEK_API_KEY` 时才开放在线模型；密钥只放 Render 私密环境变量，不进仓库/前端/日志。公网调用有进程内限流，供应商账户还需单独设置费用上限；免费服务重启会重置进程内计数。
- “平台内置 skill”指本仓库的 `spacefl-workflow`。维护此文件中的入口、能力边界与验证命令；代码由具备本地工作区权限的编码工具修改。

## 直接编码与构建

1. 读 `AGENTS.md`、当前实现和 `git status`，确认用户已有改动及目标运行环境。需要改变结构、模型或物理保真度时，先读相应参考文档并写清边界。
2. 找到完整调用路径，从用户界面或 CLI 到 API、领域逻辑、输出与测试；按现有模块边界实施改动。对于新项目构建，先做一个可运行的纵向流程，再补齐必需的状态、错误处理和可视化。
3. 数据集改动要核对任务类型、标注格式、训练/测试划分、类别映射、来源和许可。只有符合 `train/类别/图片` 与 `test/类别/图片` 的单标签图像可直接进入现有 `ImageFolder` 训练器；多标签、检测和分割需要相应模型与评价指标。
4. 对网页训练流程，验证原子提交的 Session 快照、数据加载、后台作业、进度、结果归档与失败日志。公网训练还必须验证 HTTPS 口令登录、访客隔离、受管数据路径、CPU/样本/轮次限制和持久化磁盘；服务重启后的未完成任务应标记中断。采用单实例、单 worker、单训练作业的课题组共享空间，不宣称逐用户隔离。对卫星拓扑可视化，图上连接必须对应实际拓扑配置，物理量和示意位置分开标注。
5. 运行覆盖改动的测试、静态检查和最小真实执行；前端改动启动本地服务检查 API 与页面。报告已实现内容、验证证据，以及公网部署与本地能力的差异。不要在未经用户授权时把本地改动推到公网。

## 参考与命令

涉及轨道、卫星、通信、能量、阻力或姿态时，阅读 [卫星物理路由](references/space-satellite-physics.md) 和 [工程规范路由](references/engineering-standards.md)，再按其链接阅读 `docs/references/` 中的完整规范。只改现有 FL 算法时不需要加载物理参考。

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_web_datasets.py tests/test_local_cluster.py -q
.\venv\Scripts\python.exe -m pytest tests/test_public_training.py tests/test_web_training.py -q
.\venv\Scripts\python.exe -m pytest tests/test_local_compute.py -q
.\venv\Scripts\python.exe -m fl_space.cli run train --rounds 2 --epochs 1 --output output/train.json
.\venv\Scripts\python.exe -m uvicorn web.server:create_app --factory --host 127.0.0.1 --port 8721
```

这些命令是本地示例；先确认虚拟环境、端口、数据集与 Session 参数。不要使用旧 `fl-space experiment` 示例作为当前 CLI 的依据。
