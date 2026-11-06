# Render 课题组训练部署

本仓库已提供口令登录、数据集导入、后台 CPU 训练和持久化配置。**提供配置文件不代表现有公网服务已升级。** 免费演示部署继续使用 `Dockerfile` / `render.yaml`。

若训练改为使用用户电脑，可优先使用 [本地算力连接器](LOCAL_COMPUTE.md)。这种方式不需要本页的付费训练资源；公网仅中转任务，电脑需要保持连接器运行。

## 部署决定

课题组规模先采用单个 Web Service、单个 Uvicorn worker、单个并发训练进程。这样无需额外队列服务，任务、Session、上传数据与结果都保存在同一持久化磁盘。代价是训练期间资源与网页共用，重部署会中断训练。需要多人并发或 GPU 时，应拆出独立训练 worker 与任务队列。

口令代表课题组成员身份，成员之间共享数据和结果，没有逐用户账户与隔离。访客继续浏览演示。成员权限只增加数据导入、训练、共享 Session 与训练档案；研究助手执行、任意文件导入及服务器供应商配置仍受公开演示限制。

## 资源提案

`render-training.yaml` 提供待审核的付费配置：`2c-4g` CPU 实例（4 GB 内存、2 CPU）和 20 GB 持久化磁盘。内存余量用于 PyTorch、数据集和网页同时运行。本次核对 [Render 官方定价](https://render.com/pricing)，实例为 **85 美元/月**，磁盘为 **0.25 美元/GB/月**，这套配置约 **90 美元/月**，未计税费、额外流量及可能的工作区费用。**最终费用以 Render Dashboard 的确认页面为准**。[Blueprint 规范](https://render.com/docs/blueprint-spec) 当前使用 `2c-4g` 等 plan ID。

训练镜像安装 CPU 版 `torch==2.5.1`、`torchvision==0.20.1`，使用 Python 3.11；没有 CUDA/GPU。启动脚本只初始化挂载目录属主，随后以 UID 10001 运行 Python。只允许挂载路径 `/var/lib/spacefl`。

| 环境变量 | 设置 |
| --- | --- |
| `SPACEFL_PUBLIC_DEMO` | `1`，保留其他公开 API 的保护 |
| `SPACEFL_PUBLIC_TRAINING` | `1`，启用课题组空间 |
| `SPACEFL_DATA_DIR` | `/var/lib/spacefl` |
| `SPACEFL_TRAINING_ACCESS_KEY` | 管理员在 Render 私密环境变量中设置，至少 32 字符的随机口令 |
| `SPACEFL_OFFLINE_ONLY` | `1`，研究助手使用离线规则；不阻止 torchvision 下载内置数据集 |

口令不要写进 Git、构建参数、公开示例、URL 或前端存储。可在本机密码管理器生成，然后仅在 Render 与成员之间分发。更换口令并重部署会使原 Cookie 失效。登录要求 HTTPS；Cookie 为 HttpOnly、Secure、SameSite=Strict，有效期 12 小时。退出清除当前浏览器 Cookie；共享口令模式不提供逐个会话的服务端撤销。登录尝试按来源限速。

## 将现有网址升级

现有服务记录：`srv-dbm9qh08v5cc73e0b1lg`，`https://spacefl-research-agent.onrender.com`。保持该网址时，修改现有服务，避免另建服务导致地址改变。

1. 将审核并测试后的代码发布到现有服务使用的仓库分支；先关闭 Auto Deploy，避免分阶段配置期间自动部署。
2. 在现有服务设置中核对并选择至少 4 GB / 2 CPU 的付费实例；查看其费用后再确认。
3. 添加 20 GB 磁盘，挂载到 `/var/lib/spacefl`；核对磁盘费用后确认。
4. Dockerfile Path 设置为 `./Dockerfile.training`，Docker Context 为 `.`；按上表设置环境变量。`SPACEFL_TRAINING_ACCESS_KEY` 由管理员填写。
5. 保持实例数为 1，健康检查 `/api/health`。手动部署并检查日志；脚本必须能够写入挂载目录。
6. 按下文验收。磁盘部署存在重启停机时间，请在无训练任务运行时发布。

也可以使用 `render-training.yaml` 创建 Blueprint，需在 Render 指定该文件路径。新 Blueprint 不保证自动接管现有手工创建的服务；保留已有网址优先按上述步骤修改现有服务。不要直接覆盖免费 `render.yaml` 后自动推送。

## 训练范围与限制

- 支持 FedAvg、FedProx、FedBuff；MNIST、Fashion-MNIST、CIFAR-10，以及已上传的 ImageFolder 单标签 RGB 分类。FEMNIST 与独立 FedLEO 验证仍用本地环境。
- CPU、小模型、单客户端训练 worker、零 DataLoader 子进程。最多 30 轮、每轮本地 5 Epoch、10 颗卫星、每星 2000 样本、Batch 128、动态 Epoch 上限 5。单任务最长 30 分钟；超时任务失败，日志保留。
- 一次只有一个训练任务；重复提交返回 409。提交时直接冻结表单配置，之后保存共享 Session 不改变已有任务。
- ZIP 最大 200 MB、解压后最大 1 GB、最多 50000 张 JPG/PNG；结构为 `train/类别/图片`、`test/类别/图片`，两侧包含相同的至少两个类别。
- 最多 20 个共享数据集；每次导入前预留 1 GB 解压空间和 2 GB 空闲余量。达到数量或空间限制时由管理员备份和清理未使用目录；当前没有网页删除接口。
- 内置数据集下载到持久化 `cache/`；自定义训练只允许注册导入的数据集目录。不同成员不能传入服务器任意文件路径。

## 持久化与验收

磁盘布局：

```text
/var/lib/spacefl/
  datasets/<dataset-id>/{metadata.json,train/,test/}
  cache/                         # 内置数据集
  session.json                   # 共享表单默认配置
  experiments/train_<job-id>/
    session.json                 # 冻结的本次训练配置
    job.json                     # 状态和最终日志摘要
    training.log
    training_summary.json        # 成功训练后生成
```

重启恢复上传数据、历史状态和实验结果；尚未完成的任务标为“服务重启，训练已中断”，需要成员重新提交。不会恢复优化器或自动续训。管理员应定期备份持久化磁盘，清理时避开运行中的任务及其数据集。

1. 访客访问 `/api/health` 应返回 200；`/api/training/access` 应显示 `enabled=true, authenticated=false`；访客数据目录不包含成员上传，上传和启动训练返回 401。
2. 用 HTTPS 打开“实验配置”，输入口令。错误口令失败；登录后显示资源上限。导入一个小型两类别 ZIP，以 2 颗卫星、1 轮、1 Epoch 训练。
3. 检查轮次进度、最终准确率、实验档案与失败日志；任务运行期间再次提交应返回 409。超限参数应返回 422。
4. 退出后成员数据及任务不可读取。访客仍能查看预置演示。
5. 在无训练任务运行时重部署；重新登录后核对上传数据、已完成任务与档案仍在。可在专门测试任务运行时重启，确认中断状态正确。

本地自动验证：

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_public_training.py tests/test_web_training.py tests/test_web_datasets.py -q
node --check web/assets/app.js
docker build -f Dockerfile.training -t spacefl-training .
```

Docker 构建与实际 Render 部署还需在具备对应运行环境和资源的机器上完成。不要把 Python 集成测试通过等同于镜像或公网验收通过。
