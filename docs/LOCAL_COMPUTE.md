# 使用用户电脑训练

公网平台负责页面和任务中转；用户主动启动本地连接器，以 HTTPS 轮询领取训练任务。
现有 PyTorch 训练器在该电脑执行，可以选择 CPU 或 NVIDIA CUDA。
网页无需请求 localhost、开放入站端口或修改路由器。
本功能需将本次代码发布至公网服务才能在公网使用；仅修改本地文件不会更新 Render。

## 用户操作

1. 电脑准备本项目的当前版本和 Python 3.10+。首次安装，在项目目录执行：

   ```powershell
   python -m venv venv
   .\venv\Scripts\python.exe -m pip install -e ".[web]"
   .\venv\Scripts\python.exe -m pip install torch torchvision
   ```

   CUDA 用户应按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择适合驱动的安装命令。
   已有本项目虚拟环境且能训练的电脑无需重复安装。

2. 启动本地连接器并保持终端运行：

   ```powershell
   .\venv\Scripts\python.exe -m web.local_worker connect --device auto
   ```

   也可以执行 `./start-local-compute.ps1`。`auto` 在 CUDA 可用时使用 CUDA，否则使用 CPU。
   强制 CPU：`--device cpu`；强制 CUDA：`--device cuda`，检测不到时明确报错。
   默认 CPU 数学库线程上限为 2，单任务限时 3600 秒。
   可在本人电脑增加 `--threads 4 --max-seconds 7200`，最大分别为 64 和 86400。
   一次运行一个任务。关闭浏览器不会停止已领取的训练；本地终端 Ctrl+C 会停止子进程。

3. 在网页“实验配置 → 训练设置 → 训练算力”选择“我的电脑”，输入终端的 12 位配对码。
   码 10 分钟有效、只能使用一次，只交给本人。配对后网页获得该连接器的训练控制权限。
   连接有效期为注册起 12 小时，刷新同一标签页保留连接；其他电脑使用各自的码和任务队列。
   此连接使用独立配对凭据；课题组口令用于平台服务器训练空间。

4. 在表单选择数据集、设置参数，点击“启动联邦训练”。网页显示进度、准确率，支持停止训练。
   完成后点击“查看并下载本机训练摘要”，下载 JSON 并查看逐轮曲线。
   连接面板显示最近 10 个任务，点击可重新查看。

## 自定义卫星影像数据

在同一项目目录、另一个终端中执行：

```powershell
.\venv\Scripts\python.exe -m web.local_worker import "D:\data\satellite.zip" --name "卫星影像"
```

ZIP 包含 `train/类别/图片` 和 `test/类别/图片`，两边包含相同的至少两个类别；JPG/PNG，
ZIP 最大 200 MB，解压最大 1 GB，最多 50000 张图片。验证通过后导入本机受管目录。
连接器每次空闲轮询同步数据集名称、类别和样本数；训练期间新导入的目录在任务结束后同步。
原始图片不上传平台。使用当前版本连接器时，也可在网页的公开数据集选项中选择 EuroSAT RGB，点击“添加到数据集”；连接器从 Zenodo 下载约 95 MB 的官方 RGB 包、校验并按类别固定划分 80/20，完成后自动选中。旧版连接器需要更新并重新配对。其他数据集见
[卫星公开数据集](SATELLITE_DATASETS.md)。多标签、检测、分割任务仍需对应训练器。

默认本地目录为项目下的 `local-compute/`（已加入 Git 忽略）：

- `datasets/<id>/`：导入的影像和元数据。
- `cache/`：MNIST、Fashion-MNIST、CIFAR-10 等内置数据首次下载到这里。
- `experiments/<job-id>/session.json`：实际执行的配置快照。
- `experiments/<job-id>/training.log`：末尾训练日志。
- `experiments/<job-id>/training_summary.json`：完整训练输出，包含逐轮指标和事件。
- `experiments/<job-id>/job.json`：成功、失败或取消的最终本地记录。

可用 `--workspace "E:\SpaceFLData"` 更换存储目录；**import 和 connect 必须使用同一个目录**。
回传的内容只有任务配置（剔除本机路径）、状态、准确率曲线；不会回传原始图片、模型权重或本地日志。
模型是否额外保存检查点取决于原训练器；本连接器不额外实现权重归档。

## 运行范围和中断

本机连接器当前支持 small 分类模型、FedAvg/FedProx/FedBuff、CPU/CUDA、上述内置数据和 ImageFolder。
使用既有 `fl_space.cli run train`，不新增轨道物理计算。最多 1000 轮、100 Epoch、100 个客户端、
单星 20000 样本、Batch 512；所有参数在平台和本机各校验一次，设备由本机选择。
本机程序只运行固定训练命令，忽略网页传入的数据路径，只能选本人导入的逻辑 ID。
公网网页的本机模式直接提交配置快照，不改写平台共享 Session；切换运行位置不会自动迁移或停止任务。

平台中转记录保存在**单进程内存**，适用于当前单实例演示服务，无需在平台安装 PyTorch 或附加训练磁盘。
Render 休眠、重启、发布会丢失配对和任务摘要。请重新启动连接器并重新配对，历史完整文件仍在本机。
临时网络故障时训练继续，连接器尝试恢复进度回传；最终摘要回传失败会提示并退出，文件仍可在本地读取。
解除连接会撤销网页和连接器的凭据；已开始的训练会继续在本机完成，停止请用网页停止按钮或本地 Ctrl+C。
长时间离线的网页停止请求无法立刻送达，以本地 Ctrl+C 为准。
不支持跨平台重启恢复中转、自动恢复训练、多副本中转和后台无人值守系统服务。

## 本地验证

终端一启动平台：

```powershell
.\venv\Scripts\python.exe -m uvicorn web.server:create_app --factory --host 127.0.0.1 --port 8722
```

终端二启动连接器：

```powershell
.\venv\Scripts\python.exe -m web.local_worker connect --server http://127.0.0.1:8722 --device cpu
```

公网必须使用 HTTPS，只有 localhost / 127.0.0.1 / ::1 调试允许 HTTP。
浏览器打开同一平台地址并配对。测试：`python -m pytest tests/test_local_compute.py -q`。

## 架构决定

采用本机主动访问平台的 HTTPS 轮询：复用 PyTorch、固定 CLI 和现有部署，没有入站服务。
代价是用户必须先安装并保持本地程序运行，进度延迟约 3 秒，平台内存记录不持久化。
浏览器 WebGPU 能省去 Python 安装，但需要另写模型、数据处理与联邦训练运行时；现阶段不采用。
平台直接访问 localhost 则受 HTTPS 混合内容和浏览器本地网络权限影响，因此采用出站连接。
平台服务器训练仍可单独使用，资源要求参见 [Render 训练部署](RENDER_TRAINING.md)。
