# fls CLI 参考手册

> SpaceFL 三层指令架构：`tune` 调参 → `mount` 挂载 → `run` 运行

---

## 1. 快速开始

```powershell
# 安装
pip install -e "."

# 查看完整帮助
fls help

# 查看环境信息
fls info
```

---

## 2. 指令架构总览

```
fls
├── help                        显示分类帮助
├── info                        系统与环境信息
├── completion install          安装 Tab 补全
│
├── tune <参数> <值>            调参指令 — 管理超参数
│   ├── lr, rounds, epochs, batch, mu, seed, ...
│   ├── show                    查看当前调参
│   └── reset                   重置为默认值
│
├── mount <组件> <值>           挂载指令 — 选择算法 / 组件
│   ├── algo, isl, backend, body, distribution, ...
│   ├── config <path>           加载 JSON 配置文件
│   ├── show                    查看当前挂载
│   └── clear                   重置为默认值
│
└── run <实验类型> [--覆盖]     运行指令 — 执行实验
    ├── simulate                轨道接触模拟
    ├── train                   FL 训练实验
    ├── experiment              完整太空实验 (FedAvg/FedProx 网格)
    ├── quick-test              FedProxSat 快速测试
    ├── list [资源]             查看内置资源
    ├── export                  导出模拟结果 JSON
    ├── serve                   启动 CesiumJS 3D 可视化
    └── show                    查看完整 session 状态
```

**核心设计：**
- `tune` / `mount` 的修改持久化到 `.fls_session.json`
- `run` 消费当前 session，支持 `--` 参数覆盖
- 参数优先级：**CLI 覆盖 > session 值 > 默认值**

---

## 3. 工作流示例

### 3.1 快速单次实验

```powershell
# 1. 调参
fls tune lr 0.001
fls tune rounds 500
fls tune dataset cifar10

# 2. 挂载算法
fls mount algo fedprox
fls mount isl wgs84
fls mount isl-buffer 80

# 3. 查看当前配置
fls run show

# 4. 运行训练
fls run train --quiet --output result.json

# 5. 运行完整模拟
fls run simulate --sats 10 --hours 48
```

### 3.2 覆盖式运行（不改 session）

```powershell
# 不改session，仅本次运行覆盖参数
fls run train --lr 0.1 --rounds 100 --device cuda --no-session
```

### 3.3 批量实验脚本

```powershell
# 基准 FedAvg
fls mount algo fedavg
fls tune lr 0.01
fls run train --output fedavg_bench.json

# FedProx 对比
fls mount algo fedprox
fls tune mu 0.1
fls run train --output fedprox_bench.json

# ISL 启用对比
fls mount isl wgs84
fls mount isl-buffer 80
fls run simulate --output sim_with_isl.json
```

---

## 4. tune — 调参指令

所有 tune 参数保存到 session，`run` 时自动消费。

| 指令 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fls tune lr` | float | 0.01 | 学习率 |
| `fls tune rounds` | int | 300 | 训练轮次 |
| `fls tune epochs` | int | 5 | 本地训练 epoch |
| `fls tune batch` | int | 32 | batch size |
| `fls tune mu` | float | 0.01 | FedProx 近端项系数 μ |
| `fls tune buffer-size` | int | 5 | FedBuff 缓冲区大小 K |
| `fls tune seed` | int | 42 | 随机种子 |
| `fls tune dataset` | mnist / fashion_mnist / cifar10 | mnist | 数据集 |
| `fls tune scale` | small / medium / large | small | 实验规模 |
| `fls tune early-stop` | float | 0.90 | 早停准确率阈值 |
| `fls tune workers` | int | 1 | 并行训练线程数 |
| `fls tune data-workers` | int | 0 | DataLoader 进程数 |
| `fls tune non-iid` | on / off | off | non-IID 数据分布 |
| `fls tune alpha` | float | 0.5 | Dirichlet α (non-IID) |
| `fls tune device` | cpu / cuda | cpu | 计算设备 |
| `fls tune show` | — | — | 查看当前调参 |
| `fls tune reset` | — | — | 重置为默认值 |

---

## 5. mount — 挂载指令

选择算法和功能模块，保存到 session。

| 指令 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| `fls mount algo` | fedavg / fedprox / fedbuff | fedavg | FL 算法 |
| `fls mount isl` | disabled / wgs84 | disabled | ISL 星间链路计算器 |
| `fls mount isl-buffer` | float | 0.0 | ISL WGS84 大气余量 (km) |
| `fls mount isl-step` | float | 60.0 | ISL 采样步长 (秒) |
| `fls mount time-model` | slot / physics | slot | 虚拟时间模型 |
| `fls mount time-model-args` | JSON 字符串 | null | 时间模型参数 |
| `fls mount backend` | kepler / skyfield | kepler | 轨道计算后端 |
| `fls mount body` | earth / mars / moon / jupiter / saturn / venus | earth | 中心天体 |
| `fls mount distribution` | uniform / walker / cluster | uniform | 星座分布策略 |
| `fls mount staleness` | on / off | off | FedBuff 陈旧度降权 |
| `fls mount sats` | int | 5 | 卫星数量 |
| `fls mount stations` | int | 3 | 地面站数量 |
| `fls mount sim-hours` | float | 24.0 | 模拟时长 (小时) |
| `fls mount timeslot-min` | float | 1.0 | 时隙粒度 (分钟) |
| `fls mount altitude` | float | 500.0 | 轨道高度 (km) |
| `fls mount inclination` | float | 53.0 | 轨道倾角 (°) |
| `fls mount config` | JSON 文件路径 | — | 加载 JSON 配置 |
| `fls mount show` | — | — | 查看当前挂载 |
| `fls mount clear` | — | — | 重置为默认值 |

---

## 6. run — 运行指令

### 6.1 run simulate — 轨道接触模拟

```powershell
fls run simulate [--覆盖参数]
```

| 覆盖参数 | 说明 |
|----------|------|
| `--sats N` | 卫星数量 |
| `--stations N` | 地面站数量 |
| `--hours N` | 模拟时长 (h) |
| `--backend kepler/skyfield` | 轨道后端 |
| `--altitude N` | 轨道高度 (km) |
| `--inclination N` | 轨道倾角 (°) |
| `--distribution uniform/walker/cluster` | 分布策略 |
| `--timeslot-min N` | 时隙粒度 (min) |
| `--body earth/mars/...` | 中心天体 |
| `--isl disabled/wgs84` | ISL 计算器 |
| `--isl-buffer N` | ISL 大气余量 (km) |
| `--output PATH` | 导出 JSON 路径 |
| `--quiet` / `-q` | 安静模式 |
| `--no-session` | 忽略 session 用默认值 |

```powershell
# 示例
fls mount sats 20
fls mount sim-hours 48
fls run simulate --backend skyfield --output sim.json
fls run simulate --sats 10 --hours 6 --quiet --no-session   # 纯默认值运行
```

### 6.2 run train — FL 训练实验

```powershell
fls run train [--覆盖参数]
```

| 覆盖参数 | 说明 |
|----------|------|
| `--rounds N` | 训练轮次 |
| `--epochs N` | 本地 epoch |
| `--lr N` | 学习率 |
| `--batch-size N` | batch size |
| `--mu N` | FedProx μ |
| `--buffer-size N` | FedBuff K |
| `--device cpu/cuda` | 计算设备 |
| `--seed N` | 随机种子 |
| `--time-model slot/physics` | 时间模型 |
| `--time-model-args JSON` | 时间模型参数 |
| `--output PATH` | 导出 JSON |
| `--quiet` / `-q` | 安静模式 |
| `--no-session` | 忽略 session |

```powershell
fls mount algo fedprox
fls tune lr 0.001
fls run train --output result.json
```

### 6.3 run experiment — 完整太空实验

```powershell
fls run experiment [--覆盖参数]
```

| 覆盖参数 | 说明 |
|----------|------|
| `--gs N1 N2 ...` | 地面站数量列表 (FedAvg 默认: 3 5 7 10) |
| `--sats-list N1 N2 ...` | 卫星数量列表 (FedAvg 网格) |
| `--sats-single N` | 卫星数 (FedProx 单组实验) |
| `--output DIR` | 输出目录 |
| `--quiet` / `-q` | 安静模式 |

```powershell
# FedAvg 网格搜索
fls mount algo fedavg
fls run experiment --gs 3 5 7 --sats-list 5 10 15

# FedProx 异构轨道
fls mount algo fedprox
fls tune mu 0.1
fls run experiment --gs 1 3 5 --sats-single 10 --output fedprox_results
```

### 6.4 run quick-test — FedProxSat 快速测试

```powershell
fls run quick-test [--覆盖参数]
```

| 覆盖参数 | 说明 |
|----------|------|
| `--mu N` | 基础 μ |
| `--mu-min N` | 自适应 μ 下限 |
| `--mu-max N` | 自适应 μ 上限 |
| `--no-adaptive` | 禁用自适应 μ |
| `--oscillation-threshold N` | 震荡阈值 |
| `--stability-threshold N` | 稳定阈值 |
| `--rounds N` | 轮次 |
| `--gs N` | 地面站数 |
| `--quiet` / `-q` | 安静模式 |

```powershell
fls run quick-test --mu 0.01 --gs 5
```

### 6.5 run list — 查看内置资源

```powershell
fls run list [presets|models|satellites|experiments]
```

### 6.6 run export — 导出模拟结果

```powershell
fls run export --output result.json [--覆盖参数]
```

参数同 `run simulate`。

### 6.7 run serve — 3D 可视化

```powershell
fls run serve [--覆盖参数]
```

| 覆盖参数 | 说明 |
|----------|------|
| `--host` | 监听地址 (默认 0.0.0.0) |
| `--port` | 端口 (默认 8700) |
| `--serve-sats N` | 卫星数 |
| `--serve-gs N` | 地面站数 |
| `--serve-hours N` | 模拟时长 |
| `--serve-isl disabled/wgs84` | ISL 启用 |

```powershell
fls mount isl wgs84
fls mount isl-buffer 80
fls run serve --serve-sats 10 --port 8080
```

---

## 7. Session 文件格式

`.fls_session.json`:

```json
{
  "tune": {
    "lr": 0.01,
    "rounds": 300,
    "epochs": 5,
    "batch_size": 32,
    "mu": 0.01,
    "seed": 42,
    "dataset": "mnist",
    "scale": "small",
    "early_stop": 0.9,
    "workers": 1,
    "data_workers": 0,
    "non_iid": false,
    "alpha": 0.5,
    "device": "cpu",
    "buffer_size": 5
  },
  "mount": {
    "algo": "fedavg",
    "isl": "disabled",
    "isl_buffer": 0.0,
    "isl_step": 60.0,
    "time_model": "slot",
    "time_model_args": null,
    "backend": "kepler",
    "body": "earth",
    "distribution": "uniform",
    "staleness": false,
    "sats": 5,
    "stations": 3,
    "sim_hours": 24.0,
    "timeslot_min": 1.0,
    "altitude": 500.0,
    "inclination": 53.0,
    "config": null
  }
}
```

---

## 8. Tab 补全

### 8.1 argcomplete (推荐)

```powershell
# 安装
pip install argcomplete

# 激活 (PowerShell)
Register-ArgumentCompleter -Command fls -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    & python -m fl_space.cli _complete $commandAst
}
```

### 8.2 内置 PowerShell 脚本

```powershell
fls completion ps1 > fls_completion.ps1
. .\fls_completion.ps1
```

---

## 9. Python API 接入

```python
from fl_space.cli import load_session, save_session, cmd_run_simulate

# 编程方式读写 session
s = load_session()
s["tune"]["lr"] = 0.001
s["mount"]["algo"] = "fedprox"
save_session(s)
```

---

## 附录：常见问题

**Q: 如何分享实验配置？**
```powershell
fls run show                    # 查看当前配置
# 复制 .fls_session.json 给他人
fls mount config partner.json  # 对方加载
```

**Q: 如何批量运行多组实验？**
```powershell
for mu in 0.001 0.01 0.1; do
    fls tune mu $mu
    fls run train --output "mu_${mu}.json"
done
```

**Q: 如何只覆盖一次参数？**
```powershell
fls run train --lr 0.1 --no-session  # session 不会被修改
```
