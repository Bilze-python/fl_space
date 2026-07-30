# SpaceFL — 太空联邦学习研究框架

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/badge/Linting-Ruff-ok)](https://github.com/astral-sh/ruff)

模块化、可扩展的开源框架，用于研究太空联邦学习（Space Federated Learning）中的**轨道环境模拟**、**星地/星间通信**与 **FL 算法对比实验**。

---

## 特性

| 特性 | 说明 |
|------|------|
| 🛰️ **双轨道后端** | Kepler 二体近似（极速） / Skyfield SGP4 + JPL DE421（高精度） |
| 📡 **接触矩阵通信模型** | 地面站-卫星接触窗口 + ISL 星间链路 |
| 🧠 **3 种 FL 算法** | FedAvg / FedProx / FedBuff，可插拔架构 |
| 🔗 **FedLEO 去中心化** | 分层聚合 + 离散卸载，消除地面站瓶颈 (Zhai 2024) |
| ⏱️ **可插拔时间模型** | SlotTimeModel / PhysicsTimeModel (FLOPs/带宽驱动) |
| 📊 **4 种数据集** | MNIST / Fashion-MNIST / CIFAR-10 / **FEMNIST** (writer-level non-IID) |
| 🖥️ **专用终端 + 控制面板** | `fls` CLI 全命令 + 交互式 PySimpleGUI 面板 |
| 📈 **标准化输出** | 准确率曲线、热力图、接触统计、轨道可视化、JSON 记录 |

---

## 架构

```
fl_space/
├── environment/      # 环境模拟层 — 天体、大气、地面站、坐标工具
├── orbit/            # 轨道力学层 — Kepler/SGP4 双后端、可见性判断
├── simulator/        # 模拟器层 — 接触矩阵、主模拟引擎
├── fl/               # FL 核心层 — 算法、服务器、Runner、时间模型、FEMNIST 加载器
│   ├── algorithms/   # FedAvg / FedProx / FedBuff
│   ├── server.py     # 联邦服务器 (FedAvg/FedProx/FedBuff)
│   ├── runner.py     # 实验编排器
│   ├── config.py     # 数据集预设 & 超参数
│   ├── models.py     # MLP / SimpleCNN
│   └── femnist_loader.py  # FEMNIST writer-level 加载器
└── fedleo/           # FedLEO 去中心化算法 — 卸载规划器、分层聚合、合规声明
```

**双后端设计：**
| 后端 | 精度 | 依赖 | 适用场景 |
|------|------|------|---------|
| `kepler` (默认) | 二体近似 | 仅 numpy | 快速原型、概念验证 |
| `skyfield` | SGP4 + JPL DE421 | `pip install skyfield` | 高精度研究、真实任务规划 |

---

## 快速开始

```bash
git clone git@github.com:Bilze-python/fl_space.git
cd fl_space
pip install -e .                    # 开发模式安装
pip install -e ".[skyfield]"        # 含 Skyfield 高精度后端
pip install datasets                # FEMNIST 数据集支持
```

### 5 分钟示例

```python
from fl_space.environment import CelestialBody, create_default_network
from fl_space.orbit import KeplerOrbit, create_circular_orbit
from fl_space.simulator import OrbitSimulator

# 创建地球 + 3颗卫星 + 7个地面站
earth = CelestialBody.earth()
gss = create_default_network(7)
orbits = [create_circular_orbit(500, 53, 0, i * 120, earth) for i in range(3)]

sim = OrbitSimulator(
    body=earth, orbits=orbits, ground_stations=gss,
    duration_hours=24,
)
print(f"接触率: {sim.contact_rate:.1%}")
```

### 专用终端 (CLI)

```bash
# 查看所有命令
python -m fl_space.cli --help

# 切换数据集
python -m fl_space.cli tune dataset femnist

# 运行 FL 实验 (FedAvg/FedProx/FedBuff)
python -m fl_space.cli run train --algorithm fedprox --mu 0.05 --rounds 100

# 运行 FedLEO 去中心化实验 (自动对比 FedAvg 基线)
python -m fl_space.cli run fedleo --planes 3 --sats-per-plane 5 --rounds 30

# 三算法对比
python examples/compare_all_three.py --dataset femnist --sats 50
```

### 控制面板

```bash
python control_panel.py
# → 交互式菜单：调参面板 / FL训练 / FedLEO实验 / 数据分析
```

---

## FL 算法

| 算法 | CLI 名称 | 论文 | 关键参数 |
|------|---------|------|---------|
| **FedAvg** | `fedavg` | McMahan 2017 | 标准加权平均 |
| **FedProx** | `fedprox` | Li 2020 | `--mu` 近端约束系数 (推荐 0.01-0.1) |
| **FedBuff** | `fedbuff` | Nguyen 2022 | `--buffer-size` / `--staleness-thresh` 异步缓冲 |
| **FedLEO** | `fedleo` | Zhai 2024 | `--offload-every` / `--max-offload-iter` 去中心化卸载 |

---

## 数据集

| 数据集 | CLI 名称 | 类别 | 样本尺寸 | non-IID 方式 |
|--------|---------|------|---------|-------------|
| MNIST | `mnist` | 10 | 28×28×1 | Dirichlet shard |
| Fashion-MNIST | `fashion_mnist` | 10 | 28×28×1 | Dirichlet shard |
| CIFAR-10 | `cifar10` | 10 | 32×32×3 | Dirichlet shard |
| **FEMNIST** | `femnist` | **62** | 28×28×1 | **Writer-level** (3597 writers) |

> FEMNIST 是 LEAF benchmark 的核心数据集，62 个类别（0-9/A-Z/a-z），每个 writer 的数据天然异构，
> 比 MNIST shard 更接近真实联邦学习场景。首次使用自动从 HuggingFace 下载并缓存 (~664MB)。

---

## 时间模型

| 模型 | CLI 参数 | 适用场景 |
|------|---------|---------|
| `SlotTimeModel` | `--time-model slot` | 概念验证、算法对比（默认） |
| `PhysicsTimeModel` | `--time-model physics` | FLOPs/带宽驱动的秒级精度 |

```bash
python -m fl_space.cli run train --time-model physics --time-model-args '{"link_bw_mbps":10}'
```

---

## 标准化实验输出

每次实验自动生成 `output/` 目录，包含：

- `_ckpt_*.json` — 每轮完整状态（accuracy, loss, 卫星参与, 时间分解）
- `_accuracy_curve.html` — Plotly 交互式准确率曲线
- `_contact_heatmap.html` — 卫星-地面站接触热力图
- `_orbit_demo.png` — 轨道剖面可视化
- `_contact_stats.png` — 接触次数统计条形图

---

## 运行测试

```bash
# 全部测试
pytest tests/ -v

# FL 算法语义测试
pytest tests/test_fl_algorithms.py -v

# FedLEO 合规性测试
pytest tests/test_fedleo_conformance.py -v

# PR#2 安全优化回归测试
pytest tests/test_pr2_safe_optimizations.py -v

# 代码规范
ruff check fl_space/
```

---

## 项目开发状态

- [x] 环境模拟 (Kepler + Skyfield 双后端)
- [x] 接触矩阵通信模型
- [x] 3 种 FL 算法 (FedAvg / FedProx / FedBuff)
- [x] FedLEO 去中心化算法
- [x] 4 种数据集 (MNIST / Fashion-MNIST / CIFAR-10 / FEMNIST)
- [x] 可插拔时间模型 (Slot / Physics)
- [x] 专用终端 + 控制面板
- [x] 标准化实验输出与可视化
- [x] 接触矩阵向量化优化 (79s → 1.4s)
- [ ] Web 可视化面板
- [ ] 地面站部署优化算法

---

## 贡献指南 (CONTRIBUTING)

欢迎提交 PR，请遵守以下规则：

### PR 规范
- **单个 PR 不超过 30 文件 / 1000 行**
- 提交前运行 `ruff check fl_space/` 和 `pytest tests/`
- 大功能请拆分为多个独立 PR（每 PR 一个主题）

### 禁止事项
- ❌ 删除他人模块目录（`fedleo/`、`fl/` 等）
- ❌ 删除测试文件（`tests/` 下）
- ❌ 删除 `.codebuddy/` 工作记忆
- ❌ 提交二进制文件（`de421.bsp`、`.pkl`、`.pyc`、`.tar.gz`）
- ❌ 提交数据集文件（MNIST/FEMNIST 数据）
- ❌ 使用 UTF-16 编码（统一 UTF-8）

### 提交信息格式
```
type: 简短描述

- 具体变更1
- 具体变更2
```

Type: `feat` / `fix` / `perf` / `test` / `docs` / `chore`

---

## 相关论文

| 论文 | 标题 | 用途 |
|------|------|------|
| Matthiesen 2023 | 卫星星座联邦学习综述 | 框架设计参考 |
| Li 2020 (FedProx) | 异构网络联邦优化 | FedProx 算法 |
| Nguyen 2022 (FedBuff) | 异步联邦学习缓冲聚合 | FedBuff 算法 |
| Zhai 2024 (FedLEO) | 去中心化卸载辅助 FL | FedLEO 算法 |
| 主论文 (arXiv:2511.14889) | Bringing Federated Learning to Space | 整体框架 |

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)
