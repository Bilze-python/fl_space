# 太空联邦学习参考文献阅读指南

> 来源：论文 *"Bringing Federated Learning to Space"* (arXiv:2511.14889v1)
> 
> 作者：Grace Ra Kim (Stanford), Filip Svoboda, Nicholas D. Lane (Cambridge)
> 
> 从31篇参考文献中筛选了对 SpaceFL 项目最有价值的文献，按优先级排列。

---

## Tier 1: 太空FL核心文献（必读，4篇）

这些是与我们项目最直接相关的太空联邦学习论文。

| # | 文件 | 核心贡献 |
|---|------|---------|
| [11] | `Matthiesen2023_FL_in_Satellite_Constellations.pdf` | **太空FL综述** — 首次系统阐述FL在卫星星座中的挑战与机遇。IEEE Network 2023 |
| [20] | `Razmi2022_Ground_Assisted_FL_LEO.pdf` | **地面辅助FL** — 提出地面站辅助的LEO卫星FL架构，分析接入窗口对训练的影响。IEEE WCL 2022 |
| [21] | `Razmi2022_Scheduling_FL_LEO.pdf` | **通信调度优化** — 利用确定性轨道优化FL通信调度。EUSIPCO 2022 |
| [22] | `Zhai2024_FedLEO.pdf` | **去中心化FL** — FedLEO框架，利用星间链路实现去中心化联邦学习。IEEE TMC 2024 |

### 建议阅读顺序
1. **[11] Matthiesen 2023** — 先看综述，建立全局认知
2. **[20] Razmi 2022** — 理解地面辅助FL的基本范式
3. **[21] Razmi 2022** — 深入调度优化
4. **[22] Zhai 2024** — 了解去中心化方案（对标 FedBuff）

### 与 SpaceFL 的对应关系
| 文献概念 | SpaceFL 实现 |
|---------|-------------|
| 地面站辅助聚合 | CommunicationScheduler + ContactMatrix |
| 确定性轨道调度 | `_advance_to_next_contact()` |
| 通信窗口受限 | `get_next_contact()` / `get_connected_sats()` |
| 星间链路 | 预留接口 (Intra-satellite communication) |

---

## Tier 2: 星上ML + 通信（重点参考，2篇）

| # | 文件 | 核心贡献 |
|---|------|---------|
| [10] | `Ruzicka2023_RaVEN_Onboard_Training.pdf` | **RaVÆN** — 首次在轨模型训练演示（ION SCV004卫星），无监督变化检测 |
| — | Mateo-Garcia 2023 (未下载) | **WorldFloods** — 在轨可重训练ML载荷，证明星上重训练可行性。Scientific Reports 2023 |

### 关键启示
- 星上训练已通过飞行验证（RaVÆN, WorldFloods, OPS-SAT）
- 模型规模受星载计算资源限制（当前为轻量级模型）
- FL可以解决单星训练的局限性 → 这正是 SpaceFL 的定位

---

## Tier 3: FL基础算法（经典必读，3篇）

这些是FL领域的奠基性论文，SpaceFL 直接实现了其中三种算法。

| # | 文件 | 算法 | SpaceFL 实现 |
|---|------|------|-------------|
| [15] | `McMahan2017_FedAvg.pdf` | **FedAvg** | `fl_space/fl/fedavg.py` |
| [18] | `Li2020_FedProx.pdf` | **FedProx** | `fl_space/fl/fedprox.py` |
| [19] | `Nguyen2022_FedBuff.pdf` | **FedBuff** | `fl_space/fl/fedbuff.py` |

### 阅读要点
- **[15] FedAvg**: Section 3 的算法伪代码 → 对应 `FixedEpochTrainer` + `SyncWeightedAggregator`
- **[18] FedProx**: proximal term（μ‖w−w_global‖²）→ 对应 `ProximalTrainer`
- **[19] FedBuff**: FIFO缓冲区 + staleness降权 → 对应 `BufferAggregator`

---

## Tier 4: 工具与数据集（3篇）

| # | 文件 | 用途 |
|---|------|------|
| [28] | `Beutel2020_Flower_Framework.pdf` | **Flower FL框架** — 业界主流FL框架，我们的四组件架构参考了其设计理念 |
| [29] | `Caldas2018_LEAF_Benchmark.pdf` | **LEAF基准** — FL标准评测基准（FEMNIST/Shakespeare等），论文中使用的FEMNIST由此而来 |
| [30] | `Cohen2017_EMNIST.pdf` | **EMNIST数据集** — 论文实验使用的数据集，MNIST扩展版（含手写字母） |

---

## Tier 5: 未下载文献（2篇，建议通过机构订阅获取）

| # | 文献 | 获取方式 |
|---|------|---------|
| [13] | Mateo-Garcia et al. "In-orbit retrainable ML payload" (Scientific Reports 2023) | Nature OA: https://doi.org/10.1038/s41598-023-37436-0 |
| [08] | Denby et al. "Computational bottleneck in space" (ACM ASPLOS 2023) | ACM DL: https://doi.org/10.1145/3582016.3582044 |

---

## 其他相关文献（论文引用但未下载）

| # | 主题 | 建议查阅场景 |
|---|------|-------------|
| [7] | Larson & Wertz, *Space Mission Analysis and Design* | 轨道设计基础知识 |
| [12] | Ghasemi et al. "Onboard processing of hyperspectral imagery" | 星上数据处理综述 |
| [14] | Meoni et al. "OPS-SAT case" | 星上ML竞赛验证 |
| [23][24] | Elmahallawy et al. "FedHAP" | 高空平台辅助FL |
| [31] | Flordal et al. "SpaceCloud" | 星载云计算演示 |

---

## 对 SpaceFL 项目的启示

### 已验证方向（论文结论支持）
1. **三种算法都可行**：论文验证了 FedAvg/FedProx/FedBuff 在768种星座配置下均可达到 >80% 准确率
2. **地面站数量是瓶颈**：1-2个地面站 → 训练需数月；5+个地面站 → 可缩短至数天
3. **调度优化收益巨大**：最多 9× 加速（3个月 → 10天）
4. **FedBuff 优势明显**：异步模式下空闲时间几乎为零

### 可探索方向（论文指出的空白）
1. **星间链路 (ISL)** — 论文验证了 ISL 能显著减少轮次时间，我们在 scheduler 中预留了接口
2. **多星簇协同** — 增加簇内卫星数比增加星簇数更有效
3. **地面站最优部署** — 超过5个地面站后收益递减，地理分布比数量更重要

### 待补充到 SpaceFL 的功能
- [ ] EMNIST/FEMNIST 数据集支持（目前只有 MNIST/CIFAR-10）
- [ ] ISL 星间链路通信模型
- [ ] 更多地面站部署策略（极地站、赤道站的最优配比）
- [ ] Staleness-aware 调度（FedBuff 已有基础）
