# 太空联邦学习 · 通信压缩模块文献指南

> **用途**：为 SpaceFL 项目新增「通信压缩」板块提供 2024–2026 论文地图、精读清单与代码落地映射。
>
> **调研依据**：本项目已收录 *Bringing Federated Learning to Space* (arXiv:2511.14889) 指出通信开销是星上 FL 的首要瓶颈；本文档在其 31 篇参考文献之外，补充检索 2024–2026 年最新发表的通信压缩（量化 / 稀疏化 / 过空计算 / 单轮通信 / 剪枝）方向论文。

---

## 0. 一页速览（TL;DR）

| 你的现状 | 缺口 | 本文档推荐的第一梯队 |
|---------|------|---------------------|
| ✅ FedBuff（异步缓冲） | ⚠️ 通信压缩 | **[DFedSat](https://arxiv.org/abs/2407.05850)**（去中心化+压缩+鲁棒）· [渐进权重量化 (IEEE TMC 2024)](https://ieeexplore.ieee.org/abstract/document/10415259) |
| ✅ FedLEO（去中心化） | ⚠️ 压缩未接入 ISL | [OTA 异步 FL (IEEE TWC 2024)](https://ieeexplore.ieee.org/abstract/document/10746330) · [NomaFedHAP](https://arxiv.org/abs/2401.00685) |
| ✅ MNIST/CIFAR-10 实验框架 | ⚠️ 压缩-精度 trade-off 评测 | [One-Shot FL (90分钟收敛)](https://ar5iv.labs.arxiv.org/html/2305.12316) · [FedX 剪枝](https://arxiv.org/abs/2508.06256) |

**核心结论**：太空 FL 的通信压缩已从「通用 FL 压缩算法的搬运」走向「面向 ISL 时变拓扑的专门设计」。落地顺序建议：**量化（QSGD/渐进量化）→ 误差反馈 → Top-k 稀疏化 → 压缩感知调度（按链路质量动态调比特）**。

---

## 1. 通信压缩技术谱系（在太空场景下的分类）

| 类别 | 代表技术 | 太空场景适配点 | 典型压缩比 |
|------|---------|---------------|-----------|
| **量化 (Quantization)** | QSGD、低比特定点、渐进量化 | 星上计算资源受限，定点/低比特天然友好；ISL 带宽小 | 4–32× |
| **稀疏化 (Sparsification)** | Top-k、随机掩码、剪枝 | 遥感模型参数冗余度高；配合误差反馈可保精度 | 10–100× |
| **过空计算 (Over-the-Air, AirComp)** | 模拟波束成形聚合 | 利用无线信道的叠加特性，聚合与传输一体，无需逐比特传输 | 聚合即通信，O(1) 轮内开销 |
| **单轮/少轮通信 (One-shot)** | One-shot FL、FedSat 类异步 | 星地接入窗口极短，减少轮次比压缩单轮更彻底 | 轮次 10–100× 减少 |
| **结构化压缩** | 低秩分解、共享随机种子 | 与星上轻量模型结合 | 10–50× |
| **语义/任务导向通信** | 语义编码、FSO OTA | 6G 星地一体化的前沿方向 | 视任务而定 |

> 注：多数论文组合使用多种技术（如 DFedSat = 去中心化 + 量化 + 鲁棒性；OTA 论文 = 异步 + 过空计算）。

---

## 2. Tier 1 —— 通信压缩专项（必读，直接支撑新板块）

### [C1] DFedSat: Communication-Efficient and Robust Decentralized FL for LEO Satellite Constellations ⭐ 重点

| 项目 | 内容 |
|------|------|
| arXiv | [2407.05850](https://arxiv.org/abs/2407.05850)（v1 2024-07）· [HTML 版](https://arxiv.org/html/2407.05850v1/) · [ar5iv](https://ar5iv.labs.arxiv.org/html/2407.05850) |
| 期刊版 | IEEE（[IEEE Xplore Doc 11271435](https://ieeexplore.ieee.org/abstract/document/11271435)，2025） |
| 一句话 | 在 LEO 星座 ISL 上做**拓扑感知的量化压缩去中心化 FL**，兼顾通信高效与对时变拓扑/链路中断的鲁棒性。 |

- **为什么是首选**：它是「去中心化（对标你的 FedLEO）+ 通信压缩 + 鲁棒性」三者结合最完整的 2024–2025 工作；方法上与项目已有 `FedLEOAggregator`（面内→面外分层聚合）直接对接。
- **要点**：动态拓扑建模（轨道确定性可预测）→ 拓扑感知混合矩阵（按数据集大小加权）→ ISL 量化压缩 + 误差控制 → 收敛性分析。
- **详细中文分析见**：[`[23]_DFedSat2025_通信高效鲁棒去中心化FL_中文分析.md`]([23]_DFedSat2025_通信高效鲁棒去中心化FL_中文分析.md)

### [C2] Communication-Efficient Satellite-Ground FL Through Progressive Weight Quantization

| 项目 | 内容 |
|------|------|
| 期刊 | IEEE Transactions on Mobile Computing, 2024（[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10415259)）· [ACM DL 镜像](https://dl.acm.org/doi/10.1109/TMC.2024.3358804) |
| 作者 | Yang & Yuan（以期刊页为准） |
| 一句话 | 星-地链路采用**渐进式权重量化**：随训练轮次逐步降低量化位宽，在有限接入窗口内最大化有效更新量。 |

- **为什么推荐**：与你的星地异步范式（FedBuff + `CommunicationScheduler`）天然互补——量化位宽可作为一个新的调度维度（如按剩余接入窗口调整）。
- **落地点**：在 `fl_space/fl/fedbuff.py` 的客户端上传路径前加 `ProgressiveQuantizer`，位宽随轮次递减。

### [C3] Asynchronous Federated Learning via Over-the-Air Computation in LEO Satellite Networks

| 项目 | 内容 |
|------|------|
| 期刊 | IEEE Transactions on Wireless Communications, 2024（[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10746330)）· DOI [10.1109/TWC.2024.3487986](https://dl.acm.org/doi/abs/10.1109/TWC.2024.3487986) |
| 作者 | Huang & Li（以期刊页为准） |
| 一句话 | 用**过空计算 (AirComp)** 在 LEO 星座中实现异步 FL：利用多址信道叠加特性把「传输+聚合」合成一步，规避逐比特下行带宽瓶颈。 |

- **为什么推荐**：OTA 是压缩的极端形态（聚合在物理层完成），是 2024–2026 星上 FL 的热点方向；可作为「通信压缩板块」的远期展望章节。

### [C4] Communication-Efficient FL for LEO Satellite Networks Integrated with HAPs Using Hybrid NOMA-OFDM（NomaFedHAP）

| 项目 | 内容 |
|------|------|
| arXiv | [2401.00685](https://arxiv.org/abs/2401.00685)（[HTML 版](https://arxiv-org.ezproxy.obspm.fr/html/2401.00685v2)） |
| 一句话 | LEO 星座 + 高空平台 (HAP) 混合组网，用 **NOMA-OFDM 混合接入 + 量化压缩**提升多用户接入效率与通信效率。 |

- **为什么推荐**：高空平台/气球等中继是星地通信的常见补充；NOMA 的多用户叠加思路可在调度器层面与现有 `ContactMatrix` 结合。

### [C5] Communication-Efficient Learning for Satellite Constellations

| 项目 | 内容 |
|------|------|
| arXiv | [2511.20220](https://arxiv.org/abs/2511.20220)（2025-11，作者 Tudose & Grüss）· [ar5iv](https://ar5iv.labs.arxiv.org/html/2511.20220) |
| 一句话 | 2025 年末最新工作：面向星座的通信高效 FL，采用带辅助变量更新的优化框架（AR5IV 显示为交替方向类更新）。 |

- **为什么推荐**：这是检索到的最新的「星座通信高效 FL」工作之一，用于保持板块的前沿性（2026 视角）。

### [C6] FedX: Explanation-Guided Pruning for Communication-Efficient Federated Learning in Remote Sensing

| 项目 | 内容 |
|------|------|
| arXiv | [2508.06256](https://arxiv.org/abs/2508.06256)（2025-08）· [IEEE Xplore Doc 11398102](https://ieeexplore.ieee.org/abstract/document/11398102) · [ar5iv](https://ar5iv.labs.arxiv.org/html/2508.06256) |
| 一句话 | 用**可解释性（解释图）引导的结构化剪枝**压缩遥感 FL 通信——稀疏化的结构化版本，比 Top-k 更适合卷积网络。 |

- **为什么推荐**：项目实验基于 MNIST/CIFAR-10（CNN 模型），结构化剪枝可直接复用现有 `models.py` 中的模型结构。

### [C7] One-Shot Federated Learning for LEO Constellations that Reduces Convergence Time from Days to 90 Minutes

| 项目 | 内容 |
|------|------|
| arXiv | [2305.12316](https://arxiv.org/abs/2305.12316)（[ar5iv](https://ar5iv.labs.arxiv.org/html/2305.12316)） |
| 一句话 | **单轮 FL**：卫星只在一次接入窗口内上传，把收敛时间从数天压缩到 90 分钟。 |

- **为什么推荐**：属于「减少通信轮次」这一与压缩正交但目标一致的路线；单轮结果可作为压缩模块的精度上界参考。

### [C8] FedSpace: An Efficient Federated Learning Framework at Satellites and Ground Stations

| 项目 | 内容 |
|------|------|
| arXiv | [2202.01267](https://arxiv.org/abs/2202.01267)（So & Hsieh 等） |
| 一句话 | 早期星-地 FL 框架基线：地面站聚合 + 星上训练，含通信效率设计。 |

- **为什么推荐**：作为对比基线（与你已有的 Razmi2022 地面辅助系列互为印证），并参考其框架设计。

---

## 3. Tier 2 —— 去中心化与拓扑（与 FedLEO 衔接，压缩的承载场景）

| 编号 | 论文 | 出处 | 与本项目关系 |
|------|------|------|-------------|
| [D1] | **DSFL: Decentralized Satellite FL for Energy-Aware LEO Constellation Computing**（Wu & Zhu） | [Semantic Scholar](https://www.semanticscholar.org/paper/DSFL%3A-Decentralized-Satellite-Federated-Learning-Wu-Zhu/e5af02b544c83cf9c24becba237c425935ee13bc) | 去中心化 + 能量感知：压缩在降低能耗上的收益可引用 |
| [D2] | **Adaptive Satellite-to-Device Association Based Asynchronous Federated Edge Learning in STIN**（Zhang & Meng） | [Semantic Scholar](https://www.semanticscholar.org/paper/Adaptive-Satellite-to-Device-Association-Based-Edge-Zhang-Meng/debe83f5eb97ff7501b1bd82d5d75d2dc702eb3a) | 星地一体化网络 (STIN) 异步 FL，补充星地侧压缩场景 |
| [D3] | **Topology-Aware Routing for FL Over Multi-Layer Satellite Networks** | [IEEE Doc 10978815](https://ieeexplore.ieee.org/abstract/document/10978815)（2025） | 多层星座拓扑感知路由：压缩后的模型更新如何选路传输 |
| [D4] | **On-Board FL for Satellite Clusters With Inter-Satellite Links** | [IEEE Doc 10409275](https://ieeexplore.ieee.org/document/10409275)（2024） | 星簇内 ISL FL 的工程实现参考（IEEE TVT 系列） |
| [D5] | **Brain-Inspired Decentralized Satellite Learning in Space Computing Power Networks** | [arXiv 2501.15995](https://ar5iv.labs.arxiv.org/html/2501.15995)（2025-01） | 空间算力网络中的去中心化学习，引用 DFedSat 等压缩工作 |
| [D6] | **A Semi-Supervised FL Framework with Hierarchical Clustering Aggregation for Heterogeneous Satellite Networks** | [arXiv 2507.22339](https://ui.adsabs.harvard.edu/abs/2025arXiv250722339L/abstract)（2025-07） | 异构星座下的分层聚类聚合，减少无效通信 |

---

## 4. Tier 3 —— 6G 新范式与前沿（前瞻章节素材）

| 编号 | 论文 | 出处 | 亮点 |
|------|------|------|------|
| [N1] | **OptiVote: Non-Coherent FSO Over-the-Air Majority Vote for Communication-Efficient Distributed FL in Space Data Centers** | [arXiv 2512.24334](https://arxiv.org/pdf/2512.24334.pdf)（2025-12） | 空间数据中心 + **自由空间光 (FSO) 过空多数投票**：2025 年末最前沿，把 OTA 推进到光通信域 |
| [N2] | **Cognitive Semantic Augmentation LEO Satellite Networks for Earth Observation** | [arXiv 2410.21916](https://export-test.arxiv.org/pdf/2410.21916v1)（2024-10） | 语义通信 + 对地观测：任务导向压缩，遥感场景直接相关 |
| [N3] | **FEDGE: Federated Learning at the Edge on Space Platforms** | [Springer IJIT 2025](https://link.springer.com/article/10.1007/s41870-025-03010-0)（DOI 10.1007/s41870-025-03010-0） | 星上边缘 FL 框架，含资源受限约束下的模型部署 |
| [N4] | **FedSat-LAM: Enabling Large AI Models on Resource-Constrained Satellites via Hierarchical FL** | [ACM (DOI 10.1145/3737902.3768355)](https://dl.acm.org/doi/10.1145/3737902.3768355) | 星上大模型 + 分层 FL：压缩对「模型越来越大」趋势的必要性 |
| [N5] | **SatelliteEdgeNet: Secure Edge-Aware FL for Satellite Imagery** | [ScienceDirect 2026](https://www.sciencedirect.com/science/article/pii/S235214652600270X) | 安全 + 边缘感知：压缩与隐私（差分隐私/加密）结合的示例 |

---

## 5. Tier 4 —— 基线、综述与已有文献（对照锚点）

| 论文 | 出处 | 角色 |
|------|------|------|
| **Bringing Federated Learning to Space**（Kim, Svoboda, Lane） | arXiv [2511.14889](https://arxiv.org/abs/2511.14889)（**已收录** `文献/2511.14889v1.pdf`） | 项目综述锚点：768 星座配置验证 FedAvg/FedProx/FedBuff；明确指出通信压缩是待补空白 |
| **Federated Learning in Satellite Constellations**（Matthiesen 2023） | IEEE Network 2023（**已收录**） | 太空 FL 综述，通信开销挑战的权威出处 |
| **FedSat**（Chowdhury & Kim 2022, "Federated Learning for Distributed Sensing in LEO Satellite Networks"） | IEEE 2022（被 [NomaFedHAP](https://arxiv.org/abs/2401.00685) 等引用为 [13]） | 星上 FL 的异步聚合早期基线 |
| **FedLEO**（Zhai 2024） | IEEE TMC 2024（**已收录**） | 去中心化基座，压缩模块的宿主 |
| **FedBuff**（Nguyen 2022） | NeurIPS 2022（**已收录**） | 异步缓冲基座，压缩的星地侧宿主 |

---

## 6. 落地映射：压缩模块如何接入 SpaceFL 现有架构

### 6.1 建议新增模块

```
fl_space/compression/
├── __init__.py            # 导出统一接口 compress/decompress
├── base.py                # Compressor 抽象基类（compress/decompress/compression_ratio）
├── quantized.py           # QuantizedCompressor：QSGD / 低比特定点 / 渐进位宽（对标 [C2]）
├── sparse.py              # SparseCompressor：Top-k 稀疏化 + 随机掩码（对标 [C6] 结构化变体）
├── error_feedback.py      # ErrorFeedback 包装器：压缩误差本地累积回补（对标 [C1] 误差控制）
└── otc.py                 # （远期）Over-the-Air 模拟聚合占位（对标 [C3][N1]）
```

### 6.2 接入点（基于现有代码定位）

| 现有组件 | 文件 | 压缩接入方式 |
|---------|------|-------------|
| FedLEO 分层聚合 | `fl_space/fedleo/aggregator.py` | 在 `aggregate()` 的输入 `local_weights_list` 之前对每份权重做 `compress()`，聚合后 `decompress()`；量化等级可随链路窗口动态调整（DFedSat 思路） |
| ISL 窗口模型 | `fl_space/isl/base.py` | 在 `ISLWindow` 中增加 `bandwidth_bps` / `compression_enabled` 元数据，供调度器决策 |
| 异步缓冲聚合 | `fl_space/fl/fedbuff.py` | 客户端上传前套 `ErrorFeedback + QuantizedCompressor`（渐进位宽 [C2] 思路） |
| 通信调度 | `fl_space/fl/scheduler.py` | 新增「压缩感知调度」：剩余接入窗口短 → 提高量化比；链路质量差 → 降级为 Top-k |
| 时间模型 | `fl_space/fl/time_model.py` | 将「传输字节数」按压缩比折算，量化通信时间收益（对齐 [C7] 90 分钟式结论） |
| 实验框架 | `fl_space/fedleo/experiment.py`、`fl_space/fl/runner.py` | 新增 CLI 参数 `--compression {none,qsgd,topk,progressive}`、`--compression-ratio` |

### 6.3 评测指标建议

1. **通信量**：总传输字节数 / 压缩比（与收敛精度画 trade-off 曲线，对标 DFedSat 与渐进量化的实验设计）；
2. **收敛性**：固定压缩比下的最终精度与收敛轮数（对比 `none` 基线）；
3. **星地场景**：在既有 `ContactMatrix` 调度下统计「相同精度所需接入窗口数」的减少；
4. **鲁棒性**：注入链路中断/丢包（ISL 窗口随机剔除），对比有无误差反馈的精度回退（DFedSat 鲁棒性实验思路）。

---

## 7. 获取方式说明

- 本环境**无法直接联网下载 PDF**，上述条目均已给出 arXiv / IEEE Xplore / 出版社链接；建议通过以下方式补全 PDF 并放入 `文献/` 目录（沿用现有命名：`[编号]_作者年份_主题.pdf`）：
  1. arXiv：`https://arxiv.org/pdf/<ID>`（如 `https://arxiv.org/pdf/2407.05850`）；
  2. IEEE Xplore：机构订阅下载（文档号见各条目）；
  3. 镜像：ar5iv / X-MOL / Semantic Scholar 开放页。
- 已收录在本目录的 PDF：`2511.14889v1.pdf`（Bringing FL to Space）、`FedLEO_*.pdf`、`Ground-Assisted_*.pdf`、`2206.00307v1.pdf` 等，可与本文档 Tier 4 对照使用。

---

## 8. 快速行动清单（推荐阅读顺序）

1. [ ] 精读 **[C1] DFedSat**（中文分析已就绪）→ 确立「去中心化+压缩+鲁棒」框架；
2. [ ] 精读 **[C2] 渐进权重量化** → 实现第一个 `QuantizedCompressor`（渐进位宽）；
3. [ ] 泛读 **[C7] One-Shot** 与 **[C6] FedX** → 明确压缩比的实验区间；
4. [ ] 浏览 **[C3][N1] OTA** 系列 → 写入板块的「远期展望」；
5. [ ] 代码落地：按 §6 新增 `fl_space/compression/` 并接入 `FedLEOAggregator` 与 `fedbuff.py`。

---

*整理：基于 2024–2026 公开论文检索。所有条目均附原始链接，论文细节以官方版本为准。*
