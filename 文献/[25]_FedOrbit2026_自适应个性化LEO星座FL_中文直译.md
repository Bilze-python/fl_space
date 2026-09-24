# [25] FedOrbit：面向非 IID 低轨卫星星座的自适应个性化联邦学习

> **原文标题**: FedOrbit: Adaptive Personalized Federated Learning for Non-IID LEO Satellite Constellations
> **作者**: Satwat Bashir, Tasos Dagiuklas, Muddesar Iqbal
> **机构**: School of Engineering, London South Bank University, London, UK
> **邮箱**: {bashis11, tdagiuklas, m.iqbal}@lsbu.ac.uk
> **arXiv**: 2608.09687v1 [cs.LG] 10 Aug 2026
> **资助**: 欧盟 Horizon Europe MSCA 计划（HORIZON-MSCA-2024-SE-01-01），资助协议号 101236523（AeroNet 项目）
> **翻译说明**: 全文直译，保留原文章节编号、公式编号、图表编号与参考文献编号。表 I、表 II 数据按原文完整转录。
> **附**: 文末附有对关键机制与"困难参数"的专门解释（见附录 A），以及与 SpaceFL / fl_space 项目的映射（附录 B）。

---

## 摘要

低地球轨道（LEO）卫星星座中的联邦学习（FL）同时受到**非 IID 数据**与**不规则地面站可见性**的影响，而这两者都由**轨道几何**驱动。当**轨道级别的类别分布互不相交**时，全局聚合表现很差；而当这些分布**相互重叠**时，强个性化又可能**过度**。我们提出 **FedOrbit**，它结合了：基于**星间链路（ISL）**的**连续轨道级训练**、**类别感知的分层聚合**、带**回归率阻尼（return-rate dampening）**的**质量加权特征聚合**，以及基于**轨道间类别相似度**的**自适应特征分解**。在三个遥感基准与两种非 IID 划分下，FedOrbit 在**六种设置中的五种取得最高准确率**，在第六种中与最佳结果的差距在 **0.9 个百分点**以内。相对最强基线的增益在 **Dirichlet 划分下达到 16.1 个百分点**，在 **pathological 划分下达到 8.6 个百分点**，并在六种设置中的五种取得**最小的逐轨道准确率离散度**。

**关键词**：联邦学习，LEO 卫星星座，个性化聚合，非 IID 数据，非地面网络。

---

## I. 引言

低地球轨道（LEO）卫星星座如今是全球地球观测的重要平台 [1], [2]，而**星上处理**使得直接在卫星上训练与运行机器学习模型成为可能 [3], [4]。然而，鉴于星地链路的间歇性、数据隐私要求与带宽约束，将原始观测数据传输到地面站进行集中式训练是**不切实际的**。联邦学习（FL）[5] 减少了对原始数据传输的需求：每颗卫星在自身观测数据上本地训练，仅与地面站（GS）交换模型参数。

**LEO 星座中的 FL 与地面场景有本质区别。** 在 Walker Delta 星座中，每个轨道沿一条独立的地面轨迹运行，观测不同的地理区域，从而产生**轨道级别固有非 IID 的类别分布** [6], [7]。轨道到 GS 的可见性由**同一套轨道几何**决定，从而导致**参与不均衡**。在本文考虑的星座中，约 **47%** 的训练轮次**没有任何轨道对 GS 可见**，且**最少被访问的轨道参与轮数不到最多被访问轨道的三分之一** [8]。

由于一个轨道的类别分布与其地面轨迹相关联，而其可见性时间表由轨道几何决定，**类别贡献与时序是系统性地相关的，而非独立的**。这种**数据异构性与参与度之间的相关性**，正是 LEO FL 场景的**定义性特征**。

本文提出 **FedOrbit**，一个个性化 FL 框架，用于解决 LEO 星座中数据与参与度之间的相关性。贡献如下：

(i) **识别出该相关性引发的两种失效模式**，即 pathological 划分下的**全局模型崩溃**与 Dirichlet 划分下的**过度个性化**；

(ii) **基于 ISL 的连续轨道级训练**，使每个轨道在每一轮都精炼模型；

(iii) **类别感知的分层聚合**，按**逐类数据归属**路由分类器更新；

(iv) **带回归率阻尼的质量加权特征聚合**，限制某个从长期缺席中回归的轨道的影响力；

(v) **自适应特征分解**，以**轨道间类别相似度**决定的速率，将每个轨道的个性化特征提取器向全局方向混合。

我们在三个遥感分类基准上，在 pathological 与 Dirichlet（$\alpha = 0.5$）两种划分下评估 FedOrbit，在现有方法表现最差的划分上取得最大增益。

---

## II. 相关工作

**非 IID 数据下的 FL。** FedAvg [5] 在异构分布下会退化 [9]；FedProx [10]、SCAFFOLD [11] 与 FedNova [12] 分别使用近端正则化、控制变量或本地步数归一化。这些方法产生**单一全局模型**，但**没有建模本地数据与依赖可见性的参与度之间的耦合**。

**个性化 FL。** Ditto [13] 添加向全局模型的 $\ell_2$ 惩罚；APFL [14] 对本地与全局模型做插值；其他方法使用正则化内循环 [15]、元学习初始化 [16] 或分割表示 [17]。这些方法**没有考虑**由依赖可见性的参与度所导致的**全局参考模型的偏置**。

**非地面网络中的 FL。** 此前的 LEO FL 工作已研究了不规则可见性下的调度 [8]、半异步聚合 [18]、空中参数服务器 [19], [20]、去中心化轨道内聚合 [21]、过空计算 [22]、系统异构性 [23] 与去中心化个性化 [24]。其共同焦点是**通信架构、收敛速度与调度**。据我们所知，此前工作**没有同时**处理轨道级非 IID 数据与依赖可见性的参与度，也**没有**让个性化自适应于轨道间相似度。

---

## III. 系统模型

### A. LEO 星座与可见性模型

我们考虑一个 Walker Delta 星座，参数沿用此前 FL-over-LEO 工作 [8]：

$$L = 5 \text{ 个轨道}, \quad N_l = 4 \text{ 颗卫星/轨道（共 20 颗卫星）}, \quad \text{高度 } h = 550 \text{ km},$$
$$\text{倾角 } i = 53^\circ, \quad \text{GS 纬度 } \phi_g = 51^\circ\text{N}, \quad \text{最小仰角 } \alpha_e = 10^\circ, \quad \text{轮时长 } \Delta t = 5 \text{ min}.$$

GS 充当**参数服务器**。

每颗卫星沿圆形开普勒轨道运行，平均角速度 $n_o = \sqrt{\mu/(R_E + h)^3}$，其中 $\mu = 3.986 \times 10^5 \ \text{km}^3/\text{s}^2$ 是地球引力参数，$R_E = 6371$ km 是地球平均半径。轨道 $l$ 的**升交点赤经（RAAN）**为 $\Omega_l = 2\pi l/L$（$l = 0, \dots, L-1$），从而轨道等间隔分布。GS 在**地心惯性系（ECI）**中的位置随地球以角速度 $\omega_E = 7.292 \times 10^{-5}$ rad/s 旋转。

轨道 $l$ 中的卫星 $k$ 对 GS 可见，当且仅当

$$\varepsilon_k(t) = \arcsin\left( \frac{(r_k(t) - r_g(t)) \cdot \hat{r}_g(t)}{\| r_k(t) - r_g(t) \|} \right) \ge \alpha_e, \tag{1}$$

其中 $r_k(t)$ 与 $r_g(t)$ 分别是卫星与 GS 在 ECI 系中的位置，$\hat{r}_g(t)$ 是 GS 处的**本地天顶单位向量**。只要某个轨道至少有一颗卫星满足式 (1)，就**将该轨道视为在轮 $t$ 可见**，因为上传到 GS 的是**轨道级模型**。

用二值矩阵 $V \in \{0,1\}^{T \times L}$ 记录可见性：若轨道 $l$ 在轮 $t$ 可见则 $V_{t,l} = 1$，否则为 0；轮 $t$ 的可见轨道集合为 $\mathcal{V}(t) = \{l : V_{t,l} = 1\}$。**所有被评估的方法使用同一个 $V$。**

对于 $T = 400$：约 **47%** 的轮次无可见轨道；**逐轨道可见份额**分别为 **15.2%、10.2%、8.5%、5.2%、14.0%**；**最长的无可见轨道连续区间**为 $\tau_{\max} = 191$ 轮。

> **图 1：FedOrbit 架构。** 卫星本地训练，在轨道内通过 ISL 聚合，并在可见性窗口期间将轨道级模型下行到 GS。

### B. 分层联邦学习架构

该架构包含**卫星、轨道、GS** 三层（图 1），由所有方法共享。以下基于 FedAvg 的流程被基线使用。FedOrbit 修改**两个聚合层**并通过**星间链路训练不可见轨道**，如第 V 节所述。

**本地训练。** 卫星 $k$ 持有大小为 $n_k$ 的数据集 $\mathcal{D}_k$，本地目标为 $F_k(w) = (1/n_k)\sum_{x \in \mathcal{D}_k} f(x; w)$，运行 $E$ 个 epoch 的小批量 SGD，学习率 $\eta_t = \eta_0 \gamma^t$，产生

$$w_k^t = \text{LocalSGD}\left( w^t, \mathcal{D}_k, E, \eta_t \right). \tag{2}$$

**轨道内聚合。** 轨道 $l$ 中的卫星通过短距 ISL，以**样本量加权平均**合并更新：

$$w_l^t = \sum_{k \in \mathcal{S}_l} \frac{n_k}{\sum_{j \in \mathcal{S}_l} n_j} w_k^t, \tag{3}$$

其中 $\mathcal{S}_l$ 是轨道 $l$ 中的卫星集合。

**GS 处的轨道间聚合。** 基线在可见轨道上应用 FedAvg [5]：

$$w^{t+1} = \sum_{l \in \mathcal{V}(t)} \frac{n_l}{\sum_{j \in \mathcal{V}(t)} n_j} w_l^t, \tag{4}$$

其中 $n_l = \sum_{k \in \mathcal{S}_l} n_k$。FedProx、APFL 与 Ditto 使用**相同的加权方式**，仅在其本地目标上不同。

### C. 非 IID 数据分布模型

每个轨道沿其地面轨迹采集影像，产生反映其所飞越地形的**类别分布 $P_l$**。由于同一轨道内的卫星共享同一地面轨迹，**异构性是轨道级别的，而非卫星级别的**。我们考虑两种划分方法来代表对比鲜明的**轨道间重叠程度**：

**Pathological 划分。** 每个轨道被分配一个**互不相交**的类别子集，模拟观测互不相交土地覆盖类别的轨道。在每个轨道内部，卫星分配遵循 $\text{Dirichlet}(\alpha = 0.5)$ 划分。

**Dirichlet 划分（$\alpha = 0.5$）。** 轨道间类别比例从 $\text{Dirichlet}(0.5)$ 抽取，给出**相互重叠但有偏斜**的轨道分布。轨道内分配同样遵循 $\text{Dirichlet}(0.5)$。这模拟了当轨道覆盖地理上邻近但彼此不同的区域时产生的**部分重叠**。

---

## IV. 问题建模

在第 III 节的 Walker Delta 星座中，类别分布 $P_l$ 与轨道 $l$ 的**地面轨迹**相关联，而可见性时间表 $\{V_{t,l}\}_{t=1}^{T}$ 由式 (1) 通过**轨道几何**决定。因此，**"客户端参与独立于本地数据"这一标准 FL 假设 [5], [10], [11] 并不成立**，全局模型也**无法作为个性化的无偏参考**。

令 $\{\tilde{w}_l\}_{l=1}^{L}$ 表示轨道级推理模型，$F_l$ 为 $P_l$ 上的期望损失，$\text{Acc}_l$ 为相应的准确率。FedOrbit 由以下**公平性约束目标**引导：

$$\min_{\{\tilde{w}_l\}}\ \frac{1}{L} \sum_{l=1}^{L} F_l(\tilde{w}_l) \qquad \text{s.t.} \quad \max_l \text{Acc}_l - \min_l \text{Acc}_l \le \delta. \tag{5}$$

式 (5) 表达的是**设计目标**；**公平性项在优化过程中并未被显式施加**。经验离散度在第 VII 节报告。

该建模提出四项设计要求：

- **(R1)** 减少由 $V$ 造成的**贡献不均衡**；
- **(R2)** 按**类别归属**对分类器更新加权；
- **(R3)** 使个性化**自适应于轨道间异构性**，无需人工重新调参；
- **(R4)** 保持 **GS 上行流量与 FedAvg 相当**，额外工作放在星上或轨道内部完成。

---

## V. 提出的方法：FedOrbit

FedOrbit 修改了**星上调度**、**轨道内与轨道间聚合**以及**逐轨道推理**。模型被拆分为**特征提取器**与**分类器**，$w = [w_f, w_c]$，因为特征提取器编码的是**跨星座共享的内容**，而**分类器的每一行绑定到特定类别**。

### A. 带 ISL 中继的连续轨道级训练

在标准 FL 下，**最少被访问的轨道对训练好的模型几乎没有影响**，因为它只在很小一部分轮次中参与。FedOrbit **在每一轮训练每一个轨道**，依据可见性使用两种流程。

**可见轨道。** 对于 $l \in \mathcal{V}(t)$，本地 epoch 数随**前一个不可见区间长度**增长：

$$E_l(t) = \min\left( E_{\max},\ E_{\text{base}} + s_e \tau_l(t) \right), \tag{6}$$

其中 $\tau_l(t)$ 是 $t$ 之前**连续不可见轮数**，$s_e \in (0,1)$ 缩放追赶幅度，$E_{\max}$ 封顶负载。$R_{\text{intra}}$ 个轨道内轮次中的每一个都使用 $E_l(t)$ 个本地 epoch，随后跟一个轨道内聚合步骤（第 V-B 节）。

**不可见轨道（ISL 中继）。** 对于 $l \notin \mathcal{V}(t)$，该轨道中的每颗卫星在**轨道级模型状态**上运行 $E_{\text{dark}}$ 个 epoch 的本地 SGD，之后一次轨道内聚合产生更新的轨道模型。**这一步骤只使用轨道内 ISL，不消耗 GS 上行链路。** $E_{\text{dark}}$ 保持较小（两个 epoch），使不可见期间的细化**不会覆盖更重的可见更新**。

### B. 类别感知的分层聚合

当某个类别集中在**很少可见的轨道**上时，普通平均会用**不持有该类别**的轨道的更新来**稀释对应的分类器行**。FedOrbit 在**轨道内与轨道间两个层级**都按**逐类数据归属**加权聚合。

**类别强度与类别亲和度。** 对于卫星 $k \in \mathcal{S}_l$ 与类别 $c \in \{1, \dots, C\}$：

$$s_{k,c} = \left| \{ x \in \mathcal{D}_k : y(x) = c \} \right|, \tag{7}$$

以及轨道 $l$ 对类别 $c$ 的**类别亲和度**：

$$a_{l,c} = \frac{\sum_{k \in \mathcal{S}_l} s_{k,c}}{\sum_{j=1}^{L} \sum_{k \in \mathcal{S}_j} s_{k,c}}. \tag{8}$$

**这两个量在初始化时计算一次，并在整个训练过程中保持固定。**

**轨道内聚合。** 类别 $c$ 的分类器行在轨道内卫星之间按**类别强度**平均，而特征提取器按**逐卫星更新的幅度** $\Delta w_k^f = w_k^f - w^f$ 平均：

$$w_l^c[c,:] = \sum_{k \in \mathcal{S}_l} \frac{s_{k,c}}{\sum_{j \in \mathcal{S}_l} s_{j,c}} w_k^c[c,:], \tag{9}$$

$$w_l^f = \sum_{k \in \mathcal{S}_l} \frac{\left| \Delta w_k^f \right|}{\sum_{j \in \mathcal{S}_l} \left| \Delta w_j^f \right|} w_k^f, \tag{10}$$

其中 $|\cdot|$ 是**逐元素绝对值**，式 (10) 中的比值是**逐元素**的。

**分类器的轨道间聚合。** 在 GS 处，类别 $c$ 的分类器行在全部 $L$ 个轨道上平均，权重为**类别亲和度乘以陈旧度因子**：

$$w^c[c,:] \leftarrow \frac{\sum_{l=1}^{L} a_{l,c}\, \rho^{\sigma_l(t)}\, w_l^c[c,:]}{\sum_{l=1}^{L} a_{l,c}\, \rho^{\sigma_l(t)}}, \tag{11}$$

其中 $\rho \in (0,1]$，$\sigma_l(t)$ 是自轨道 $l$ 上次更新其分类器以来的**轮数**；**$\sigma_l(t)$ 很小，因为每个轨道在每一轮都训练**。

### C. 质量加权的特征聚合与回归率阻尼

连续训练产生具有**不同训练预算**的轨道级更新：一个从长期不可见区间回归的可见轨道执行多达 $E_{\max} \cdot R_{\text{intra}}$ 个 epoch，而一个不可见轨道只执行 $E_{\text{dark}}$ 个。FedOrbit 按**每个轨道实际执行的训练量**对轨道间特征聚合加权：

$$\bar{w}^f(t) = \frac{\sum_{l=1}^{L} q_l(t)\, w_l^f}{\sum_{l=1}^{L} q_l(t)}, \tag{12a}$$

$$q_l(t) = \begin{cases} E_l(t)\, R_{\text{intra}}, & l \in \mathcal{V}(t), \\ E_{\text{dark}}, & l \notin \mathcal{V}(t). \end{cases} \tag{12b}$$

在我们实验所用的配置下（$E_{\text{base}} = 5$、$E_{\max} = 10$、$R_{\text{intra}} = 2$、$E_{\text{dark}} = 2$）：**一个可见轨道贡献权重 10 到 20，一个不可见轨道贡献权重 2。**

**回归轨道**可以在全局学习率下、在每个轨道内轮次中训练多达 $E_{\max}$ 个 epoch，产生一种**其他轨道的分类器行尚未适应**的特征漂移。为衰减这一效应，FedOrbit 按**刚刚结束的不可见区间长度**成比例地**阻尼任何重新进入可见性的轨道的本地学习率**：

$$\eta_l(t) = \frac{\eta_t}{1 + \kappa\, \tau_l(t)/\tau_{\max}}, \tag{13}$$

其中 $\tau_{\max} = 191$ 轮是最长不可见区间，$\kappa \ge 0$ 是超参数。当 $\kappa = 0.5$ 时，一个**在最长的区间之后回归**的轨道使用 $\eta_l(t) = \eta_t/1.5$，即全局值的**三分之二**；一个**在上一轮可见**的轨道（$\tau_l(t) = 0$）**不被阻尼**。该规则**仅在重新进入后的第一轮**应用。

### D. 自适应特征分解

上述聚合规则产生**单一的全局模型** $[\bar{w}^f(t), w^c(t)]$。当类别分布**重叠**时，单一全局模型是合适的；当类别分布**互不相交**时，它是**次优**的。FedOrbit 使用一个**由数据计算出的单一系数**在这两种情形之间自适应。

对每个轨道 $l$，令 $p_l \in \Delta^{C-1}$ 是其卫星上聚合得到的归一化类别直方图，$p_{l,c} = \sum_k s_{k,c} / \sum_{c',k} s_{k,c'}$。轨道 $i$ 与 $j$ 之间的 **Jensen-Shannon 距离**为

$$\text{JS}(p_i, p_j) = \sqrt{ \tfrac{1}{2} D_{\text{KL}}(p_i \| m) + \tfrac{1}{2} D_{\text{KL}}(p_j \| m) }, \tag{14}$$

其中 $m = \tfrac{1}{2}(p_i + p_j)$，$D_{\text{KL}}$ 使用 $\log_2$，因此 $\text{JS} \in [0,1]$。**轨道间相似度系数**为

$$\beta = \frac{2}{L(L-1)} \sum_{i<j} \left( 1 - \text{JS}(p_i, p_j) \right). \tag{15}$$

**该系数计算一次并固定。**

> **算法 1：FedOrbit 的一轮通信**
>
> **Require:** $V$；由 (7)、(8) 得到的 $\{s_{k,c}\}, \{a_{l,c}\}$；由 (15) 得到的 $\beta$；$\tau_{\max}$
> 1: **状态**: $w^c(t), \bar{w}^f(t), \{w_l^{f,p}(t)\}, \{\tau_l, \sigma_l\}$；计算 $\mathcal{V}(t)$
> 2: **for all** $l \in \mathcal{V}(t)$ ▷ 可见 **do**
> 3: &emsp;由 (6) 设定 $E_l(t)$；由 (13) 设定 $\eta_l(t)$；由 (17) 初始化
> 4: &emsp;**for** $r = 1, \dots, R_{\text{intra}}$ **do**
> 5: &emsp;&emsp;每个 $k \in \mathcal{S}_l$ 以 $\eta_l(t)$ 运行 $E_l(t)$ 个 epoch；经由 (9)、(10) 聚合
> 6: &emsp;**end for**
> 7: &emsp;$q_l = E_l(t) R_{\text{intra}}$；重置 $\tau_l, \sigma_l$
> 8: **end for**
> 9: **for all** $l \notin \mathcal{V}(t)$ ▷ 不可见 **do**
> 10: &emsp;由 (17) 初始化；每个 $k$ 以 $\eta_t$ 运行 $E_{\text{dark}}$ 个 epoch；经由 (9)、(10) 聚合
> 11: &emsp;$q_l = E_{\text{dark}}$；$\tau_l, \sigma_l \mathrel{+}= 1$
> 12: **end for**
> 13: **GS:** $\bar{w}^f(t) \leftarrow$ (12)；$w^c(t+1) \leftarrow$ (11)；$w_l^{f,p}(t+1) \leftarrow$ (16)
> 14: **return** $\{\tilde{w}_l(t+1)\}$ 经由 (17)

每个轨道 $l$ 维护一个**个性化特征提取器** $w_l^{f,p}$，初始化为 $w_l^{f,p}(0) = w^f(0)$，并通过一次**向全局质量加权平均的指数混合**进行更新：

$$w_l^{f,p}(t+1) = (1 - \beta)\, w_l^{f,p}(t) + \beta\, \bar{w}^f(t). \tag{16}$$

**所有轨道与所有轮次使用同一个 $\beta$。** 该轨道的推理模型是其**个性化特征提取器与全局分类器的配对**：

$$\tilde{w}_l(t) = \left[ w_l^{f,p}(t),\ w^c(t) \right]. \tag{17}$$

**较大的 $\beta$** 使个性化提取器更快地向全局方向移动；**较小的 $\beta$** 保留更多轨道特定信息。**两种划分都由同一流程、同一套超参数处理。**

### E. 通信与计算成本

每个可见轨道向 GS 上传**一个轨道级模型**，与 FedAvg 相同。FedOrbit **额外增加**了轨道内 ISL 交换与不可见期间的本地训练。这两类操作都**跨轨道并行**执行。对于一个含 $P$ 个参数的模型，FP32 模型约需 $4P$ 字节；训练还需要为梯度、动量与激活值分配内存。此处使用的**亚百万参数的 LeNet 模型**因此需要**数十兆字节**的工作内存。**实际部署需要一颗星上多核处理器，并由 FPGA、GPU 或 AI 加速器支持。**

在 GS 处，聚合与个性化模型存储随 $L$ **线性增长**，而式 (15) 中的**一次性相似度计算**复杂度为 $O(L^2 C)$。可见性矩阵 $V$ 支持时变连通性，但**更大规模与动态重构的星座尚未被评估**。这些设置与基于硬件的验证留待未来工作。

---

**表 I：训练超参数**

| 项目 | 值 | 项目 | 值 |
|---|---|---|---|
| 优化器 | SGD, 动量 0.9 | $E_{\text{base}}$ | 5 |
| 权重衰减 | $5 \times 10^{-4}$ | $E_{\max}$ | 10 |
| 梯度裁剪 | 1.0 | $E_{\text{dark}}$ | 2 |
| Batch | 64 | $s_e$ | 0.7 |
| $\eta_0$ | 0.01 | $R_{\text{intra}}$ | 2 |
| $\gamma$ | 0.998 | $\rho,\ \kappa$ | 0.95, 0.5 |

---

## VI. 实验设置

**星座与可见性。** 所有实验使用第 III-A 节的 Walker Delta 配置。可见性矩阵 $V$ 通过式 (1) 预计算，**每个方法都使用它**。每次仿真对 EuroSAT 与 So2Sat LCZ42 使用 $T := 200$ 个通信轮，对 RESISC45 使用 $T = 400$。

**数据集。**

- **EuroSAT** [25]：Sentinel-2 土地利用，10 类，64×64 RGB，27,000 样本。
- **So2Sat LCZ42** [26]：Sentinel-2 局地气候区，10 个城市类别，32×32，10 个波段，约 352,000 训练样本与 24,000 测试样本。
- **NWPU-RESISC45** [27]：航拍场景，45 类，32×32 RGB，每类 700 张图像，80/20 划分。

**非 IID 划分。** 两种划分（第 III-C 节）都被评估。测试样本按相应的划分规则分配到各轨道。

**模型与训练。** 星上模型为 **LeNet**（两个宽度 6 与 16 的 5×5 卷积层，配 2×2 最大池化，随后是全连接层 fc1、fc2、fc3，宽度分别为 120、84、$C$）。输入通道数：EuroSAT 与 RESISC45 为 3，So2Sat 为 10。**特征/分类器的切分点位于 fc1 与 fc2 之间**，**类别亲和度加权 (11) 施加于 fc3 的各行**。训练超参数（表 I）在**同一数据集内跨方法完全相同**。

**基线。** FedAvg [5] 对可见轨道模型做平均 (4)；FedProx [10] 添加近端项（$\mu = 0.01$）；APFL [14] 用一个**自适应学习的混合系数**插值 FedAvg 全局模型与一个本地轨道模型；Ditto [13] 训练一个逐轨道个性化模型，带一个向**并行训练的 FedAvg 全局模型**的 $\ell_2$ 近端项（$\lambda_{\text{Ditto}} = 0.1$）。优化器、batch size、初始学习率与调度器在**所有方法间共享**。

**评估。** 对每个轨道 $l$，$\text{Acc}_l$ 是 $\tilde{w}_l$ 在**轨道 $l$ 的测试划分**上的准确率（对 FedAvg 与 FedProx 是全局模型，对 APFL 与 Ditto 是个性化模型，对 FedOrbit 是式 (17)）。**主指标**是**逐轨道平均** $\frac{1}{L}\sum_l \text{Acc}_l$，在**最后十个被评估轮次上取平均**；**次指标**是**逐轨道离散度** $\max_l \text{Acc}_l - \min_l \text{Acc}_l$。所有方法都用 PyTorch 实现，在每次重复中使用**相同的划分与可见性矩阵**。

表 I 中的超参数**在三个数据集与两种划分下固定不变**。对异构性的自适应由式 (15) 的 $\beta$ 提供，**而非数据集特定的调参**。对 $\kappa$、$\rho$ 与 $s_e$ 更广泛的敏感性分析留待未来工作。

---

**表 II：个性化准确率（上部）与逐轨道离散度（下部）**

| 方法 | EuroSAT Dir. | EuroSAT Path. | So2Sat Dir. | So2Sat Path. | RESISC45 Dir. | RESISC45 Path. |
|---|---|---|---|---|---|---|
| **准确率** | | | | | | |
| FedAvg | 68.7 | 20.0 | 38.9 | 16.9 | 37.3 | 14.8 |
| FedProx | 68.5 | 20.3 | 39.1 | 16.9 | 38.1 | 15.4 |
| APFL | 60.9 | 74.7 | 39.9 | 40.7 | 32.3 | 54.4 |
| Ditto | 58.5 | 95.8 | 35.7 | 84.0 | 36.3 | 77.7 |
| **FedOrbit** | **76.4** | **94.9** | **50.9** | **86.9** | **54.2** | **86.3** |
| **离散度** | | | | | | |
| FedAvg | 28.4 | 78.1 | 26.1 | 68.2 | 10.0 | 53.3 |
| FedProx | 31.0 | 78.2 | 27.9 | 68.4 | 9.9 | 55.4 |
| APFL | 40.2 | 53.9 | 31.6 | 70.2 | 12.1 | 38.4 |
| Ditto | 49.0 | 9.7 | 32.6 | 17.6 | 11.6 | 17.6 |
| **FedOrbit** | **15.8** | **12.5** | **24.7** | **15.8** | **8.7** | **8.9** |

---

## VII. 结果与分析

表 II 报告了三个数据集在两种划分下的**逐轨道个性化准确率**与**逐轨道准确率离散度**。**FedOrbit 在六种设置中的五种取得最高准确率，在第六种中与最佳结果的差距在 0.9 个百分点以内。**

**Pathological 划分。** 在轨道间类别互不相交的情况下，FedAvg 与 FedProx 仅达到 **14.8% 到 20.0%**，因为全局分类器**只服务于频繁可见轨道上的类别**。APFL 通过插值有所改善，但仍被正则化向**同一个有偏锚点**靠拢。**Ditto 是最强基线**；FedOrbit 在 RESISC45 上相对 Ditto 增益 **8.6 个百分点**、在 So2Sat 上 **2.9 个百分点**，而在 EuroSAT 上**落后 Ditto 0.9 个百分点**。

**Dirichlet 划分。** 观察到**不同的模式**。Ditto 比其在 pathological 划分下的表现**低 37 到 48 个百分点**，APFL 也退化，因为**当轨道分布重叠时，强个性化带来的收益变小**。FedOrbit 在 EuroSAT、So2Sat 与 RESISC45 上相对最强基线分别增益 **7.7、11.0 与 16.1 个百分点**。自适应特征分解提供了一个一致的解释：**$\beta$ (15) 在 Dirichlet 划分下为中等（≈0.44），在 pathological 划分下很小（≈0.11）**。因此，个性化特征提取器**在 Dirichlet 划分下更快地跟踪全局平均**，**在 pathological 划分下更长时间地保留轨道特定信息**。

**逐轨道公平性。** 表 II 的离散度部分将式 (5) 中的公平性项**操作化**。**FedOrbit 在六种设置中的五种取得最小离散度**，在第六种（EuroSAT pathological）中与最小值差距在 3 个百分点以内。在 pathological 划分下，全局基线留下 **53 到 78 个百分点**的离散度，**证实了低可见性轨道的准确率很差**。

**组件解释。** 由于这些机制**联合运作**，表 II **没有隔离它们的单独效应**。固定超参数避免了数据集特定调参，而 $\beta$ 的变化与**"Dirichlet 划分下更强的共享、pathological 划分下更强的轨道特定保留"**相一致。**受控消融留待未来工作。**

---

## VIII. 结论

FedOrbit 解决了 LEO 星座中**轨道级数据异构性与依赖可见性的参与度之间的耦合**。它结合了连续轨道级训练、类别感知的分层聚合、带回归率阻尼的质量加权特征聚合，以及自适应特征分解。在三个遥感数据集上，**FedOrbit 在六种设置中的五种取得最高准确率，在第六种中与最佳结果的差距在 0.9 个百分点以内**。它也在六种设置中的五种取得**最小的逐轨道准确率离散度**。未来工作将评估各个组件的贡献、硬件部署与更大规模的动态星座。

---

## 参考文献

[1] O. Kodheli et al., "Satellite communications in the new space era: A survey and future challenges," *IEEE Commun. Surveys Tuts.*, vol. 23, no. 1, pp. 70–109, 2021. 1st Quart.

[2] I. del Portillo, B. G. Cameron, and E. F. Crawley, "A technical comparison of three low earth orbit satellite constellation systems to provide global broadband," *Acta Astronautica*, vol. 159, pp. 123–135, 2019.

[3] G. Giuffrida et al., "CloudScout: A deep neural network for on-board cloud detection on hyperspectral images," *Remote Sens.*, vol. 12, no. 14, p. 2205, 2020.

[4] G. Mateo-Garcia et al., "Towards global flood mapping onboard low cost satellites with machine learning," *Sci. Rep.*, vol. 11, p. 7249, 2021.

[5] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, "Communication-efficient learning of deep networks from decentralized data," in *Proc. 20th Int. Conf. Artif. Intell. Statist. (AISTATS)*, vol. 54, pp. 1273–1282, 2017.

[6] B. Matthiesen, N. Razmi, I. Leyva-Mayorga, A. Dekorsy, and P. Popovski, "Federated learning in satellite constellations," *IEEE Netw.*, vol. 37, no. 5, pp. 32–39, 2023.

[7] I. Leyva-Mayorga et al., "LEO small-satellite constellations for 5G and beyond-5G communications," *IEEE Access*, vol. 8, pp. 184955–184964, 2020.

[8] N. Razmi, B. Matthiesen, A. Dekorsy, and P. Popovski, "Ground-assisted federated learning in LEO satellite constellations," *IEEE Wireless Commun. Lett.*, vol. 11, pp. 717–721, Apr. 2022.

[9] Q. Li, Y. Diao, Q. Chen, and B. He, "Federated learning on non-IID data silos: An experimental study," in *Proc. 38th IEEE Int. Conf. Data Eng. (ICDE)*, pp. 965–978, 2022.

[10] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, "Federated optimization in heterogeneous networks," in *Proc. Mach. Learn. Syst. (MLSys)*, pp. 429–450, 2020.

[11] S. P. Karimireddy, S. Kale, M. Mohri, S. J. Reddi, S. U. Stich, and A. T. Suresh, "SCAFFOLD: Stochastic controlled averaging for federated learning," in *Proc. 37th Int. Conf. Mach. Learn. (ICML)*, pp. 5132–5143, 2020.

[12] J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor, "Tackling the objective inconsistency problem in heterogeneous federated optimization," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, pp. 7611–7623, 2020.

[13] T. Li, S. Hu, A. Beirami, and V. Smith, "Ditto: Fair and robust federated learning through personalization," in *Proc. Int. Conf. Mach. Learn. (ICML)*, pp. 6357–6368, 2021.

[14] Y. Deng, M. M. Kamani, and M. Mahdavi, "Adaptive personalized federated learning," arXiv preprint arXiv:2003.13461, 2020.

[15] C. T. Dinh, N. H. Tran, and T. D. Nguyen, "Personalized federated learning with moreau envelopes," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, pp. 21394–21405, 2020.

[16] A. Fallah, A. Mokhtari, and A. Ozdaglar, "Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, pp. 3557–3568, 2020.

[17] L. Collins, H. Hassani, A. Mokhtari, and S. Shakkottai, "Exploiting shared representations for personalized federated learning," in *Proc. 38th Int. Conf. Mach. Learn. (ICML)*, pp. 2089–2099, 2021.

[18] J. So, K. Hsieh, B. Arzani, S. Noghabi, S. Avestimehr, and R. Chandra, "FedSpace: An efficient federated learning framework at satellites and ground stations," arXiv preprint arXiv:2202.01267, 2022.

[19] M. Elmahallawy and T. Luo, "FedHAP: Fast federated learning for LEO constellations using collaborative HAPs," in *Proc. IEEE Int. Conf. Wireless Commun. Signal Process. (WCSP)*, pp. 1–6, 2022.

[20] M. Elmahallawy and T. Luo, "AsyncFLEO: Asynchronous federated learning for LEO satellite constellations with high-altitude platforms," in *Proc. IEEE Int. Conf. Big Data*, pp. 5478–5487, Dec. 2022.

[21] Z. Zhai, Q. Wu, S. Yu, R. Li, F. Zhang, and X. Chen, "FedLEO: An offloading-assisted decentralized federated learning framework for low earth orbit satellite networks," *IEEE Trans. Mobile Comput.*, vol. 23, pp. 5260–5279, May 2024.

[22] Y. Huang, X. Li, M. Zhao, H. Li, and M. Peng, "Asynchronous federated learning via over-the-air computation in LEO satellite networks," *IEEE Trans. Wireless Commun.*, vol. 23, pp. 19885–19897, Dec. 2024.

[23] Z. Lin, Z. Chen, Z. Fang, X. Chen, X. Wang, and Y. Gao, "FedSN: A federated learning framework over heterogeneous LEO satellite networks," *IEEE Trans. Mobile Comput.*, vol. 24, pp. 1293–1307, Mar. 2025.

[24] L. Zhao et al., "ALANINE: A novel decentralized personalized federated learning for heterogeneous LEO satellite constellation," arXiv preprint arXiv:2411.07752, 2024.

[25] P. Helber, B. Bischke, A. Dengel, and D. Borth, "EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification," *IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.*, vol. 12, no. 7, pp. 2217–2226, 2019.

[26] X. X. Zhu, J. Hu, C. Qiu, Y. Shi, J. Kang, L. Mou, H. Bagheri, M. Häberle, Y. Hua, R. Huang, L. Hughes, H. Li, Y. Sun, G. Zhang, S. Han, M. Schmitt, and Y. Wang, "So2Sat LCZ42: A benchmark data set for the classification of global local climate zones," *IEEE Geosci. Remote Sens. Mag.*, vol. 8, no. 3, pp. 76–89, 2020.

[27] G. Cheng, J. Han, and X. Lu, "Remote sensing image scene classification: Benchmark and state of the art," *Proc. IEEE*, vol. 105, pp. 1865–1883, Oct. 2017.

---
---

# 附录 A：关键机制与"困难参数"详解

> 这篇论文篇幅短（8 页），但公式密度高、符号复用多。下面按"**先讲清核心洞察 → 再逐个拆解机制 → 最后解释难懂参数**"的顺序展开。特别地，第 A.6 节专门解释 **$\beta$ 这个全文最关键的"自适应"参数**——它只用一个数字就统一了两种相反的划分场景，是这篇论文最巧妙也最容易看漏的地方。

## A.0 先抓住核心洞察：一个"相关性"

全文建立在一个观察之上，理解它，后面全部机制都会变得顺理成章：

> **在 LEO 星座中，"看到什么"和"什么时候能看到"是同一套轨道几何同时决定的。**

具体地：

- **类别分布 $P_l$** 取决于轨道 $l$ 的**地面轨迹**（飞过陆地还是海洋、森林还是城市）；
- **可见性时间表 $\{V_{t,l}\}$** 取决于轨道 $l$ 的**轨道几何**（通过式 (1) 的仰角条件）。

两者都由轨道决定，因此**系统性地相关**。论文的原话是：

> "class contribution and timing are **systematically correlated rather than independent**. This correlation between data heterogeneity and participation is the **defining property** of the LEO FL setting."

**为什么这很致命**：标准 FL 的收敛分析 [5], [10], [11] 普遍假设 **"客户端参与独立于本地数据"**（participation independent of local data）。在这个假设下，全局模型是全体数据的**无偏代表**，因此可以作为个性化的**可信锚点**。

而在 LEO 中这个假设**破了**：如果**持有类别 A 的轨道恰好很少可见**，那么全局模型就**几乎没见过类别 A**。此时：

- 想用全局模型直接推理 → **类别 A 的准确率极差**；
- 想用全局模型作为个性化的起点/正则锚点 → **锚点本身就是有偏的**，Ditto/APFL 这类方法的正则化会把模型**拉向一个错误的参照**。

论文在摘要里把这一点的**双向后果**总结得很精炼：

| 情形 | 后果 |
|---|---|
| 轨道类别分布**互不相交**（pathological） | 全局聚合**崩溃**（global-model collapse）—— 全局分类器只服务高频可见轨道的类别 |
| 轨道类别分布**相互重叠**（Dirichlet） | **过度个性化**（over-personalisation）—— 本来共享信息更有利，却过度拟合本地 |

**注意这里有一个反直觉之处，也是本文最漂亮的论点**：个性化方法（Ditto、APFL）在 **pathological** 划分下表现**很好**（Ditto 在 EuroSAT 上 95.8%），而在 **Dirichlet** 划分下表现**很差**（Ditto 掉到 58.5%）。

**为什么？** 因为：
- pathological 下，全局模型**没用**（只见过部分类别）→ 越"不听全局的"越对 → **强个性化取胜**；
- Dirichlet 下，全局模型**有用**（各类别都有一定覆盖）→ 应当**多共享** → 此时强个性化反而**浪费了共享信息的价值**。

这就解释了**为什么需要"自适应"**：**不存在一个对所有划分都最优的个性化强度**。而 FedOrbit 用一个**从数据算出来的标量 $\beta$** 自动切换——见 A.6。

## A.1 四种机制总览

FedOrbit 由四个可分离的机制组成，分别对应第 IV 节的四项设计要求：

| 机制 | 位置 | 对应要求 | 解决什么 |
|---|---|---|---|
| **A. 连续轨道级训练 + ISL 中继** | 星上调度 | R1 | 不可见轨道**几乎不参与**的问题 |
| **B. 类别感知分层聚合** | 轨道内 + GS | R2 | 分类器行被**无关轨道稀释**的问题 |
| **C. 质量加权 + 回归率阻尼** | GS | R1 补充 | **训练预算不均**与**回归轨道冲击**的问题 |
| **D. 自适应特征分解** | 个性化推理 | R3 | **个性化强度该多大**的问题 |

## A.2 机制 A：连续训练与 ISL 中继

**问题**：在 $T = 400$ 轮中，**约 47% 的轮次没有任何轨道可见**，最少可见轨道只参与 15.2%→5.2% 的份额（最少的仅 5.2%）。标准 FL 下，这个轨道**在 95% 的时间里什么都不做**，它对最终全局模型的贡献可以忽略。

**做法**：把"是否参与训练"与"是否可见"**解耦**。

| 状态 | 训练方式 | epoch 数 | 是否用 GS 上行 |
|---|---|---|---|
| **可见** $l \in \mathcal{V}(t)$ | 正常路径 | $E_l(t) = \min(E_{\max}, E_{\text{base}} + s_e \tau_l(t))$，即 **5 → 最多 10** | 是 |
| **不可见** $l \notin \mathcal{V}(t)$ | 在上一轮轨道模型上继续本地 SGD，仅用**轨道内 ISL** 聚合 | $E_{\text{dark}} = 2$（刻意设小） | **否** |

**两个细节值得注意**：

1. **$E_l(t)$ 随缺席时长增长** —— 这是在"追赶"：缺席越久，回归时补的课越多（$s_e = 0.7$ 缩放，$E_{\max} = 10$ 封顶）。
2. **$E_{\text{dark}} = 2$ 刻意设得很小** —— 论文原话："so non-visible refinements **do not overwrite the heavier visible updates**"。这是一个**隐蔽但重要的设计**：不可见期间的训练是在**没有新鲜数据**的情况下做的（卫星还在飞，但看不到 GS；实际上数据仍在采集），如果让它训练太多，可能会把轨道模型**推离**全局一致的方向。设小是为了"保温"而非"改造"。

**代价**：星上额外计算 + 轨道内 ISL 流量。**但 GS 上行流量不变**（满足 R4）。

## A.3 机制 B：类别感知分层聚合

**问题**：普通平均（式 3、4）**按样本量加权**。假设类别 $c$ 只在轨道 5 上有，那么：
- 分类器 $c$ 行的更新，来自轨道 5 的是**有效信号**；
- 来自轨道 1–4 的是什么？它们**没有类别 $c$ 的样本**，所以它们对 $w^c[c,:]$ 的"更新"要么是零、要么是噪声。

按样本量加权时，如果轨道 1–4 样本量大，它们会把轨道 5 的有效更新**稀释掉**。

**做法**：把模型**拆成两部分**，分别用**不同的加权方式**：

$$w = [\underbrace{w_f}_{\text{特征提取器}},\ \underbrace{w_c}_{\text{分类器}}]$$

**关键区分（这是理解本文的一个要点）**：

| 组件 | 性质 | 该用什么加权 |
|---|---|---|
| **特征提取器** $w_f$ | 编码**跨星座共享**的内容（边缘、纹理、光谱模式） | 按**更新幅度**加权（式 10）—— 更新越大的卫星说明它学到了越多 |
| **分类器** $w_c$ | **每一行绑定到特定类别** | 按**该类别的样本归属**加权（式 9、11） |

所以：

- **式 (9)**（轨道内，分类器）：类别 $c$ 的行只按**持有类别 $c$ 的样本数** $s_{k,c}$ 加权。**没这个类别的卫星，权重自动为 0。**
- **式 (10)**（轨道内，特征提取器）：按**更新的逐元素绝对值**加权。直觉是"谁的更新幅度大，说明谁从这个数据里学到的东西多"。
- **式 (11)**（轨道间，分类器）：按**类别亲和度 $a_{l,c}$**（该轨道持有的类别 $c$ 样本占**全星座**的比例）加权，**再乘一个陈旧度因子 $\rho^{\sigma_l(t)}$**。

**关于 $\rho^{\sigma_l(t)}$ 的一个有趣之处**：论文说"**$\sigma_l(t)$ 很小，因为每个轨道在每一轮都训练**"。这意味着——**如果机制 A（连续训练）生效，机制 B 里的陈旧度因子几乎不起作用**。二者是**互补的**：A 保证了新鲜度，B 才敢按归属加权而不必担心用到过时数据。

**$s_{k,c}$ 与 $a_{l,c}$ 只算一次并固定**（式 7、8）。这是合理的，因为类别归属在训练过程中不会改变，把它作为常量可以省去每轮统计开销。

## A.4 机制 C：质量加权与回归率阻尼

**问题**：机制 A 造成了一个**新的不平衡**——不同轨道的**训练预算差异巨大**：

| 轨道状态 | 训练量 | 权重 $q_l$（式 12b） |
|---|---|---|
| 可见（刚从长缺席回归） | $E_{\max} \cdot R_{\text{intra}} = 10 \times 2 = 20$ | **20** |
| 可见（常规） | $E_{\text{base}} \cdot R_{\text{intra}} = 5 \times 2 = 10$ | **10** |
| 不可见 | $E_{\text{dark}} = 2$ | **2** |

**如果不处理，会出现两个问题**：

**(1) 加权问题**：一个只训练 2 个 epoch 的轨道，和一个训练 20 个 epoch 的轨道，凭什么同等发言权？式 (12) 直接用 $q_l$ 加权，让**训练更多的轨道贡献更大**。

**(2) 更隐蔽的"回归冲击"（return-rate shock）问题** —— 这是本节最精妙的地方。

考虑一个**刚从 191 轮缺席中回归**的轨道。它会以**全局学习率 $\eta_t$** 训练 **最多 $E_{\max}=10$ 个 epoch**（在每个轨道内轮次中）。这会带来一个**很大的特征漂移**（feature shift）。而与此同时：

> **其他轨道的分类器行，是基于"旧的"特征分布训练出来的，还没有适应这个新的特征。**

于是轨道间就会出现**特征—分类器不匹配**。这类似于一个突然变了输入分布的分类器，表现会崩。

**阻尼方案（式 13）**：

$$\eta_l(t) = \frac{\eta_t}{1 + \kappa\, \tau_l(t)/\tau_{\max}}, \qquad \kappa = 0.5,\ \tau_{\max} = 191$$

| 情形 | $\tau_l(t)$ | $\eta_l(t)$ | 含义 |
|---|---|---|---|
| 上一轮就可见 | 0 | $\eta_t$ | **不阻尼**（正常情况） |
| 缺席 191 轮后回归 | 191 | $\eta_t / 1.5 \approx 0.667\,\eta_t$ | 只用**三分之二**学习率 |
| 缺席一半后回归 | ~96 | $\eta_t / 1.25 = 0.8\,\eta_t$ | 用 80% 学习率 |

**关键限定**：**"只在重新进入后的第一轮应用"**（The rule is applied only on the first round after re-entry）。这是很克制的做法——只削掉最尖的那个冲击，之后恢复正常。

## A.5 机制 D 前半：为什么"单一全局模型"不够

机制 A–C 做完后，GS 得到一个**单一的全局模型** $[\bar{w}^f(t), w^c(t)]$。论文指出：

> "A single global model is appropriate when class distributions **overlap** and **suboptimal when they are disjoint**."

配合 A.0 的洞察，这就构成了**一对矛盾的需求**：

- **分布重叠时**（Dirichlet）→ **应该多共享**（单一全局模型好）
- **分布不相交时**（pathological）→ **应该少共享**（要保留轨道特异性）

**问题**：怎么**自动**知道当前是哪种情况？**不能人工调参**（R3 明确要求"without manual retuning"）。

## A.6 机制 D 后半：$\beta$ —— 全文最关键的参数（重点）

这是我认为整篇论文**最值得单独讲清楚**的部分，因为：

1. 它**只用一个标量**就统一了两种相反的划分场景；
2. 它**完全由数据算出**，不需要任何调参；
3. 它的取值变化（0.44 vs 0.11）为表格里的结果提供了**唯一的一致性解释**。

### 第一步：度量两个轨道"有多像"

用 **Jensen-Shannon 距离**（式 14）：

$$\text{JS}(p_i, p_j) = \sqrt{ \tfrac{1}{2} D_{\text{KL}}(p_i \| m) + \tfrac{1}{2} D_{\text{KL}}(p_j \| m) }, \quad m = \tfrac{1}{2}(p_i + p_j)$$

其中 $p_l$ 是**轨道 $l$ 的归一化类别直方图**（把该轨道所有卫星的逐类样本数汇总后归一化）。

**为什么用 JS 而不是 KL 散度？** 这是个关键的技术选择：

| 性质 | KL 散度 | JS 距离 |
|---|---|---|
| 对称性 | ✗ $D_{\text{KL}}(p\|q) \ne D_{\text{KL}}(q\|p)$ | ✓ 对称 |
| 有界性 | ✗ 可无限大（当 $q$ 中某类为 0 而 $p$ 不为 0） | ✓ 用 $\log_2$ 时 $\in [0,1]$ |
| 三点不等式 | ✗ 不满足 | ✓ 是真正的度量（metric） |

**在 pathological 划分下，这个选择至关重要**：轨道 $i$ 只持有类别 {0,1}，轨道 $j$ 只持有类别 {2,3}。那么 $p_i$ 在类别 2 上为 0，而 $p_j$ 在类别 2 上不为 0 → **KL 散度会趋向无穷**（出现 $\log 0$ 或除零）。而 **JS 距离天然有界，最多为 1**（完全不重叠）。

**这就是为什么式 (15) 能给出一个有意义的、可比较的相似度数值。**

### 第二步：把"所有轨道对"聚合成一个数字

$$\beta = \frac{2}{L(L-1)} \sum_{i<j} \left( 1 - \text{JS}(p_i, p_j) \right)$$

- $L(L-1)/2$ 是轨道对总数，因此 $\frac{2}{L(L-1)}$ 是**平均因子**；
- $1 - \text{JS}$ 把"**距离**"转成"**相似度**"（JS=0 完全相同时相似度为 1）；
- 所以 **$\beta$ = 所有轨道对的平均相似度**，$\beta \in [0,1]$。

**计算一次并固定**（原文："The coefficient is computed once and fixed"）。

### 第三步：用 $\beta$ 控制个性化强度

每个轨道维护自己的**个性化特征提取器** $w_l^{f,p}$，向全局平均做**指数混合**（式 16）：

$$w_l^{f,p}(t+1) = (1-\beta)\, w_l^{f,p}(t) + \beta\, \bar{w}^f(t)$$

**注意这里的设计巧思**：

| $\beta$ | 行为 | 适用场景 |
|---|---|---|
| **大**（→1） | 个性化提取器**快速**向全局靠拢 → 更"共享" | **分布重叠**（Dirichlet） |
| **小**（→0） | 个性化提取器**长时间保留**轨道特定信息 → 更"个性化" | **分布不相交**（pathological） |

**而 $\beta$ 恰好由数据自动给出**：

| 划分 | 实测 $\beta$ | 自动选的策略 | 与最优策略是否一致 |
|---|---|---|---|
| Dirichlet（$\alpha=0.5$） | **≈ 0.44**（中等） | 较强共享 | ✓ 一致（此时共享有益） |
| Pathological | **≈ 0.11**（很小） | 强个性化 | ✓ 一致（此时共享有害） |

**这就是全文最漂亮的一点**：不需要指定"现在是什么划分"，$\beta$ **自己会算出答案**。论文原文：

> "Both partitions are handled by the **same procedure with the same hyperparameters**."

**并且注意式 (17) 的非对称设计**：

$$\tilde{w}_l(t) = \left[ w_l^{f,p}(t),\ w^c(t) \right]$$

**只有特征提取器是个性化的，分类器始终用全局的。** 为什么？因为：

- **特征提取器**承载"内容"层面的差异（不同地形、不同光谱特征）→ 应该个性化；
- **分类器**是**类别决策边界**，而类别空间是全体星共享的语义 → 用全局的可以**集中所有轨道对每个类别的判断**。

这与机制 B 的"分类器按类别归属聚合"是**同一个设计哲学的两面**：分类器是全局共享的语义层，特征提取器是允许个性化的内容层。

## A.7 需要留意的"困难参数"与论文的自我限定

论文有几处**明确的自我限定**，读的时候要留意——这些是作者自己承认的缺口，而不是读者的理解偏差：

| 项目 | 论文原话/事实 | 含义 |
|---|---|---|
| **$\delta$（式 5 的公平性约束）** | "the fairness term is **not explicitly enforced** during optimisation" | 公平性只是**设计目标**，**优化时没有真正施加**；实际用离散度指标事后观察 |
| **组件隔离** | "Table II does **not isolate their individual effects**" | 表 II 是四个机制的**联合结果**，**无法归因到单个机制**；消融"left for future work" |
| **$T = 400$ 但只评最后 10 轮** | "averaged over the **last ten** evaluated rounds" | 主指标是**末期稳定值**，不反映收敛速度 |
| **单次仿真** | 论文未提及多种子/多次重复的方差 | 表格中只有**点估计**，**没有置信区间或标准差**（这一点与 [24] FN-HFL 形成对比） |
| **星座规模** | "larger and **dynamically reconfigured** constellations have **not been evaluated**" | 20 颗卫星（5 轨道 × 4）是**很小的规模**，可扩展性未验证 |
| **硬件验证** | "hardware-based validation ... left for future work" | 星上处理器的实际可行性**未做验证** |
| **$\kappa,\rho,s_e$ 的敏感性** | "A broader sensitivity analysis of $\kappa$, $\rho$, and $s_e$ is **left for future work**" | 三个关键超参数**未做敏感性分析** |
| **LeNet 规模** | "sub-million-parameter LeNet models" | 实验用的是**亚百万参数的 LeNet**，而非现代骨干网；"需数十兆字节工作内存" |

**另外一处值得注意的"参数不对称"**：$\kappa = 0.5$ 使最长缺席的轨道学习率降到 $2/3$。这个幅度**相当温和**——如果回归冲击是主要问题，可能需要更强的阻尼。论文自己没有做这个 ablation，所以**无法判断 $\kappa = 0.5$ 是否接近最优**。

## A.8 结果解读：一个容易被误读的对比

表 II 里最容易误读的是这一组：

| 数据集 | 划分 | Ditto | FedOrbit | 差值 |
|---|---|---|---|---|
| EuroSAT | **Path.** | **95.8** | 94.9 | **−0.9**（FedOrbit 输） |
| EuroSAT | Dir. | 58.5 | **76.4** | **+17.9** |
| So2Sat | Path. | 84.0 | **86.9** | +2.9 |
| RESISC45 | Path. | 77.7 | **86.3** | +8.6 |
| RESISC45 | Dir. | 36.3 | **54.2** | +17.9 |

**解读时要注意三点**：

1. **Ditto 是一个"偏科"方法**。它在 pathological 下极强（95.8），因为此时"完全不共享"恰好是对的。但它在 Dirichlet 下**崩到 58.5**——比 FedAvg 的 68.7 **还低 10 个百分点**。原因是：**它把向全局模型的正则化施加在一个有偏的锚点上，同时过度个性化**。
2. **FedOrbit 的唯一失利（EuroSAT pathological，−0.9）在统计意义上是否显著，论文没有说明**（无标准差）。在只有单次仿真、无置信区间的情况下，0.9 个百分点**可能只是噪声**。作者也只说"within 0.9 percentage points"，措辞很谨慎。
3. **真正有说服力的证据是"跨场景一致性"**：Ditto 在两种划分间的**波动达 37–48 个百分点**，而 FedOrbit 在两种划分间都很稳（EuroSAT: 94.9 vs 76.4；RESISC45: 86.3 vs 54.2）。**一个不需要事先知道是哪种划分就能工作的方法，比一个在特定划分下极强的方法更有工程价值。**

**这一点与 [24] FN-HFL 的"深度自适应"在哲学上是一致的**：都是**让系统根据环境自动调整共享强度**，而不是要求运维者事先选对配置。

---

## 附录 B：与 SpaceFL / fl_space 项目的对应关系

| 论文概念 | 可映射到 fl_space 的实现 |
|---|---|
| 可见性矩阵 $V \in \{0,1\}^{T \times L}$ | `fl_space/fl/contact_matrix.py`（`ContactMatrix` 的布尔矩阵形式） |
| 式 (1) 仰角判据 $\varepsilon_k(t) \ge \alpha_e$ | `orbit_simulator` 的地面站可见性计算（需确认是否已实现仰角阈值） |
| 轨道级聚合（式 3、4） | 现成的层次化聚合路径 |
| **ISL 中继训练不可见轨道**（机制 A） | **需新增**：`ContactMatrix` 已有拓扑信息，可扩展为"轨道内 ISL 可用即训练" |
| **类别感知加权** $s_{k,c}$, $a_{l,c}$（式 7、8） | **需新增**：需要 client 端的逐类样本统计接口（`FedBuff`/`FedProx` 的 dataset 统计可复用） |
| **质量加权** $q_l$（式 12） | **需新增**：按实际训练量加权，是对现有 sample-count 加权的推广 |
| **回归率阻尼** $\eta_l(t)$（式 13） | **需新增**：per-client 学习率调节，可挂在 `trainer` 的学习率调度钩子上 |
| **$\beta$ 自适应个性化**（式 14–16） | **需新增**：这是最值得借鉴的部分 —— 用 `scipy.spatial.distance.jensenshannon` 一行算出 |
| $T$、$\tau_{\max}$ 等时间参数 | `fl_space/fl/time_model.py` 的 `SlotTimeModel` |

**最值得借鉴的三点**：

1. **$\beta$ 的"用数据自动定个性化强度"思路** —— 这是**唯一一个不需要调参就能在两种极端划分下都工作**的机制。`fl_space` 目前若要支持个性化，可以直接复用这个"JS 距离 → 单一系数 → 指数混合"的模式，代码量极小（约 20 行）而收益明确。**这是我建议优先实现的。**

2. **"训练量与可见性解耦"（机制 A）** —— 用 ISL 让不可见轨道继续训练，把"47% 轮次无可见轨道"这个硬约束变成软约束。这一思路与 `fl_space` 的 `ContactMatrix` 天然契合，且与 FedBuff 的异步思想互补。

3. **"分类器全局共享 + 特征提取器个性化"的非对称设计（式 17）** —— 比"整个模型都个性化"或"整个模型都共享"都更合理，且有机制 B 的类别归属加权作为理论支撑。

**一处需要警惕的对比**：本文与 [24] FN-HFL 都做"自适应"，但**自适应对象不同**：
- **FN-HFL** 自适应的是**模型深度**（按算力/SWaP-C）；
- **FedOrbit** 自适应的是**共享强度**（按数据相似度）。

两者**并不冲突，可以叠加**：一个系统可以同时按算力分深度、按数据相似度定共享强度。如果要在一个框架里同时支持，建议把二者都实现为**由数据/状态算出的标量**（深度由一个调度器定，$\beta$ 由 JS 距离定），保持"不靠人工调参"这一共同原则。
