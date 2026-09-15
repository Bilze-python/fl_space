# [20] 地面辅助的LEO卫星星座联邦学习

> **原文标题**: Ground-Assisted Federated Learning in LEO Satellite Constellations
> **作者**: Nasrin Razmi, Bho Matthiesen, Armin Dekorsy, Petar Popovski
> **发表**: IEEE Wireless Communications Letters, Vol. 11, No. 4, April 2022
> **DOI**: 10.1109/LWC.2022.3141120

---

## 摘要

在低地球轨道（LEO）巨型星座中，存在着诸如基于卫星成像的推理等相关的应用场景，其中大量卫星在不共享本地数据集的前提下协作训练一个机器学习模型。为了解决这一问题，我们提出了一组基于联邦学习（FL）的新算法，其中包括一种基于FedAvg的新型异步FL过程，该过程在异构场景下比现有最优方法表现出更好的鲁棒性。基于MNIST和CIFAR-10数据集的广泛数值评估凸显了所提方法的快速收敛速度和优异的渐近测试准确率。

**关键词**: 卫星通信，低地球轨道（LEO），联邦优化。

---

## I. 引言

由小型卫星组成的低地球轨道（LEO）星座是中高轨道和地球静止轨道传统大型卫星的一种成本高效且灵活的替代方案。目前已有多个此类星座部署实施，旨在提供无处不在的连接和低延迟互联网服务[1]。它们与地面移动网络的融合是一个活跃的研究领域，涵盖了地球观测任务等多种应用场景[2]-[6]。可以预见，机器学习（ML）将成为管理这些星座并利用其传感器测量数据的关键工具[7]-[9]。

传统的ML方法是将所有数据集中到某个中心位置然后求解学习问题。考虑到训练深度神经网络所需的海量数据[10]，这涉及高昂的传输成本和时延。而且，随着小型卫星私营所有者的涌现，出于隐私或数据所有权方面的考虑，共享数据可能被禁止。解决这一困境的显而易见的方案是本地训练并仅聚合所得到的模型参数。这通过协作求解ML问题、仅共享更新后的模型参数来实现。将数据异构性和有限连接性纳入考量的分布式ML范式被称为联邦学习（FL）[11],[12]。将分布式ML应用于卫星星座是自然而然的选择。

通用FL环境的一个核心假设是客户端（即本例中的卫星）的参与是间歇且不可预测的。为了应对这一点，近期已有异步算法被提出[14]。然而，LEO学习场景的独特特征在于客户端的可预测可用性，结合同一地面站（GS）两次访问之间很长的间隔时间。

在本文中，我们研究了当训练过程由一个GS编排时，这种可预测可用性如何影响FL场景，并提出了一种新型异步算法。我们确凿地证明，与当前最优的FL算法相比，我们的方法能够带来更优的训练性能。具体而言，我们的核心贡献为：

1. 定义了LEO FL场景，并识别了与传统FL相比的核心挑战；
2. 为卫星FL提出了一个算法框架和通信协议；
3. 将FedAvg[12]和FedAsync[14]适配到该场景，并提出一种新型异步FedAvg变体，特别适用于卫星星座中的地面辅助FL；
4. 对所讨论的算法进行数值评估以验证我们的理论考量。

实验结果表明，所提出的异步FL算法在异构场景中比FedAsync具有更高的鲁棒性。

---

## II. 系统模型与FL背景

考虑一个由L个轨道面上的K颗卫星组成的LEO星座。在地心惯性坐标系中，卫星k（k ∈ K = {1,...,K}）具有轨迹 **r_k**(t)，而GS虽固定于地球表面某恒定位置，其轨迹为 **r_g**(t)。当卫星k从GS处可见时，即π/2 − ∠(**r_g**(t), **r_k**(t) − **r_g**(t)) ≥ α_e 时，星地链路才可行，其中α_e为最小仰角。一般而言，一次仅有一小部分卫星与GS保持连接，且两次连接之间的时间远长于实际在线时间。

卫星k从其星载仪器收集数据并将其存储在数据集 **D_k** 中。由于轨道和轨道位置的差异，两颗不同卫星的数据集是互不重叠且可能非独立同分布（non-IID）的。数据采集阶段结束后，卫星协作求解如下形式的优化问题：

$$\min_{\theta \in \mathbb{R}^d} \frac{1}{n} \sum_{k \in \mathcal{K}} \frac{n_k}{n} \sum_{x \in \mathcal{D}_k} \frac{1}{n_k} f(x; \theta)$$

目标是训练一个ML模型，其中f(x;θ)为训练损失函数，由训练任务定义。该过程由GS编排并以迭代方式进行，无需在卫星之间共享数据集。

我们假设卫星上可用于求解(1)的计算资源非常有限。因此，我们考虑卫星在两次GS访问之间对(1)进行计算，并利用接触时间进行模型参数θ的交换。

### A. 联邦学习背景

在间歇连接、异构数据集且不共享本地原始数据的假设下分布式求解ML训练问题(1)称为FL。最广泛使用的方法是FedAvg算法[12]。参数服务器（PS）管理学习过程并维护当前全局模型参数θ^i的版本。在每个全局迭代i（即"epoch"）中，PS选择一个可用工作节点子集S_i参与下一轮。它将当前全局模型版本发送给选中的工作节点，然后等待所有工作节点返回结果。

工作节点在其本地数据集上执行一轮或多轮小批量随机梯度下降（SGD）。具体而言，在每个本地epoch中，本地数据集被划分为⌈n_k/B⌉个大小为B的随机批次B，并对每个小批量以学习率η执行一步SGD。损失函数基于(1)定义为：

$$g_\theta(B; \theta) = |B|^{-1} \sum_{x \in B} f(x; \theta) + \tilde{g}_\theta(B; \theta)$$

其中g̃为可选的正则化项[15]。完成后，更新后的本地模型参数被发送至PS。整个流程见**算法1**。

在收到所有已调度工作节点的结果后，PS将结果聚合为一个新版本的全局模型参数：

$$\theta^{i+1} = \sum_{k \in S_i} \frac{n_k}{\sum_{k \in S_i} n_k} \theta_k^i$$

这是**同步FL**，如果PS不得不等待落后者，收敛速度可能变慢。解决这一问题的一种方式是当客户端更新到达时即予以纳入，这被称为**异步FL**。FedAsync[14]就是这样一个算法，在某些情况下被证明优于FedAvg。在FedAsync中，客户端操作与算法1相同，但PS操作方式不同，通过向部分工作节点发送当前全局模型参数随同epoch周期性地分配计算任务。客户端更新以异步方式在到达时即被纳入。具体地，来自客户端k在epoch i的更新按如下方式纳入：

$$\theta^{i+1} = (1-\alpha)\theta^i + \alpha\theta_k^i$$

其中混合因子α ∈ (0,1)决定了对传入客户端更新的权重分配。该因子确定为α = α' · s(i − τ_k)，其中α'为固定基础权重，i为当前epoch，τ_k为工作节点收到全局模型时的epoch，s(i) ∈ (0,1]是一个问题特定的陈旧度函数，可用于降低对基于旧版全局模型的更新的权重。

---

## III. 卫星上的联邦学习

第II-A节讨论的FL算法是在设备可用性由随机过程驱动且并行通信无显著时延的前提下设计的。然而，卫星场景在几个方面有根本性不同：工作节点的数量比地面应用少几个量级；设备随时可用于计算任务，但通信仅在短暂且高度可预测的时间窗口内可行。此外，任何时刻只有极小部分工作节点在通信范围内。

虽然该场景最适合采用异步FL算法，但我们仍将同步FedAvg算法作为基线。我们首先概述通信协议，定义卫星操作，并讨论FedAvg和FedAsync在卫星场景中的应用。然后，我们设计一种新颖的异步算法，利用卫星通信的可预测连接性来实现不带不必要延迟的FedAvg。

### A. 通信协议与卫星操作

通信以客户端-服务器协议实现，所有连接均由卫星发起。当卫星不在执行通信任务时，它尝试联系GS。因此，通信要么在GS进入通信范围时启动，要么在完成通信任务后直接启动。连接建立后，卫星k发送一个本地模型参数更新θ_k^i（如果有且之前未发送），其中i表示当前全局epoch。然后，GS更新全局模型参数(θ^i, θ_k^i) ↦ θ^{i+1}，并决定卫星k是否应继续计算。如果继续，GS将更新后的全局模型参数向量发送给卫星k并终止连接。否则，连接终止且卫星在此次飞越期间不再重新建立连接。

卫星上的计算任务如**算法1**所述。为避免由于异步操作及GS访问之间的长时间延迟导致与全局模型偏差过大，采用对模型参数的L2正则化[15]，即(2)中的正则化项选为：

$$\tilde{g}_\theta(B; \theta) = \frac{\lambda}{2} \|\theta - \tilde{\theta}\|_2^2$$

其中λ为参数。第3行的停止准则为固定的迭代次数，应选择使计算在下一次GS访问之前完成。

### B. 同步地面站操作

我们通过将FedAvg适配到卫星场景来开始GS操作的讨论。回忆FedAvg服务器在epoch i中选择一个工作节点子集S_i对当前模型θ^i执行更新，然后等待所有已调度结果到达后再依据(3)更新模型。将此算法朴素的适配到卫星场景的结果见**算法2**。

主循环持续运行直到收敛（由任意常用标准判定，如epoch数、已流逝时间或早停法）。新epoch的工作节点由SCHEDULE函数选择并存储于S_i和R_i中。S_i包含应接收当前全局模型参数的已调度工作节点，而R_i保存尚未返回其模型更新的工作节点。内层循环（第6-19行）运行直到两个集合均为空。

与标准FedAvg的关键区别在于通信是异步的，以允许调度并非同时对GS可见的卫星，而更新仍然是同步计算的。一个重要观察是，第6-19行的工作循环是阻塞的，即它在启动新epoch前等待所有已调度卫星与GS两次连接。假设卫星在单次飞越中无法完成计算，这意味着一次epoch平均至少需要一个轨道周期。而且，当调度多颗卫星时，这一时间还会增加。

从优化理论角度看，算法2等价于FedProx[15]（一种使用了(5)中正则化的FedAvg变体）。其收敛性由[15, Th. 4]保证。

### C. 异步地面站操作

与算法2相比，异步FL操作允许卫星在不同版本的全局模型上工作，从而减少时延。**算法3**概述了GS的操作。在第3行中，它等待任意卫星连接。通信假定为非阻塞的，即通信时延对算法3透明。如果收到模型更新，epoch在第5行递增，全局模型基于其当前版本和接收到的更新进行更新。如果该卫星被调度继续进行进一步计算，则新的全局模型在第12行发送。连接在第14行终止。

#### 1) FedAsync

算法3中SERVERUPDATE和SCHEDULE过程的实现取决于FL方案和通信场景。对于FedAsync，SERVERUPDATE首先基于陈旧度函数s(i − τ_k)计算混合因子α，然后返回(4)。由于没有任何客户端更新可以比一个轨道周期更"新鲜"，我们提出使用根据[14, Sec. 5.2]中定义的铰链型陈旧度函数。设s(i − τ_k) = s̃(t_i − t_{τ_k})，其中t_j是epoch j在GS处被处理的时间，并且：

$$\tilde{s}(t) =
\begin{cases}
1 & \text{if } t \leq (1+\varepsilon)T_{o,\max} \\
(1 + a(t - (1+\varepsilon)T_{o,\max}))^{-1} & \text{otherwise}
\end{cases}$$

其中ε ≥ 0为一个小值，a为一个正常数，T_{o,max}为星座内最大轨道周期。调度器可以轻松计算给定卫星下一次飞越时的s(·)值。因此，如果权重α将低于某个阈值，可将SCHEDULE(k, t)设为false以节省能源和计算资源。

---

## IV. 展开式联邦平均算法

应用于地面辅助卫星在轨学习的同步FL过程面临高延迟问题。这可以通过像FedAsync这样的异步FL过程来缓解。然而，在全客户端参与和参数选择恰当的情况下，FedAvg比FedAsync具有更强的收敛性质。因此，以异步方式实现FedAvg是可取的。利用卫星的可预测连接性，这确实可行。

首先，考虑一个近极地Walker Delta星座[16]，具有单一轨道壳层且GS位于北极。这是一个对称场景，每颗卫星每个轨道周期恰好访问GS一次。此外，卫星与GS的连接顺序是恒定的，即如果卫星按某区间[t, t+T_o]内的顺序排列（T_o为轨道周期），且卫星连接顺序为1→2→3→···→K，则该顺序在后续每个轨道周期中重复。

在此情况下，全客户端参与的FedAvg更新规则(3)可以增量式实现而无需更新阶段的同步性。具体而言，假设卫星k在时刻t_{i1}访问GS（对应epoch i1），并在t_{i2} = t_{i1}+T_o时刻再次访问。那么客户端更新θ^{i2}_k基于θ^{i1+1}，且可以按如下方式纳入全局模型：

$$\theta^{i2+1} = \theta^{i2} - \alpha_k (\theta^{i1}_k - \theta^{i2}_k), \qquad \alpha_k = \frac{n_k}{n}$$

每个轨道周期恰好有K次更新，而且由于卫星-GS联系顺序的周期性，经过K的整数倍个epoch后的模型应接近算法2的结果。

更一般地，考虑一个星座，其中每颗卫星相对于任意固定位置的GS的重访周期趋近于相同值。那么，算法3中采用如(7)的SERVERUPDATE函数的流程按[17, Sec. 2]所证明的那样收敛。

最后，考虑具有多个轨道壳层的星座。在这种情况下，等重访率的假设通常不成立。按[17]中的讨论，这不会阻止收敛但可能导致有偏解。然而，这对于许多其他当前最优FL算法（包括本文中介绍的方法）也是同样的情况。事实上，下一节的数值结果将表明，在该算法中这一效应远不如在异步基线FedAsync中那么显著。

---

## V. 实证结果

我们以MNIST[18]和CIFAR-10[19]数据集上的测试准确率来数值评估所提出算法的性能。对于MNIST，我们训练一个具有7850个可训练参数的逻辑回归模型[15]。集中式训练的期望准确率约为89%。对于CIFAR，我们训练一个ResNet-18，集中训练时可达到略高于90%的准确率[20]。训练数据集在所有工作节点间随机均匀分配。

每颗卫星按算法1操作，η=0.1，λ=0，在GS访问之间以10为批次大小对本地数据集进行一次完整遍历。我们基于FedML框架[21]进行FL实现。在结果中，我们称算法2为"FedAvg"，称第III-C1节中的异步基线为"FedAsync"，称第IV节中的算法为"FedSat"。

FedAsync的陈旧度函数（如使用）参数为ε=0.01和a=5(1+ε)T_{o,max}，根据开普勒第三定律T_{o,max}≈127分钟。FedAsync的混合参数α针对每个实验单独微调。

考虑一个具有两个轨道壳层（分别在500km和2000km高度）的卫星星座，每个壳层包含5颗卫星。两者均为Walker Delta星座[16]，倾角80°，5个轨道面。它们经过偏移使得壳层间升交点赤经（RAAN）最小差为36°。最小仰角α_e为10°。对于非IID情况，一半可用类别分配给500km轨道壳层，另一半分配给2000km壳层。

首先，考虑GS位于德国不来梅且数据呈非IID分布的情况。由于不均匀的设备参与和异构数据集，该场景对算法提出了相当大的挑战。图1和图2分别展示了MNIST和CIFAR的测试准确率。FedAvg在2T_{o,max}的延迟后对MNIST几乎瞬时收敛（缘于简单模型）。而在CIFAR实验中，准确率呈现阶跃函数形态，收敛非常缓慢。这清楚地展示了FedAvg及同步算法在地面辅助卫星学习中的不足之处。在两个实验中，FedAsync均比FedAvg收敛更快，但训练性能较差。有趣的是，(6)中提出的陈旧度函数对MNIST的稳定收敛是必要的，但对CIFAR训练有负面影响。混合因子α'在MNIST和CIFAR中分别设为0.5和0.1，FedAsync的学习率η为0.01。所提出的FedSat算法在收敛速度和最终测试准确率两方面均展现出优越的训练性能。

表/图关键结果：
- **图1**: GS在不来梅，非IID MNIST数据下的Top-1准确率
- **图2**: GS在不来梅，非IID CIFAR数据下的Top-1准确率
- **图3**: GS在北极，IID CIFAR数据下的Top-1准确率
- **图4**: GS在不来梅，IID CIFAR数据下的Top-1准确率
- **图5**: 均匀客户端采样下的Top-1准确率对比

其次，我们考虑均匀场景（IID数据分布，GS在北极）。准确率结果如图3所示。FedAvg行为同前，但更新之间的时间周期较短，这是因为星座围绕GS旋转。在异步算法中，FedAsync比FedSat略有优势。然而，这需要微调额外的超参数α，本例中其最优值为0.3。最后，图4展示了相同数据分布下但GS在不来梅的结果，这引入了不均匀的设备参与。核心观察是FedAsync现在的表现严格差于FedSat。由此我们得出结论：所提出的方法对异构性表现出显著更高的鲁棒性，这对当前场景是一个重要性质。

总之，这些实验验证了我们的理论考量。我们观察到FedAvg[12]的朴素实现导致巨大的时延，且异步FL的当前最优算法（即FedAsync）难以应对卫星学习场景固有的异构性。我们推测这不仅适用于卫星星座，也适用于具有异构性的通用FL场景，这一结论得到图5中额外仿真的支持。相反，所提出的算法在所有实验中均展现出优异的性能。

---

## VI. 结论

我们考虑了LEO星座中的FL，其中卫星在不共享本地数据集的前提下协作训练ML模型。我们识别并应对了与地面网络相比的独特挑战，通过将FedAvg和FedAsync适配到该场景。我们演示了如何利用确定性的工作节点可用性来"展开"FedAvg，从而有效地将其从同步算法转换为异步学习算法，同时不牺牲训练性能。这将FedAvg的训练时间缩短了数小时，并得出一个在收敛时间和测试准确率两方面均优于FedAsync的算法。所提出的算法还具有比FedAsync更少的超参数需要调优。

这项初步工作留下了若干有待未来探索的课题，包括工作节点的合理调度、单次GS飞越期间的多次数据交换以及采用多个GS。这些方法可能带来相当可观的训练加速。

---

## 参考文献

[1] I. del Portillo, B. G. Cameron, and E. F. Crawley, "A technical comparison of three low earth orbit satellite constellation systems to provide global broadband," *Acta Astronaut.*, vol. 159, pp. 123–135, Jun. 2019.

[2] I. Leyva-Mayorga et al., "LEO small-satellite constellations for 5G and beyond-5G communications," *IEEE Access*, vol. 8, pp. 184955–184964, 2020.

[3] Y. Qian, "Integrated terrestrial-satellite communication networks and services," *IEEE Wireless Commun.*, vol. 27, no. 6, pp. 2–3, Dec. 2020.

[4] O. Kodheli et al., "Satellite communications in the new space era: A survey and future challenges," *IEEE Commun. Surveys Tuts.*, vol. 23, no. 1, pp. 70–109, 1st Quart., 2021.

[5] B. Di, L. Song, Y. Li, and H. V. Poor, "Ultra-dense LEO: Integration of satellite access networks into 5G and beyond," *IEEE Wireless Commun.*, vol. 26, no. 2, pp. 62–69, Apr. 2019.

[6] Z. Lin, M. Lin, T. de Cola, J.-B. Wang, W.-P. Zhu, and J. Cheng, "Supporting IoT with rate-splitting multiple access in satellite and aerial-integrated networks," *IEEE Internet Things J.*, vol. 8, no. 14, pp. 11123–11134, Jul. 2021.

[7] M. A. Vazquez et al., "Machine learning for satellite communications operations," *IEEE Commun. Mag.*, vol. 59, no. 2, pp. 22–27, Feb. 2021.

[8] G. Giuffrida et al., "CloudScout: A deep neural network for on-board cloud detection on hyperspectral images," *Remote Sens.*, vol. 12, no. 14, p. 2205, Jul. 2020.

[9] G. Mateo-Garcia et al., "Towards global flood mapping onboard low cost satellites with machine learning," *Sci. Rep.*, vol. 11, p. 7249, Mar. 2021.

[10] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA, USA: MIT Press, 2016.

[11] J. Konečný, H. B. McMahan, and D. Ramage, "Federated optimization: Distributed optimization beyond the datacenter," in *Proc. 8th NIPS Workshop Optim. Mach. Learn. (OPT)*, Dec. 2015.

[12] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. Y. Arcas, "Communication-efficient learning of deep networks from decentralized data," in *Proc. 20th Int. Conf. Artif. Intell. Statist. (AISTATS)*, vol. 54, Apr. 2017.

[13] X. Wang, Y. Han, V. C. M. Leung, D. Niyato, X. Yan, and X. Chen, "Convergence of edge computing and deep learning: A comprehensive survey," *IEEE Commun. Surveys Tuts.*, vol. 22, no. 2, pp. 869–904, 2nd Quart., 2020.

[14] C. Xie, O. Koyejo, and I. Gupta, "Asynchronous federated optimization," in *Proc. Annu. Workshop Optim. Mach. Learn. (OPT)*, Dec. 2020.

[15] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, "Federated optimization in heterogeneous networks," in *Proc. Mach. Learn. Syst. (MLSys)*, Austin, TX, USA, Mar. 2020, pp. 429–450.

[16] J. G. Walker, "Satellite constellations," *J. Brit. Interplanet. Soc.*, vol. 37, pp. 559–571, Dec. 1984.

[17] A. Nedić, D. Bertsekas, and V. Borkar, "Distributed asynchronous incremental subgradient methods," in *Studies in Computational Mathematics*, vol. 8. Amsterdam, The Netherlands: Elsevier, 2001, pp. 381–407.

[18] Y. LeCun, C. Cortes, and C. J. C. Burges, "The MNIST Database of Handwritten Digits." [Online]. Available: http://yann.lecun.com/exdb/mnist/

[19] A. Krizhevsky, "Learning multiple layers of features from tiny images," Dept. Comput. Sci., Univ. Toronto, Toronto, ON, USA, Rep. R-2009, 2009.

[20] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," Microsoft Res., Redmond, WA, USA, Rep., 2015.

[21] C. He et al., "FedML: A research library and benchmark for federated machine learning," 2020, arXiv:2007.13518.
