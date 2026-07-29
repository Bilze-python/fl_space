# [21] LEO卫星星座中地面辅助联邦学习的调度

> **原文标题**: Scheduling for Ground-Assisted Federated Learning in LEO Satellite Constellations
> **作者**: Nasrin Razmi, Bho Matthiesen, Armin Dekorsy, Petar Popovski
> **发表**: EUSIPCO 2022
> **ISBN**: 978-1-6654-6798-8

---

## 摘要

本文考虑了直接在低地球轨道（LEO）卫星上进行机器学习模型的分布式训练。基于专门针对卫星场景独特挑战的联邦学习（FL）算法，我们设计了一个调度器，利用地面站（GS）与卫星之间访问时间的可预测性来减少模型陈旧度。数值实验表明，这可以将收敛速度提高三倍。

**关键词**: LEO星座，联邦学习，调度。

---

## I. 引言

成本低且易于部署的小型低地球轨道（LEO）卫星正在开启卫星通信及其与地面网络融合的新纪元。这些LEO卫星通常部署于大型星座中，从而形成一个可移动的基础设施，为通信服务和地球观测等多种应用提供无缝的全球覆盖[1]。这些应用中有许多是数据密集型的。例如，地球观测中成像设备的高空间、光谱和时间分辨率导致采集了大量数据[2]。这些数据被用于灾害预防、环境监测和城市规划等多种应用。由于射频频谱资源稀缺或严格的时延要求[3]，将如此大量的数据传输到地球可能并不实际。

为解决这些限制，一个合理的方案是直接在卫星上处理数据，仅将抽象化信息发送给地面站（GS）。在这方面，使用联邦学习（FL）[4]是可行的，它是一种协作机器学习（ML）方案，卫星只需向服务器传输模型参数而非原始数据。图1描绘了FL用于地球观测的场景。在原始FL设置中，用户在训练过程中的参与是间歇且随机化的，取决于用户活动和通信可用性。在卫星场景中，链路的（不）可用性是可预测的，且与卫星到GS位置的访问模式相关。

这种由GS编排的FL场景首次在[5]中被识别，其中提出了一种新的异步FL过程来应对卫星场景的独特挑战。在[6]中，研究表明星间链路（ISL）的存在显著提高了卫星上FL的收敛速度。在不具备此能力的情况下，对[5]中方法的若干扩展可能带来更快的收敛速度。事实上，[7]的作者考虑了一种启发式GS更新过程和梯度缓冲来加速收敛。

在本文中，我们采取了不同的方法，基于GS-卫星访问时长的可预测性设计一个调度器以减少模型陈旧度。这带来了比[5]中基线方法显著更快的收敛速度，并且可以与不同的聚合规则（如[7]中的）相结合。

---

## II. 系统模型

我们考虑一个具有P条圆形轨道的LEO卫星星座，其中第p条轨道包含K_p颗卫星。集合K={S_1, S_2, ..., S_K}表示所有K = ΣK_p颗卫星。轨道p的高度和倾角分别记为h_p和i_p。设T_p = 2π√((r_E+h_p)³/μ)和v_p = √(μ/(h_p+r_E))分别为轨道p中卫星的轨道周期和速度，其中r_E=6371km为地球半径，μ=3.98×10¹⁴ m³/s²为地心引力常数。

如果卫星与GS之间的视距链路未被地球遮挡，即处于访问状态，卫星可以与GS通信。否则处于非访问状态。卫星k与GS之间的视距链路在π/2 − ∠(**r**_GS, **r**_k − **r**_GS) ≥ α_e时可用，其中**r**_k和**r**_GS分别表示卫星k和GS的位置，α_e为最小仰角。

我们定义两个重要时刻：卫星k的**升起时间** t^n_{r,k} 是其进入第n次访问的时刻，而**落下时间** t^n_{s,k} 是其完成第n次访问的时刻。所有K颗卫星的升起时间序列τ_rise表示为：

$$\tau_{\text{rise}} = (\{t^n_{r,1}\}^{N_1}_{n=1}, \{t^n_{r,2}\}^{N_2}_{n=1}, ..., \{t^n_{r,K}\}^{N_K}_{n=1})$$

其中t_{r,k} = {t^n_{r,k}}^{N_k}_{n=1}是卫星k的升起时间序列，N_k是卫星k在考虑的时间区间[T_b, T_f]内的访问状态次数。类似地，所有K颗卫星的落下时间序列τ_set为：

$$\tau_{\text{set}} = (\{t^n_{s,1}\}^{N_1}_{n=1}, \{t^n_{s,2}\}^{N_2}_{n=1}, ..., \{t^n_{s,K}\}^{N_K}_{n=1})$$

接下来，我们定义与这些序列相关的两种时间间隔。**在线时间**是升起时间和落下时间之间的时间间隔，对应访问状态的持续时间。**离线时间**是两个访问状态之间的时间间隔。所有K颗卫星的在线时间序列τ_on定义为：

$$\tau_{\text{on}} = (\{[t^n_{r,1}, t^n_{s,1}]\}^{N_1}_{n=1}, ..., \{[t^n_{r,K}, t^n_{s,K}]\}^{N_K}_{n=1})$$

卫星的离线时间序列τ_off为：

$$\tau_{\text{off}} = (\{t^n_{\text{off},1}\}^{N_1+1}_{n=1}, ..., \{t^n_{\text{off},K}\}^{N_K+1}_{n=1})$$

其中t_off,k = {t^n_off,k}^{N_k+1}_{n=1} = {[T_b, t¹_{r,k}], [t¹_{s,k}, t²_{r,k}], ..., [t^{N_k}_{s,k}, T_f]}。

状态E(t,k)表示卫星k在时刻t的访问状态，定义为：

$$E(t,k) = \begin{cases} E_{\text{on}}, & t \in t_{\text{on},k} \\ E_{\text{off}}, & t \in t_{\text{off},k} \end{cases}$$

为简化起见，后续我们省略时间分量t，将卫星k的状态记为E(k)。

### A. 计算模型

每颗卫星k从地球采集一个本地数据集**D**_k = {**x**_1, ..., **x**_D_k}，其中**x**_i和D_k分别表示该卫星的第i个样本和样本数量。这些数据用于训练一个ML模型，每颗卫星k构建的损失函数F_k(**w**)为：

$$F_k(\mathbf{w}) = \frac{1}{D_k} \sum_{\mathbf{x} \in \mathcal{D}_k} f_k(\mathbf{x}, \mathbf{w})$$

其中f_k(**x**, **w**)是卫星k处的单样本损失函数，建立在学习目标之上（可以是任意凸或非凸函数）。向量**w**表示描述模型的参数。

每颗卫星的本地原始数据集保持私密，既不与其他卫星共享也不与GS共享。卫星旨在协作最小化全局损失函数：

$$F(\mathbf{w}) = \sum_{k \in \mathcal{K}} \frac{D_k}{D} F_k(\mathbf{w})$$

其中D = ΣD_k为总样本数。与著名的FedAvg算法[4]仅有全局epoch单个计数器不同，我们为GS和每颗卫星分别定义计数器。全局模型epoch记为n，由GS跟踪。此外，对任意卫星k，定义本地计数器n_k以跟踪卫星的参与情况。例如，在同步FL场景中，对所有k有n_k=n。

设**w**^n为epoch n时的全局模型参数。假设每颗卫星使用SGD训练模型I次迭代，卫星k在迭代i≥1的本地模型参数为：

$$\mathbf{w}^{n_k,i}_k = \mathbf{w}^{n_k,i-1}_k - \eta \nabla F_k(\mathbf{w}^{n_k,i-1}_k)$$

其中**w**^{n_k,0}_k是卫星k在其第n_k次更新中接收到的全局模型，η为学习率。遵循[8]中的线性计算时间模型，卫星k计算一次对全局模型更新所需的时间t_l(k)为：

$$t_l(k) = \frac{c_k I S(D_k)}{\nu_k}$$

其中c_k是处理单个数据比特所需的CPU周期数，S(D_k)是数据量（比特），ν_k是CPU频率。

### B. 通信模型

如果卫星与GS之间的视距链路不被地球遮挡，即卫星k处于在线时间段且E(k)=E_on，则通信可行。卫星k与GS之间的信噪比(SNR)为[9]：

$$\text{SNR}(k, GS) = \begin{cases} \frac{P_t G_k G_{GS}}{N_0 L(k, GS)}, & \text{if } E(k)=E_{\text{on}} \\ 0, & \text{if } E(k)=E_{\text{off}} \end{cases}$$

其中P_t为发射功率，G_k和G_GS分别是卫星k对GS及GS对卫星k的平均天线增益，N_0 = k_B T B为总噪声功率（k_B=1.380649×10⁻²³ J/K为玻尔兹曼常数，T为接收机噪声温度，B为信道带宽）。卫星k与GS之间的自由空间路径损耗L(k, GS)为：

$$L(k, GS) = \left(\frac{4\pi f_c d(k, GS)}{c}\right)^2$$

其中f_c为载波频率，c为光速，d(k, GS)为卫星k与GS之间的距离。高斯信道假设下卫星k的最大可达数据率为：

$$R(k, GS) = B \log_2(1 + \text{SNR}(k, GS))$$

我们使用每颗卫星在每次在线时段内与GS的最远距离来推导SNR和速率。卫星k与GS之间交换模型参数**w**的时间为：

$$t_c(k, GS) = \frac{S(\mathbf{w})}{R(k, GS)} + \frac{d(k, GS)}{c}$$

其中S(**w**)/R(k, GS)和d(k, GS)/c分别为传输和传播所需时间，S(**w**)为**w**的数据量（比特）。

---

## III. 所提出的调度算法

如上所述，当卫星与GS之间存在视距链路时（即卫星k处于E_on状态），卫星可以与GS通信。一个值得注意的事实是：地球的自转导致卫星两次访问同一个GS之间的时间间隔与其轨道周期T_p不同。

联邦平均（FedAvg）算法是众所周知且广泛使用的FL过程[4],[10]。用其在卫星上以全客户端参与训练FL模型的[6]大致流程如下：
1. GS在所有卫星访问时向其发送全局模型参数；
2. 卫星使用本地SGD训练模型；
3. 卫星在下一次访问时将更新后的本地参数发送给GS；
4. GS聚合从所有卫星接收到的模型参数。

在地面辅助的卫星FL场景中实现FedAvg会导致模型收敛非常缓慢，因为卫星在不同时间访问GS，GS必须等待所有更新收到后才能开始新的全局epoch。FedAvg算法的一种异步版本——FedSat——在[5]中针对卫星场景被提出，并显示出显著减少收敛时间的效果。在FedSat中，GS在收到任意一颗卫星的更新后本地参数时即更新全局模型参数。

在本文中，我们提出了一种通用方法，有助于在任何形式的卫星星座中实现FL。该方法如图2所示，由三个连续的步骤组成。输入为卫星和GS信息，如卫星数量及其高度、倾角、初始位置，以及GS的位置。

利用这些输入数据，第一步中可以获得每颗卫星与GS之间的访问模式。图3展示了一个24小时内位于不来梅的GS与10颗卫星之间访问模式的示例。其中5颗卫星(S1到S5)位于500km高度，其余5颗(S6到S10)位于2000km高度。在此步骤中推导所有卫星的升起时间τ_rise、落下时间τ_set、在线时间τ_on和离线时间τ_off。

定义访问模式VP为：

$$VP = (\tau_{\text{rise}}, \tau_{\text{set}})$$

第二步中，基于推导出的VP设计调度算法。例如，将在第III-B节中提出的算法使用VP来确定卫星是在离线期间训练下一个模型迭代还是在下次访问GS期间训练。第二步中的调度算法导出第三步中卫星与GS之间的传输时间，即提取交换模型参数的上下行传输所在的时间区间。

定义传输时间序列ST为：

$$ST = (\tau_{UL}, \tau_{DL})$$

其中τ_UL和τ_DL为：

$$\tau_{UL} = (t_{u,1}, t_{u,2}, ..., t_{u,K})$$
$$\tau_{DL} = (t_{d,1}, t_{d,2}, ..., t_{d,K})$$

t_{u,k}和t_{d,k}分别为与第k颗卫星关联的上下行传输时间序列。为获得最优ST，我们建立如下优化问题：

$$ST^* = \arg\max_{ST} C(VP, ST)$$

其中C作为VP和ST的函数，是一个根据具体问题需求定义的期望设计准则。

### A. 卫星星座联邦学习（FedSat）

实现卫星星座FL的一种方法是使用如FedSat[5]中提出的异步算法。通过此方法，我们可以受益于卫星访问模式的可预测性，这有助于克服GS与卫星之间的间歇连接。

在FedSat中，每颗卫星在与GS互访时交换模型参数。这意味着在升起时刻，卫星将其更新后的本地模型参数发送给GS。然后，GS按下式更新全局模型参数：

$$\mathbf{w}^{n+1} = \mathbf{w}^n - \alpha_k (\mathbf{w}^{n_k-1,I}_k - \mathbf{w}^{n_k,I}_k)$$

其中α_k = D_k/D。然后，GS将更新后的模型参数发送给该卫星。接下来，卫星在离线期间训练模型，并在下一次升起时刻将模型参数发送给GS。该算法没有考虑在线和离线时段的持续时间。然而，如果卫星下一次访问GS的时间足够长以在该次访问期间完成训练，则在当前访问时即获取全局模型将导致相当大的模型陈旧度，对收敛产生负面影响。利用这一简单观察正是接下来提出的FedSatSchedule算法背后的核心思想。

### B. 卫星星座联邦学习调度（FedSatSchedule）

在FedSat方案中，如上所述，推导ST的t_{u,k}和t_{d,k}时未考虑每次访问的持续时间（即t_{on,k}）。然而，由于t_{on,k}和t_{off,k}的长度是完全可预测的，可以确定ST使得在更短时间内达到更高的训练准确率。FedSatSchedule方案利用这些时间来调度FL，旨在缩短收敛时间。

在我们的一般框架中将其形式化，(21)可转化为：

$$ST^* = \arg\min_{ST} CT(VP, ST)$$

其中CT为模型收敛时间，它是VP和ST的函数。精确求解此问题具有挑战性，因为连函数关系CT都难以定义。取而代之，我们采用启发式方法，旨在减少卫星处的模型陈旧度，同时确保每颗卫星在每次访问GS时都提供模型更新。具体而言，调度器预测下一次访问GS是否足够长以完成一次本地模型更新。如果是，卫星将在下次接触GS时接收当前全局模型参数。否则，它将立即接收并在离线期间计算其更新。

在FedSatSchedule算法中，在当前在线时段[t^n_{r,k}, t^n_{s,k}]内，第k颗卫星通过比较下一次在线时段的持续时间与训练所需时间来决定所需操作，即判断是否t^{n+1}_{s,k}−t^{n+1}_{r,k} < t_l(k)。图4的流程图详细展示了第n次在线期间需完成的任务。

如果下一次在线时段(t^{n+1}_{s,k}−t^{n+1}_{r,k})短于所需训练时间t_l(k)，卫星请求GS在同一次访问（即第n次在线时段）中发送全局模型参数。然后，卫星利用接收到的全局参数在接下来的离线时段[t^n_{s,k}, t^{n+1}_{r,k}]内训练模型。之后，在第n+1次在线间隔中，它将更新后的参数发送给GS。

反之，如果下一次在线时段长于所需训练时间，卫星将有足够时间在即将到来的在线间隔中使用更新鲜的参数进行训练。注意，在离线间隔中，GS持续基于从其他卫星接收到的参数更新模型参数。那么，第k颗卫星最好等待，并在开始下一次在线间隔的训练前刚好接收更新鲜的模型参数。因此，卫星不会在第n次在线间隔中请求接收新模型参数，而是在第n+1次在线中完成。利用接收到的参数，卫星训练模型并在第n+1次在线中将更新后的模型参数发送给GS。该方法在不增加额外延迟或使用额外资源的情况下带来了更高的准确率。

---

## IV. 数值结果

本节我们通过仿真结果展示所提出方案的有效性。我们考虑10条轨道上的10颗卫星，其中5颗位于500km高度，另外5颗位于2000km高度，GS位于不来梅。不同高度相近轨道之间的最小升交点赤经（RAAN）差为36°。所有卫星的倾角和最小仰角分别设为80°和10°。所有卫星和GS在带宽20MHz的信道上传输模型参数，发射功率为40dBm。收发天线增益均设为6.98dBi。载波频率f_c=2.4GHz，接收机噪声温度T=290K。

训练过程基于[11]，采用著名的CIFAR数据集和ResNet-18模型。学习率η和批次大小分别设为0.1和10。整个CIFAR数据集在所有卫星间按非IID设置划分，使得5个标签分配给500km高度的卫星，另外5个标签分配给2000km高度的卫星。

我们考察了每颗卫星的训练时间t_l(k)对测试准确率的影响。图5展示了三天中三个不同训练时间（30秒、15分钟和30分钟）的测试准确率。结果表明，与FedSat相比，我们提出的调度算法在t_l(k)=30秒和t_l(k)=15分钟的情况下能显著提升测试准确率。

我们观察到，当t_l(k)=30秒时，FedSat需要48小时才能达到约62%的测试准确率，而FedSatSchedule仅需16小时，将收敛速度提高了三倍。FedSatSchedule优于FedSat的原因在于合理的调度使其能够接收更新鲜的模型参数。

随着训练时间间隔的增加，如t_l(k)=30分钟的情况所示，FedSat和FedSatSchedule的性能趋于一致。在此类情况下，所有卫星实际上都必须在离线时段训练模型，因此FedSatSchedule无法从获取更新鲜的模型参数中受益。

---

## V. 结论

在本文中，我们提出了一种通用方法，用于在任何星座中实现FL时优化调度卫星与GS之间模型参数的发送和接收时间。然后，我们专门设计了一种调度算法FedSatSchedule，考虑了每次在线时段的持续时间。数值结果表明，该方案能够加速FL的收敛。

---

## 参考文献

[1] I. Leyva-Mayorga, B. Soret, M. Röper, D. Wübben, B. Matthiesen, A. Dekorsy, and P. Popovski, "LEO small-satellite constellations for 5G and beyond-5G communications," *IEEE Access*, vol. 8, pp. 184955–184964, 2020.

[2] J. M. Haut, M. E. Paoletti, S. Moreno-Álvarez, J. Plaza, J.-A. Rico-Gallego, and A. Plaza, "Distributed deep learning for remote sensing data interpretation," *Proceedings of the IEEE*, vol. 109, no. 8, pp. 1320–1349, 2021.

[3] G. Curzi, D. Modenini, and P. Tortora, "Large constellations of small satellites: A survey of near future challenges and missions," *Aerospace*, vol. 7, no. 9, p. 133, 2020.

[4] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Aguera y Arcas, "Communication-efficient learning of deep networks from decentralized data," *Proc. Mach. Learn. Res. (PMLR)*, vol. 54, 2017.

[5] N. Razmi, B. Matthiesen, A. Dekorsy, and P. Popovski, "Ground-assisted federated learning in LEO satellite constellations," *IEEE Wireless Communications Letters*, pp. 1–1, 2022.

[6] N. Razmi, B. Matthiesen, A. Dekorsy, P. Popovski, "On-board federated learning for dense LEO constellations," in *ICC 2022 - IEEE International Conference on Communications (ICC)*, 2022.

[7] J. So, K. Hsieh, B. Arzani, S. Noghabi, S. Avestimehr, and R. Chandra, "Fedspace: An efficient federated learning framework at satellites and ground stations," arXiv preprint arXiv:2202.01267, 2022.

[8] N. H. Tran, W. Bao, A. Zomaya, M. N. H. Nguyen, and C. S. Hong, "Federated learning over wireless networks: Optimization model design and analysis," in *IEEE INFOCOM 2019*, pp. 1387–1395.

[9] L. J. Ippolito Jr., *Satellite Communications Systems Engineering*. John Wiley & Sons, 2017.

[10] Z. Li and P. Richtárik, "A unified analysis of stochastic gradient methods for nonconvex federated optimization," arXiv preprint arXiv:2006.07013, 2020.

[11] C. He, S. Li, J. So, X. Zeng, M. Zhang, H. Wang, X. Wang, P. Vepakomma, A. Singh, H. Qiu et al., "FedML: A research library and benchmark for federated machine learning," arXiv preprint arXiv:2007.13518, 2020.
