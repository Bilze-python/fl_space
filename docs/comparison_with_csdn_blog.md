# SpaceFL vs CSDN 博客项目 对比分析

> 对比对象：CSDN 博客《在 MATLAB/SIMULINK 中构建一个支持联邦学习的多卫星协同通信系统》vs SpaceFL

## 博客项目概况

- **平台**: MATLAB/Simulink（商业闭源，需 Deep Learning + Communications + RL 3个付费工具箱）
- **任务**: 卫星链路质量预测 + 路由优化
- **算法**: 仅 FedAvg（自定义加权平均）
- **模型**: LSTM（3→20→10→2，~500参数）
- **数据**: 仿真合成信道数据（SNR/多普勒/距离）
- **轨道**: ❌ 无轨道模型，纯信道参数模拟
- **通信**: To/From Workspace 模拟星地传输
- **结果**: 89% 准确率，8轮收敛

## SpaceFL 项目概况

- **平台**: Python/PyTorch（全开源）
- **任务**: 通用 FL 图像分类（可扩展任意任务）
- **算法**: FedAvg + FedProx + FedBuff（可插拔四组件架构）
- **模型**: MLP(109K) + SimpleCNN(1.1M)，模型注册机制
- **数据**: MNIST / Fashion-MNIST / CIFAR-10（真实基准）
- **轨道**: ✅ Kepler + Skyfield/SGP4 双后端
- **通信**: ✅ 基于轨道位置的接触矩阵 + CommunicationScheduler
- **时间**: ✅ 可插拔 SlotTimeModel + PhysicsTimeModel

## 12维度深度对比

| 维度 | CSDN博客 | SpaceFL |
|------|:---:|:---:|
| 开源免费 | ❌ MATLAB商业 | ✅ Python全开源 |
| 轨道物理 | ❌ 无 | ✅ 双后端(Kepler+SGP4) |
| FL算法数 | 1 (FedAvg) | **3** (FedAvg+Prox+Buff) |
| 可插拔架构 | ❌ 硬编码 | ✅ 4组件+时间模型 |
| 真实数据集 | ❌ 合成数据 | ✅ 3个基准 |
| CLI工具 | ❌ 无 | ✅ 5个子命令 |
| 代码规范 | ❌ | ✅ Ruff+文档+CI |
| 星座灵活性 | ❌ 固定5颗 | ✅ 3分布+多星簇 |
| 通信真实性 | ❌ 无窗口模型 | ✅ 接触矩阵 |
| 时间粒度 | 固定30s | ✅ 可插拔双模式 |
| 适用领域 | 链路质量预测 | **通用FL研究** |
| 应用范围 | 通信 | ML/CV+通信 |

## 核心差异一句话

CSDN博客是**Simulink教学案例**（怎么在MATLAB里跑FL），SpaceFL是**面向研究的开源框架**（轨道约束下FL算法的收敛行为研究）。

## 项目运行逻辑

```
CLI → OrbitSimulator(轨道模拟) → ContactMatrix(接触矩阵)
    → CommunicationScheduler(通信调度)
    → FLServer(同步/异步训练) → TimeModel(时间投影)
    → FLRoundResult(结果+时间分解) → JSON导出
```
