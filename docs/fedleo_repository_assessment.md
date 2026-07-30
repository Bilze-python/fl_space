# FedLEO 外部仓库评估与集成决策

评估对象：<https://github.com/teleportup/FedLEO-Federated-Learning>（检查版本 `36d3eb4`）  
论文基准：Zhai et al., 2024, *FedLEO: An Offloading-Assisted Decentralized Federated Learning Framework for Low Earth Orbit Satellite Networks*。

## 结论

外部仓库不符合论文 FedLEO 的核心算法，不应作为 SpaceFL 的 FedLEO 后端封装。它实现的是 4 个虚拟客户端、一个固定地面站服务器和同步加权 FedAvg。接入它会从当前的 PyTorch 技术栈额外引入 TensorFlow、pandas 和 scikit-learn，同时丢失当前已有的卸载、星间邻接和分层聚合能力。

## 逐项对照

| 论文能力 | 外部仓库 | SpaceFL 当前实现 |
|---|---|---|
| 无中心服务器 | 否，固定 `GroundStation` | 是，分层聚合语义 |
| 同轨 Ring-Allreduce | 无 | 用等价聚合结果模拟 |
| 跨轨协作聚合 | 无 | 用轨道面间加权聚合模拟 |
| ISL 数据卸载 | 无 | 有静态邻接图卸载 |
| 阈值卸载策略 | 无 | 离散比例搜索近似 |
| 系统级贪心迭代 | 无 | 有离散贪心迭代 |
| 非 IID 数据 | 连续切片，接近 IID | 固定类别数 non-IID |
| 时延/通信功率约束 | 无 | 有轻量时隙模型；无 KKT 功率优化 |
| 权重散度 | 无 | 支持集中式参考权重散度；规划阶段用均衡熵代理 |
| 轨道拓扑 | 仅给每星设置显示用高度 | 静态 Walker 风格轨道面邻接 |

## 外部仓库优点

- 单文件可运行演示，适合初学者快速理解普通 FedAvg。
- 提供逐节点训练日志、测试集评估和模型参数量展示。
- MIT 许可证允许复用，但本次没有复制其代码。

这些产品化优点已用更轻量的方式吸收：结果文件增加模型复杂度和实现边界元数据，可视化脚本增加 FedLEO 时延、卸载、均衡度和散度图，不引入 TensorFlow。

## SpaceFL 的准确定位

SpaceFL 当前是“论文结构感知的轻量离散仿真”，不是公式级复现。已经覆盖卸载、贪心选择、同轨/跨轨聚合和关键指标，但仍近似了闭式阈值解、通信功率优化、动态链路、多跳流竞争和真实 Ring-Allreduce 传输过程。

可通过以下命令随时查看这一边界：

```powershell
fls run fedleo --implementation-info
```

生成 FedLEO 图表：

```powershell
python scripts/generate_result_visuals.py fedleo_output --plots accuracy,time,offload,summary
```
