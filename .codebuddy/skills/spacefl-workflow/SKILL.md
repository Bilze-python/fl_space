---
name: spacefl-workflow
description: 太空联邦学习(SpaceFL)多智能体协作工作流。涵盖轨道设计→FL实验→仿真执行→数据分析→报告生成全流程。
---

# SpaceFL 多智能体协作工作流

## 概述

本 Skill 为 `D:\fl_space` 太空联邦学习研究框架提供完整的**多智能体协作工作流**。5 个专业智能体接力协作，从需求到报告一站式完成。

## 工作流程

```
用户需求
   │
   ▼
┌──────────────┐
│ Orbit        │  — 设计星座参数(v)
│ Architect    │  — 配置地面站
│              │  — 生成模拟器配置
└──────┬───────┘
       │ satellite_specs, gs_config, sim_config
       ▼
┌──────────────┐
│ FL           │  — 选择FL算法
│ Engineer     │  — 配置模型/数据集
│              │  — 生成实验JSON
└──────┬───────┘
       │ experiment_config
       ▼
┌──────────────┐
│ Simulation   │  — 运行fl-space CLI
│ Runner       │  — 监控执行进度
│              │  — 收集原始结果
└──────┬───────┘
       │ raw_results (JSON/CSV/PNG)
       ▼
┌──────────────┐
│ Data         │  — 统计准确率/损失
│ Analyzer     │  — 算法对比分析
│              │  — 延迟/吞吐量计算
└──────┬───────┘
       │ analysis_results
       ▼
┌──────────────┐
│ Report       │  — 生成最终图表
│ Generator    │  — 撰写实验报告
│              │  — 汇总输出文件
└──────┬───────┘
       │
       ▼
   最终报告 + 可视化输出
```

## 智能体角色

| 智能体 | 职责 | 核心模块 |
|--------|------|----------|
| `orbit-architect` | 轨道设计、星座配置、可见性分析 | `orbit/`, `simulator/`, `config/` |
| `fl-engineer` | FL算法选择、实验配置、参数调优 | `fl/`, `config/` |
| `simulation-runner` | CLI执行、进度监控、结果收集 | `cli.py`, `output/` |
| `data-analyzer` | 结果分析、统计对比 | `utils/`, `viz/` |
| `report-generator` | 可视化图表、综合报告 | `viz/`, `docs/` |

## 启动方式

### 方式1: 对话触发（最简单）
```
@spacefl-workflow 我需要做一个 [实验描述]
```

### 方式2: team_create 显式创建
```
请创建 spacefl-workflow 团队，我要做个LEO星座下的FedProx对比实验
```

### 方式3: 逐步串行调用（精细控制）
```
启动 orbit-architect 设计星座
→ 把结果传给 fl-engineer 配置实验
→ 传给 simulation-runner 执行
→ 传给 data-analyzer 分析
→ 传给 report-generator 出报告
```

## 关键文件参考

- `D:\fl_space\pyproject.toml` — 项目配置与依赖
- `D:\fl_space\fl_template.json` — FL实验配置模板
- `D:\fl_space\sim_template.json` — 模拟器配置模板
- `D:\fl_space\examples\run_spacefl_experiment.py` — 完整实验示例脚本
- `D:\fl_space\fl_space\cli.py` — 命令行入口
- `D:\fl_space\fl_space\fl\server.py` — FL训练核心逻辑
- `D:\fl_space\fl_space\utils\viz.py` — 可视化工具

## 常用CLI命令

```bash
# 模拟轨道
fl-space simulate --sats 10 --gs 1 3 5 --back-end kepler --sim-hours 24

# 运行FL实验
fl-space experiment --sats 10 --gs 1 --rounds 300 --device cpu

# GPU + 并行训练
fl-space experiment --device cuda --train-workers 4 --data-workers 4

# 自定义轨道高度
fl-space experiment --altitudes 350 420 500 580 660 740 --gs 1 3 5

# 查看实验数据
fl-space list
fl-space info <experiment_id>

# 导出结果
fl-space export <experiment_id> --format json
```
