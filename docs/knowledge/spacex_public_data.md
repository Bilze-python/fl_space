# SpaceX / Starlink 公开资料知识库

**维护日期**：2026-09-16
**用途**：为 SpaceFL 初步仿真提供可追踪输入。本文是本地知识库，不是 SpaceX
官方工程规格。

## 来源边界

- `SpaceX-API-master` 来源于 `r-spacex/SpaceX-API`，是社区维护的 SpaceX 目录 API。
- API 可提供 launches、rockets、payloads、starlink 等目录字段。
- `docs/starlink/v4/schema.md` 显示 Starlink 记录含位置、高度、速度和 Space-Track
  OMM 字段，例如 `SEMIMAJOR_AXIS`、`ECCENTRICITY`、`INCLINATION`、
  `RA_OF_ASC_NODE`、`ARG_OF_PERICENTER`、`MEAN_ANOMALY`、TLE 行等。
- `docs/payloads/v4/schema.md` 显示 payload 记录含 `mass_kg`、`orbit`、
  `semi_major_axis_km`、`eccentricity`、`inclination_deg`、`period_min` 等字段。
- 该 API 没有公开 Starlink 激光终端的波长、发射功率、调制编码、指向误差、
  接收灵敏度或热控参数。

## Starlink 光链路公开信息

公开网页和报道可作为仿真假设候选：

| 信息 | 可用于仿真的字段 | 可信级别 |
| --- | --- | --- |
| Starlink 官方技术页面公开介绍卫星使用空间激光链路并强调全球覆盖能力 | `band="optical"`、`link_kind="inter_satellite"` | 官方页面，但不是完整工程参数 |
| 公开资料常见表述为每颗卫星使用多个激光链路，链路峰值可到百 Gbps 量级 | `terminal_count`、`isl_rate_mbps` 上限候选 | 需要逐条来源核验 |
| 迷你 Starlink 激光终端相关公开报道给出约 `25 Gbps` 和约 `4000 km` 最大距离 | `isl_rate_mbps=25000`、`max_range_km=4000` | 公开报道/合作信息，作为 assumption |

代码中 `public_starlink_optical_link_profile()` 默认采用：

```text
band = optical
link_kind = inter_satellite
isl_rate_mbps = 25000
max_range_km = 4000
source = public_report_assumption
```

这些值只能用于早期模拟器评估，不应写成“持续可用带宽”或“官方 Starlink 卫星规格”。

## NVIDIA 太空方向公开信息

NVIDIA 公开资料显示其太空方向重点是把边缘 AI、GPU 加速和在轨数据处理带入卫星：

- Space Computing / Jetson Orin 相关发布强调在轨 AI 处理和空间计算；
- 适合作为 SpaceFL 的“星上计算资源”背景，而不是轨道或链路物理数据来源；
- 对本项目可转化为 `ModelSpec.framework`、节点计算能力、训练能耗和边缘推理任务。

## 本项目使用方式

1. API 目录数据通过 `fl_space.integrations.spacex_api.SpaceXApiClient` 只读获取；
2. Starlink 轨道数据通过 `orbital_elements_from_starlink()` 转为二体六根数；
3. 载荷质量数据通过 `satellite_design_from_payload()` 转为卫星设计草案；
4. 新闻和网页参数只写入 `assumptions` 或 `source` 字段；
5. 实验报告需保存访问时间、来源 URL、API 快照和转换后的配置。

## 建议保存的来源链接

- [Starlink Technology](https://www.starlink.com/technology)
- [SpaceX API GitHub](https://github.com/r-spacex/SpaceX-API)
- [NVIDIA Newsroom](https://nvidianews.nvidia.com/)

Feishu 文档链接可作为协作文档入口，但当前本地环境未取得可写授权；因此本次先写入
仓库本地知识库。
