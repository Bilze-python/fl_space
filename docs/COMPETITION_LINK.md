# 作品链接与公网演示说明

> 比赛提交要求：**可访问的作品链接（可部署 Agent）**。
> 本文件记录当前公网链接、访问方式、保活机制与稳定化方案，可直接附在提交材料里。

---

## 1. 当前作品链接

### 主链接（Render 免费 Web Service，固定域名）

**https://spacefl-research-agent.onrender.com**

- 部署：Render 免费 Web Service（Docker），代码来自公开仓库分支 `deploy-demo`；
- 演示模式：只读 + 离线规则引擎 + `deploy/demo_data` 演示数据；
- 免费实例 **15 分钟无请求会休眠**，首次打开约 30–60 秒冷启动，之后由评审访问自动唤醒；
- 服务状态与日志：Render Dashboard → `spacefl-research-agent`（Service ID `srv-dbm9qh08v5cc73e0b1lg`）。

### 上线验收（2026-11-05）

| 检查项 | 结果 |
|---|---|
| `/api/health` | 200，version 0.3.1 |
| `/api/research/config` | `demo_mode=true`、`ai_configured=false`、索引 6 篇 |
| `/api/research/overview` | 文献 6 / 实验档案 3 / 情报 48 |
| `/` 首页 | 200，SpaceX 风格 Hero 正常 |
| `POST /api/research/agent/run` | 403（公网只读拦截生效） |
| 自然语言实验计划 | 1.6s 返回（离线模式，条件/评审/引用齐全） |
| 实验档案解读 | 返回指标、证据链与 4 条文献引用 |
| 文献检索 | 中文“卸载”返回 3 条结果 |


### 备用入口（Hugging Face 免费静态 Space）

**https://bilzepython-spacefl-agent-demo.static.hf.space**

- 固定域名的静态外壳，读取 `live.json` 并嵌入当前在线演示；保活脚本在隧道地址变化时自动更新；
- 用于 Render 审核/冷启动期间或比赛材料中的备用链接。

### 实时隧道（仅调试/备用）

- 地址实时变化，记录在 `deploy/public_url.txt`；依赖本机开机联网，不作为最终提交链接。



### 打开后能体验什么

1. **总览**：Hero 指标、研究目标、研究流水线（文献/情报/实验档案均为预置演示数据）；
2. **文献库**：6 篇演示文献的全文检索、AI 精读（摘要/贡献/局限/相关度）、知识库问答（带页码引用）；
3. **前沿情报**：arXiv/航天资讯缓存、按研究目标相关度排序、情报简报；
4. **研究助手**：自然语言 → 实验计划（条件/对照/指标/预计耗时）+ 计划评审分数 + 文献依据；
   对预置 3 个实验档案做结果解读（指标对照、证据链、文献对照、尚不能支持的结论）；
5. **公网演示模式**：训练/导入/配置写入被禁用，页面会显示“公网演示模式 · 只读”。

> 大模型未在公网实例启用（为避免费用与密钥泄露），Agent 使用**离线规则引擎**；
> 界面会如实显示“离线规则引擎”，能力与引用流程完整可演示。

---

## 2. 部署架构（本机 + SSH 隧道）

```
评委浏览器
   │ HTTPS
   ▼
localhost.run（公网边缘）  ──SSH 反向隧道──►  本机 OpenSSH
                                               │
                                               ▼
                                  演示服务器 127.0.0.1:8720
                                  （SPACEFL_PUBLIC_DEMO=1, OFFLINE_ONLY=1）
```

- 演示服务器与本地完整体验（8700 端口）**数据隔离**：
  公网实例只读取 `deploy/demo_data/`（6 篇演示文献 + 3 个演示实验档案）；
- 安全加固：只读白名单（Agent 规划/评审/分析/对话、文献问答/精读、情报刷新/简报）、
  每 IP 20 次/分钟限流、文献文件读取限制在演示目录内、禁止训练/导入/配置写入。

---

## 3. 保活机制

| 组件 | 机制 |
|---|---|
| 演示服务器 | 保活脚本检查 `http://127.0.0.1:8720/api/health`，异常则重启（`SPACEFL_PUBLIC_DEMO=1`） |
| 公网隧道 | 检查本地 `localhost.run` SSH 进程与已保存 URL 是否健康，异常则重连并写回新 URL |
| 计划任务 | `SpaceFLPublicDemo` 每 2 分钟运行一次 `deploy/run_keepalive.bat` |
| 开机启动 | 用户启动文件夹快捷方式 `SpaceFLPublicDemo.lnk`（登录后自动执行保活） |
| HF 静态入口 | 隧道 URL 变化时，保活脚本调用 `deploy/huggingface/update_static_space.py` 更新 `live.json` |

常用命令：

```
deploy\start_public_demo.bat     # 一键启动/刷新公网演示，输出链接
deploy\status_public_demo.bat    # 查看链接、本地服务、隧道与计划任务状态
deploy\stop_public_demo.bat      # 停止演示（不影响本地 8700 完整平台）
```

日志：`deploy/logs/keepalive.log`、`deploy/logs/ssh-tunnel.out.log`、`deploy/logs/server.err.log`。

---

## 4. 稳定链接升级方案（建议 11-08 前确定）

| 方案 | 稳定性 | 需要资源 | 操作 |
|---|---|---|---|
| **Render 免费 Web Service（当前提交方案）** | ★★★ 固定域名，常驻（空闲休眠） | Render 账号（已配置） | `render.yaml` + 根目录 `Dockerfile`；服务名 `spacefl-research-agent` |
| **Hugging Face 免费静态入口** | ★★ 固定域名，内嵌实时隧道 | 无（已创建） | `https://bilzepython-spacefl-agent-demo.static.hf.space`，自动同步 `live.json` |
| **Hugging Face Docker Space** | ★★★ 永久链接 | HF PRO 订阅 + Token | 免费账号创建 Docker Space 会返回 402；PRO 后使用根目录 `Dockerfile` |
| **自有云服务器 + 域名** | ★★★ 可控 | 服务器 SSH + 域名 | 使用 `deploy/systemd/spacefl.service` + `deploy/Caddyfile` 反代 8720 |
| 本机 SSH 隧道（调试） | ★★ 机器需常开，重启后换 URL | 无 | 已配置 2 分钟保活任务，`public_url.txt` 自动更新 |

> 若切到 Render/HF PRO/云服务器，只需把提交材料里的链接替换为新固定域名；
> 代码、演示数据与安全模式无需修改（Docker 镜像默认读取 `deploy/demo_data` 并按演示模式运行）。

---

## 5. 提交前检查清单

- [x] 公网链接可打开（HTTPS，无需登录）
- [x] Agent 主流程可用：文献检索/精读/问答、情报、自然语言实验规划、实验档案解读
- [x] 公网演示模式只读（训练/导入/配置写入返回 403）
- [x] 演示数据已脱敏（仅 `deploy/demo_data` 的演示文献与演示实验）
- [x] 保活任务 + 开机启动已配置
- [ ] 最终提交前再次确认链接可用（建议 11-09 与 11-10 各检查一次）
- [ ] 若改用 HF Space / 云服务器，更新本文件与提交材料的链接
