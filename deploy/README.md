# SpaceFL 公网演示部署（可部署 Agent 作品链接）

本目录提供把 SpaceFL 科研智能平台以**公网可访问的只读演示模式**部署出去的最小方案，
用于比赛提交的“作品链接”。

## Docker 镜像与公开数据

镜像使用 `node:22-bookworm-slim` 安装 Cesium 等前端依赖，再用
`python:3.11-slim` 运行 FastAPI。镜像仅复制 `fl_space/`、`web/` 和
`deploy/demo_data/`，不包含本机配置、API Key、完整文献目录和历史实验目录。
容器以非 root 用户运行，默认启用公开演示及离线规则模式。

```powershell
docker build -t spacefl-public-demo .
docker run --rm --name spacefl-public-demo -p 8720:8720 spacefl-public-demo
```

启动后访问 `http://127.0.0.1:8720/`，健康检查为
`http://127.0.0.1:8720/api/health`。公开文献只来自
`deploy/demo_data/library/`；公开实验只来自
`deploy/demo_data/experiments/`。当前演示数据是一组 12 星、5 轮、
seed 42 的真实 FedLEO 卸载对照功能验证，不能宣称为论文级复现。

需要从本机原始结果重新生成脱敏副本时运行
`pwsh -File deploy/prepare_demo_data.ps1`，随后检查生成的 JSON 再构建镜像。
部署到 Hugging Face Docker Space 时，在 Space 的 `README.md` 元数据里设置
`sdk: docker` 和 `app_port: 8720`；比赛提交应使用 Space 的固定 URL。

## 一、最快路径：本机 + SSH 隧道（localhost.run，零成本，2 分钟）

1. 双击 `deploy\start_public_demo.bat`；
2. 脚本会：
   - 在 **8720 端口**启动演示模式服务器（`SPACEFL_PUBLIC_DEMO=1` + `SPACEFL_OFFLINE_ONLY=1`），
     只读取 `deploy/demo_data/`（演示文献与演示实验），与本地 8700 完整平台隔离；
   - 通过本机 OpenSSH 建立 `https://<随机名>.lhr.life` 反向隧道（无需下载额外程序）；
   - 注册 Windows 计划任务 `SpaceFLPublicDemo`（每 5 分钟保活），并创建开机启动快捷方式；
   - 把当前公网地址写入 `deploy\public_url.txt`。
3. 把 `deploy\public_url.txt` 中的地址作为作品链接提交。

> 说明：localhost.run 免费隧道地址为随机子域名；**隧道重连后地址会变化**，保活任务会自动重写
> `deploy\public_url.txt`。提交前请记录当前地址；如需长期稳定链接，请使用第 2/3 节的
> Hugging Face Space 或自有服务器方案。

如果网络可以访问 GitHub Release（`cloudflared.exe` 下载正常），保活脚本也支持
Cloudflare 快速隧道（`trycloudflare.com`）：把 `cloudflared.exe` 放到 `deploy\bin\` 即可自动优先使用。
本机网络当前会重置 GitHub Release 下载，因此默认走 SSH 隧道。

常用命令：

```powershell
deploy\start_public_demo.bat     # 一键启动/刷新公网演示，输出链接
deploy\status_public_demo.bat    # 查看链接、本地服务、隧道与计划任务状态
deploy\stop_public_demo.bat      # 停止演示（不影响本地 8700 完整平台）

Get-Content deploy\public_url.txt        # 查看公网地址
Get-Content deploy\logs\keepalive.log -Tail 50
Invoke-RestMethod http://127.0.0.1:8720/api/research/config
```

## 1.5 公开演示数据（脱敏）

`deploy/demo_data/` 是公网实例唯一可见的数据目录：

- `library/`：6 篇原创演示笔记（FedLEO 卸载机制、LEO 调度要点、评测指标、复现方法、平台边界等），
  用于展示全文检索、AI 精读、知识库问答；
- `experiments/`：3 个真实 FedLEO 卸载对照功能验证档案（已去除本机路径/设备等字段），
  用于展示实验档案与结果解读；
- 重新生成：`pwsh -File deploy/prepare_demo_data.ps1`（从本机原始结果生成脱敏副本）。

公开实例不读取 `文献/`、`docs/`、`experiments/` 与任何本机配置，也不包含 API Key。

## 二、稳定链接方案 A：Render 免费 Web Service（当前提交方案）

完整步骤见 [`../docs/RENDER_DEPLOY.md`](../docs/RENDER_DEPLOY.md)。最简流程：

1. 用 GitHub 登录 [Render Dashboard](https://dashboard.render.com)（免费，无需信用卡）；
2. **New → Blueprint** → 选择 **Public Git repository** → 填入
   `https://github.com/Bilze-python/fl_space`，分支选 `deploy-demo`；
3. Render 读取根目录 `render.yaml`：Docker 运行时、Free 计划、健康检查 `/api/health`、
   环境变量 `SPACEFL_PUBLIC_DEMO=1` 与 `SPACEFL_OFFLINE_ONLY=1`；
4. 首次构建约 5–10 分钟，完成后得到固定域名 `https://<service-name>.onrender.com`。

> Render 免费实例 **15 分钟无请求会休眠**，首次打开需冷启动 30–60 秒；域名固定、无需本机开机，
> 比 SSH 隧道更适合作为最终作品链接。镜像只包含 `deploy/demo_data/` 的演示数据。

## 三、稳定链接方案 B：Hugging Face Spaces（需要 PRO 账号）

> 注意：2026-11 当前 HF 免费账号创建 Docker/Gradio Space 会返回 402，需要 PRO 订阅（约 $9/月）。
> 若账号为 PRO，可直接用仓库根目录 `Dockerfile` 部署；否则请用 Render 或自有服务器。

1. 在 Hugging Face 创建 **Docker Space**，上传本仓库（或连接 GitHub 仓库）；
2. 直接用仓库根目录的 `Dockerfile`（已内置 `SPACEFL_PUBLIC_DEMO=1`、`SPACEFL_OFFLINE_ONLY=1`；
   `PORT` 缺省时监听 8720）；
3. Space 会自动构建并给出永久链接 `https://<user>-<space>.hf.space`；
4. 镜像会安装前端依赖，只打包 `deploy/demo_data/` 内筛选过的公开数据；
   Agent 使用离线规则引擎，响应中的 `mode` 会如实标记；
5. 也可用 `deploy/huggingface/upload_space.py` 脚本上传（需要 HF write token）：
   ```powershell
   $env:HF_TOKEN = "hf_..."
   venv\Scripts\python.exe deploy\huggingface\upload_space.py --repo-id <user>/spacefl-research-agent
   ```

## 四、稳定链接方案 C：自有云服务器（最稳）

1. 将仓库部署到服务器（`/opt/fl_space`），创建 venv 并安装：
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements-deploy.txt
   ```
2. 使用 `deploy/systemd/spacefl.service` 启动演示服务（127.0.0.1:8720）；
3. 用 Nginx/Caddy 反向代理并配置 HTTPS（示例见 `deploy/Caddyfile`）；
4. 域名即可作为作品链接。

## 五、公网演示模式做了什么

设置 `SPACEFL_PUBLIC_DEMO=1` 后：

- 白名单：仅允许 Agent 规划/评审/分析/对话、文献问答/精读、情报刷新/简报（GET 全部放行）；
- 其余写操作（训练、导入、索引重建、配置修改、实验删除、会话修改等）返回 403；
- 文献文件读取限制在文献库根目录内，禁止读取项目内任意文件；
- LLM 接口按 IP 限流 20 次/分钟；情报强制刷新有 120 秒冷却；
- 前端“研究助手”页显示“公网演示模式 · 只读”，并禁用运行计划按钮。

设置 `SPACEFL_OFFLINE_ONLY=1` 后：不读取本机 CC Switch / API Key，统一走离线规则引擎，
响应更快且不会把供应商信息暴露到公网。

## 六、故障处理

| 现象 | 处理 |
|---|---|
| public_url.txt 为空 | 查看 `deploy\logs\keepalive.log` 与 `deploy\logs\ssh-tunnel.out.log`；确认能 `ssh -R 80:127.0.0.1:8720 nokey@localhost.run` 连通 |
| 公网 502/无法访问 | 确认本地 `http://127.0.0.1:8720/api/health` 正常；运行 `deploy\start_public_demo.bat` 重新建立隧道 |
| 链接地址变了 | 隧道重连会生成新地址；读取 `deploy\public_url.txt` 更新提交材料 |
| 端口 8720 被占用 | 改用其他端口：运行 `install_public_demo.ps1 -Port 8730`（并同步修改脚本与计划任务） |
| 计划任务未生效 | 以管理员身份重新运行 `install_public_demo.ps1`；或手动运行 `deploy\run_keepalive.bat` |
| 演示模式误拦截 | 检查环境变量 `SPACEFL_PUBLIC_DEMO`（必须为 1）；本地完整体验请用 8700 端口 |
| 演示数据为空 | 运行 `pwsh -File deploy/prepare_demo_data.ps1` 重新生成 `deploy/demo_data/` |

