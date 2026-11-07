# 在 Render 上部署 SpaceFL 公网演示（免费 Web Service）

用户希望使用自己电脑的 CPU / GPU 时，可使用 [本地算力连接器](LOCAL_COMPUTE.md)：部署新代码后，平台仅中转任务，保留本页的免费镜像即可。需要在 Render 服务器上上传和训练时，使用 [公网训练升级说明](RENDER_TRAINING.md)。

研究助手默认使用离线规则引擎。当前比赛实例已启用服务端 DeepSeek 并**解除公网 LLM 的调用额度限制**：
在 Render 服务端设置 `DEEPSEEK_API_KEY`（原始密钥）、`SPACEFL_OFFLINE_ONLY=0`、`SPACEFL_PUBLIC_LLM=1`。
访客不需要输入密钥；未登录访客不能调用任何平台 API，登录账号为 `spaceTS` / `11451467`（仅此一个账号）。
密钥不得写在 Blueprint、仓库或网页中；Base64 不提供保密性。请在 DeepSeek 账户侧设置独立消费上限并监测余额。

> 比赛最终作品链接的稳定方案：Render 免费 Web Service（Docker）。
> 说明：Render 免费实例 **15 分钟无请求会休眠**，下次访问冷启动约 30–60 秒；链接域名固定。

## 当前已部署实例（2026-11-07）

| 项 | 值 |
|---|---|
| Service ID | `srv-dbm9qh08v5cc73e0b1lg` |
| 名称 | `spacefl-research-agent` |
| 链接 | **https://spacefl-research-agent.onrender.com** |
| 访问账号 | `spaceTS` / `11451467`（单账号，登录后可用） |
| 仓库 / 分支 | `https://github.com/Bilze-python/fl_space` / `deploy-demo` |
| 计划 / 区域 | Free / oregon |
| 健康检查 | `/api/health`（公开） |
| 环境变量 | `SPACEFL_PUBLIC_DEMO=1`、`SPACEFL_OFFLINE_ONLY=0`、`SPACEFL_PUBLIC_LLM=1`、`SPACEFL_AUTH_ENABLED=1`、`SPACEFL_AUTH_USER=spaceTS`、`SPACEFL_AUTH_PASSWORD=<比赛口令>`、服务端私密 `DEEPSEEK_API_KEY` |
| 当前状态 | **live**；未登录返回 401/跳转登录页，登录后 LLM 可用且无限额 |


## 0. 前置条件

- 一个 Render 账号（免费注册，无需信用卡；可用 GitHub 登录）；
- 代码位于**公开 GitHub 仓库**（本仓库：`https://github.com/Bilze-python/fl_space`，分支 `deploy-demo`）；
- 仓库根目录已包含：`Dockerfile`、`render.yaml`、`requirements-deploy.txt`、`deploy/demo_data/`。

## 1. 一键 Blueprint 部署（推荐）

1. 登录 Render Dashboard；
2. 点击 **New → Blueprint**；
3. 选择 **Public Git repository**，粘贴：
   `https://github.com/Bilze-python/fl_space`
4. 分支选 `deploy-demo`（或包含最新部署代码的分支）；
5. Render 读取根目录 `render.yaml`，生成服务：
   - 类型：Web Service，Runtime: Docker，Plan: Free；
   - 健康检查：`/api/health`；
   - 环境变量：`SPACEFL_PUBLIC_DEMO=1`、`SPACEFL_OFFLINE_ONLY=1`；
6. 点击 **Apply / Create**，等待构建（首次约 5–10 分钟）；
7. 构建完成后得到固定链接：`https://<service-name>.onrender.com`。

## 2. 手动创建 Web Service

| 配置项 | 值 |
|---|---|
| Type | Web Service |
| Source | Public Git repository：`https://github.com/Bilze-python/fl_space` |
| Branch | `deploy-demo` |
| Runtime | Docker |
| Dockerfile Path | `./Dockerfile` |
| Docker Context | `.` |
| Plan | Free |
| Health Check Path | `/api/health` |
| Environment Variables | `SPACEFL_PUBLIC_DEMO=1`、`SPACEFL_OFFLINE_ONLY=1` |

> Dockerfile 已支持 Render 注入的 `PORT` 环境变量（`--port ${PORT:-8720}`），并默认演示模式 +
> 离线规则引擎；镜像只打包 `deploy/demo_data/` 的演示数据，不含本机文献目录、API Key 与历史实验。

## 3. 验收清单

```bash
# 健康检查
curl https://<service-name>.onrender.com/api/health
# 演示模式确认（demo_mode=true；当前服务 ai.configured=true）
curl https://<service-name>.onrender.com/api/research/config
# 总览数据（文献 6 / 实验 3 / 情报 48）
curl https://<service-name>.onrender.com/api/research/overview
```

浏览器应能完成：文献检索 → AI 精读 → 知识库问答 → 自然语言实验计划 → 实验档案结果解读；
训练/导入/配置写入按钮应被禁用（返回 403）。

## 4. 免费实例注意事项

- 休眠：15 分钟无请求后休眠，冷启动会导致首次打开较慢；提交材料中可注明“首次打开约需 1 分钟”。
- 资源：0.1 CPU / 512 MB RAM，登录 + 只读演示（服务端 DeepSeek 失败时回退离线规则引擎）足够；不要启用训练任务。
- 磁盘：临时文件可写但不持久；每次冷启动会重新构建演示文献索引（秒级）。
- 构建缓存：首次构建含 Cesium 前端依赖，约 5–10 分钟；后续提交代码自动重新部署。

## 5. 更新部署

1. 本地修改后推送到公开仓库对应分支：

```powershell
git add -A
git commit -m "update demo"
git push origin HEAD:deploy-demo
```

2. Render 开启 `autoDeploy: true` 时会自动重新构建；也可在 Dashboard 点击 **Manual Deploy**。

## 6. 故障处理

| 现象 | 处理 |
|---|---|
| 构建失败（npm ci） | 检查 `package-lock.json` 与 `package.json` 是否一致；本地 `npm ci --omit=dev` 验证 |
| 构建失败（pip） | 检查 `requirements-deploy.txt`；本镜像只装运行依赖 |
| 502/启动失败 | 查看 Render Logs；确认服务监听 `0.0.0.0:$PORT`（Dockerfile 已处理） |
| 健康检查失败 | 确认 `/api/health` 返回 200；首次构建需等待 warm-up 完成 |
| 页面 403 | 正常：公网演示模式禁用了训练/导入/配置写入 |
| 首次访问很慢 | 免费实例休眠后冷启动；稍等 30–60 秒 |
