# SpaceFL 开发讨论纪要 (2026-06-01 ~ 2026-06-02)

> 整理自开发对话，供组员参考。

---

## 1. Bug 修复 (06-01)

### 已修复的 4 个 Bug（由易到难）

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| Bug2 | `celestial_body.py:296` | `from_dict` 用 `**d` 解包，多余字段导致报错 | 改为过滤 `dataclass_fields` 中的有效键 |
| Bug4 | `satellite_phases.py:45` | `generate_walker_phases` 缺参数校验 | 添加 `if num_planes > num_satellites: raise ValueError` |
| Bug3 | `cli.py:388` | 导出 JSON 用 `args.algo` 而非局部变量 | 改为局部变量 `algo` |
| Bug1 | `orbit_simulator.py:520` | `summary()` 在 `constellation_config=None` 时空指针 | 改为三元表达式兜底 |

---

## 2. 通信驱动的轮次推进重写 (06-01)

### 问题
旧代码每轮固定增加 `timeslots_per_round=10` 个时隙，与轨道实际通信窗口无关。

### 解决方案
重写 `FLServer.run_sync`，改为**通信窗口驱动**的 `while` 循环：

```
while completed_rounds < num_rounds:
    1. 扫描找到下一个有卫星连接的时隙 → 分发窗口
    2. 卫星本地训练（不占虚拟时间）
    3. 对每个卫星，用 sim.get_next_contact() 找下一次接触 → 返回窗口
    4. current_ts = max(各卫星返回时间) → 聚合
    5. 聚合后 +1 时隙，进入下一轮
```

### 关键发现
- **训练时间分析**：当前 MLP 模型（109K 参数）+ MNIST（12K 样本/卫星 × 5 epoch）≈ 39.2 GFLOPs，即使在老旧抗辐射处理器（200 MFLOPS）上也只需约 3 分钟。**5 小时的窗口间隔绰绰有余。**
- 卫星 FL 的真正瓶颈是**通信窗口稀缺**和**轨道等待时间**，不是训练计算。

---

## 3. 虚拟时间投影方案讨论 (06-01 ~ 06-02)

### 当前模型的局限性
- 训练时间 = 0（瞬间完成）
- 模型传输时间 = 0
- 只有"轨道等待"一个时间成本来源
- 无法研究 epoch 数增加的时间代价

### 三种方案

| 方案 | 描述 | 适用场景 |
|------|------|---------|
| **A** | 最小侵入：可配置训练 timeslot（`slots_per_epoch`） | 研究 epoch 增加对收敛速度的影响 |
| **B** | 双时钟物理精度：FLOPs/带宽计算秒级时间 | 论文级物理真实性，硬件对比研究 |
| **C** | 增强输出：时间分解展示（等待/训练/传输） | 分析瓶颈，验证模型正确性 |

### 最终决策：A + C 作为默认，B 作为可选扩展

用户可根据研究目标自由切换：
- 研究轨道约束 → 用 slot 模型（训练成本可设为 0）
- 研究训练-通信权衡 → 用 slot 模型（设 `slots_per_epoch > 0`）
- 论文级物理真实 → 用 physics 模型

---

## 4. 可插拔时间模型实现 (06-02)

### 架构

```
fl_space/fl/time_model.py

TimeModel (ABC)
├── compute_train_slots(client_id, num_samples, num_epochs) → int
├── compute_download_slots(model_size_bytes) → int
├── compute_upload_slots(model_size_bytes) → int
└── 工厂方法: TimeModel.create("slot"|"physics"|"path:ClassName")

TimeBreakdown (dataclass)
├── wait_distribution, download, train, wait_return, upload, aggregation
├── total, per_satellite
└── summary_str(), to_dict()

SlotTimeModel (方案 A+C)
├── slots_per_epoch: int = 0       # 每 epoch 消耗 timeslot 数
├── slots_per_mb_down: int = 0     # 下载每 MB 消耗 timeslot 数
├── slots_per_mb_up: int = 0       # 上传每 MB 消耗 timeslot 数
└── 所有成本以 timeslot 为单位，简单可控

PhysicsTimeModel (方案 B)
├── compute_gflops: float = 10.0    # 卫星算力 (GFLOPS FP32)
├── downlink_mbps: float = 10.0    # 下行带宽 (Mbps)
├── uplink_mbps: float = 1.0       # 上行带宽 (Mbps)
├── flops_per_sample_forward: float # 每样本前向 FLOPs (默认 109184)
└── 训练: total_FLOPs / compute_gflops → 秒 → 向上取整 timeslot
    传输: model_bits / bandwidth → 秒 → 向上取整 timeslot
```

### 使用方式

```bash
# 零训练时间（等价旧行为）
fl-space train --algo fedprox --dataset mnist --rounds 10 \
    --time-model slot --time-model-args '{"slots_per_epoch":0}'

# 训练有成本（每 epoch 耗时 1 个 timeslot）
fl-space train --algo fedprox --dataset mnist --rounds 10 \
    --time-model slot --time-model-args '{"slots_per_epoch":1}'

# 训练 + 传输都有成本
fl-space train --algo fedprox --dataset mnist --rounds 10 \
    --time-model slot \
    --time-model-args '{"slots_per_epoch":1,"slots_per_mb_down":2,"slots_per_mb_up":3}'

# 物理精度（树莓派级别硬件）
fl-space train --algo fedprox --dataset mnist --rounds 10 \
    --time-model physics \
    --time-model-args '{"compute_gflops":10,"downlink_mbps":10,"uplink_mbps":1}'

# 自定义时间模型（从文件导入）
fl-space train --algo fedprox --dataset mnist --rounds 10 \
    --time-model path/to/my_time_model.py:MyTimeModel
```

### 代码中使用

```python
from fl_space.fl.time_model import SlotTimeModel, PhysicsTimeModel, TimeModel

# 方式1：内置工厂
tm = TimeModel.create("slot", slots_per_epoch=2)
tm = TimeModel.create("physics", compute_gflops=472)  # Jetson Nano

# 方式2：通过 FLConfig
config = FLConfig(
    algorithm="fedprox",
    num_rounds=50,
    time_model="physics",
    time_model_kwargs={"compute_gflops": 10, "uplink_mbps": 1},
)

# 方式3：自定义实现
class MyTimeModel(TimeModel):
    def compute_train_slots(self, client_id, num_samples, num_epochs, **kwargs):
        return num_epochs * 2  # 每个 epoch 固定 2 slots
    # ... 实现其他方法
```

### 验证结果（5轮 FedProx + MNIST + 5卫星 + 5地面站）

| 模式 | 虚拟时间 | 每轮分解 |
|------|---------|---------|
| slot(train=0) | 22 min | 等待返回 瞬时 |
| slot(train=3) | 37 min | 训练:3 |
| slot(全成本) | 46 min | 下载:1 训练:3 上传:2 |
| physics(10GFLOPS) | 1h13min | 下载:1 训练:1 上传:1 |

输出示例：
```
轮次   1/5 | TS=   0→   7 (7min) | 在线:2 | 选中: 2 | 
  耗时: 下载:1 | 训练:3 | 上传:2 | 准确率:0.9072
```

### 改动清单

| 文件 | 改动 |
|------|------|
| `fl_space/fl/time_model.py` | **新增** — TimeModel ABC + 2 个实现 + TimeBreakdown |
| `fl_space/fl/core.py` | FLRoundResult 新增 `timeslot_start`, `time_breakdown` |
| `fl_space/fl/server.py` | FLConfig 新增 `time_model`, `time_model_kwargs`；run_sync 集成 TimeModel |
| `fl_space/fl/runner.py` | 摘要显示时间模型信息和可读时间 |
| `fl_space/fl/cli.py` | `train` 子命令新增 `--time-model`, `--time-model-args` 参数 |
| `fl_space/fl/__init__.py` | 导出 TimeModel, TimeBreakdown, SlotTimeModel, PhysicsTimeModel |

### 扩展性

用户只需创建 Python 文件，继承 `TimeModel` 并实现 3 个抽象方法：

```python
# my_time_model.py
from fl_space.fl.time_model import TimeModel

class CustomTimeModel(TimeModel):
    @property
    def name(self): return "custom"
    
    def compute_train_slots(self, client_id, num_samples, num_epochs, **kwargs):
        # 你的训练时间计算逻辑
        return ...
    
    def compute_download_slots(self, model_size_bytes, **kwargs):
        # 你的下载时间计算逻辑
        return ...
    
    def compute_upload_slots(self, model_size_bytes, **kwargs):
        # 你的上传时间计算逻辑
        return ...
    
    def get_config_dict(self):
        return {"type": "custom"}
```

然后通过 `--time-model path/to/my_time_model.py:CustomTimeModel` 使用。

---

## 5. 项目当前状态

- **Step 1**: ✅ 环境模拟（双后端 Kepler/Skyfield）
- **Step 2**: ✅ 卫星配置系统（多星簇、注册表、可视化）
- **Step 3**: ✅ FL 算法模块（FedAvg/FedProx/FedBuff，四组件架构）
- **虚拟时间模型**: ✅ 可插拔时间投影（SlotTimeModel + PhysicsTimeModel）
- **Step 4**: 🔜 数据集增强
- **Step 5**: 🔜 地面基站设置
- **Step 6**: 🔜 保存节点（checkpoint）
- **Step 7**: 🔜 信息展示增强
