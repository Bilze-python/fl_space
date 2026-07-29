"""
联邦学习层 — 可插拔的 FL 算法框架

将 FL 算法分解为四个可替换组件：
    1. ClientSelector  — 客户端选择
    2. LocalTrainer    — 本地训练（epoch次数 + 训练逻辑）
    3. Aggregator      — 聚合（何时聚合 + 如何聚合）
    4. Evaluator       — 模型评估

已实现算法：
    - FedAvg  : 联邦平均（同步 + 加权平均）
    - FedProx : 联邦近端优化（处理数据异构）
    - FedBuff : 异步缓冲聚合（适应通信不可靠场景）

设计原则：
    - 组件可独立替换：用户只需实现一个或多个组件接口
    - 低耦合：算法逻辑与通信调度分离
    - 可扩展：通过注册机制添加自定义模型

快速开始::

    from fl_space.fl.runner import FLRunner

    runner = FLRunner.from_preset("fedavg", "small", "mnist")
    history = runner.run()
"""

from fl_space.fl.config import (
    DATASET_PRESETS as DATASET_PRESETS,
)
from fl_space.fl.config import (
    EXPERIMENT_PRESETS as EXPERIMENT_PRESETS,
)
from fl_space.fl.config import (
    SCALE_PRESETS as SCALE_PRESETS,
)
from fl_space.fl.config import (
    FLConfig as FLConfig,
)
from fl_space.fl.config import (
    get_preset_config as get_preset_config,
)
from fl_space.fl.config import (
    list_presets as list_presets,
)
from fl_space.fl.core import (
    Aggregator as Aggregator,
)
from fl_space.fl.core import (
    ClientSelector as ClientSelector,
)
from fl_space.fl.core import (
    ClientState as ClientState,
)
from fl_space.fl.core import (
    ClientUpdate as ClientUpdate,
)
from fl_space.fl.core import (
    Evaluator as Evaluator,
)
from fl_space.fl.core import (
    FLRoundResult as FLRoundResult,
)
from fl_space.fl.core import (
    LocalTrainer as LocalTrainer,
)
from fl_space.fl.fedavg import (
    CappedSelector as CappedSelector,
)
from fl_space.fl.fedavg import (
    LoadBalancedCappedSelector as LoadBalancedCappedSelector,
)
from fl_space.fl.fedavg import (
    SmoothCappedSelector as SmoothCappedSelector,
)
from fl_space.fl.models import (
    get_model as get_model,
)
from fl_space.fl.models import (
    list_models as list_models,
)
from fl_space.fl.models import (
    register_model as register_model,
)
from fl_space.fl.runner import (
    FLRunner as FLRunner,
)
from fl_space.fl.scheduler import (
    CommunicationScheduler as CommunicationScheduler,
)
from fl_space.fl.server import (
    FLServer as FLServer,
)
from fl_space.fl.time_model import (
    PhysicsTimeModel as PhysicsTimeModel,
)
from fl_space.fl.time_model import (
    SlotTimeModel as SlotTimeModel,
)
from fl_space.fl.time_model import (
    TimeBreakdown as TimeBreakdown,
)
from fl_space.fl.time_model import (
    TimeModel as TimeModel,
)
from fl_space.fl.timing_cache import (
    TimingCachePool as TimingCachePool,
)
from fl_space.fl.timing_cache import (
    TimingMetrics as TimingMetrics,
)

__all__ = [
    # 配置
    "DATASET_PRESETS",
    "EXPERIMENT_PRESETS",
    # 定时缓存优化
    "SCALE_PRESETS",
    # 核心抽象
    "Aggregator",
    "CappedSelector",
    "ClientSelector",
    "ClientState",
    "ClientUpdate",
    # 调度器
    "CommunicationScheduler",
    "Evaluator",
    # 编排器
    "FLConfig",
    "FLRoundResult",
    "FLRunner",
    "FLServer",
    "LoadBalancedCappedSelector",
    "LocalTrainer",
    # 调度器
    "PhysicsTimeModel",
    # 时间模型
    "SlotTimeModel",
    "SmoothCappedSelector",
    "TimeBreakdown",
    "TimeModel",
    "TimingCachePool",
    "TimingMetrics",
    "get_model",
    "get_preset_config",
    "list_models",
    "list_presets",
    "register_model",
]
