"""
FL Server — 联邦学习服务器编排器

职责：
    - 组合四个可插拔组件 (selector, trainer, aggregator, evaluator)
    - 编排完整的 FL 训练流程
    - 管理全局模型和客户端状态
    - 记录和返回训练历史

设计原则：
    - 与具体算法解耦：接受任意 ClientSelector/LocalTrainer/Aggregator/Evaluator 组合
    - 与通信方式解耦：通过 scheduler 获取通信状态
    - 单线程模拟异步场景：FedBuff 可通过 _train_client() 独立模拟单客户端训练
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from fl_space.fl.core import (
    Aggregator,
    ClientSelector,
    ClientState,
    ClientUpdate,
    Evaluator,
    FLRoundResult,
    LocalTrainer,
)
from fl_space.fl.scheduler import CommunicationScheduler
from fl_space.fl.time_model import TimeBreakdown, TimeModel

# PyTorch 可选依赖
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class FLConfig:
    """
    FL 实验完整配置。

    Attributes
    ----------
    algorithm : str
        算法名称: "fedavg", "fedprox", "fedbuff"。
    num_rounds : int
        全局训练轮次数。
    num_clients : int
        客户端总数。
    timeslots_per_round : int
        每轮的时间槽数（用于将连续时间离散化为 FL 轮次）。
    fraction : float
        每轮参与的客户端比例（同步算法）。
    local_epochs : int
        本地训练 epoch 数。
    batch_size : int
        训练 batch size。
    learning_rate : float
        学习率。
    mu : float
        FedProx 近端项系数（仅 FedProx 有效）。
    buffer_size : int
        FedBuff 缓冲区大小 K（仅 FedBuff 有效）。
    staleness_weight : bool
        FedBuff 是否启用陈旧度降权。
    device : str
        计算设备。
    seed : int | None
        随机种子。
    """

    algorithm: str = "fedavg"
    num_rounds: int = 50
    num_clients: int = 10
    timeslots_per_round: int = 10
    fraction: float = 0.5
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    mu: float = 0.01
    buffer_size: int = 5
    staleness_weight: bool = False
    device: str = "cpu"
    seed: int | None = None
    # 时间模型配置
    time_model: str = "slot"
    time_model_kwargs: dict = field(default_factory=dict)
    # 性能优化
    num_workers: int = 0  # DataLoader 并行进程数
    num_train_workers: int = 1  # 客户端并行训练线程数（1=串行）
    # 早停
    early_stop_acc: float | None = None  # 准确率阈值，达到后自动停止（如 0.9）
    # ISL 星间链路（可插拔）
    isl_enabled: bool = False  # 是否启用 ISL
    isl_calculator: str = "wgs84"  # ISL 计算器: wgs84 | disabled | path/to/custom.py:Cls
    isl_atmosphere_buffer_km: float = 0.0  # WGS84 大气余量 (km)
    isl_step_seconds: float = 60.0  # ISL 采样步长 (秒)
    # 数据划分（模拟太空 FL 小样本 non-IID）
    classes_per_client: int = 2  # 每个客户端限定的类别数（滑动窗口分配），0 表示使用 Dirichlet
    max_samples_per_client: int = 1000  # 每个客户端样本数上限，0 表示不限制
    partition_strategy: str = "probability"  # iid | dirichlet | shard | probability
    class_probability: float = 0.8  # probability strategy preference probability
    preference_mode: str = "class_balanced"  # client_window | class_balanced
    preferred_clients_per_class: int = 1
    sample_cap_strategy: str = "preserve"  # preserve | balanced
    data_dir: str = "./data"
    limit_to_sim_window: bool = True
    # ── 定时优化（论文七大方向）──
    timing_cache_enabled: bool = False  # 启用定时缓存池优化
    dynamic_epochs: bool = False  # 启用动态 Epoch（按轨道间隔自适应）
    dynamic_mu: bool = False  # 启用动态 μ（FedProx 按 Twait 自适应）
    dynamic_staleness: bool = False  # 启用动态陈旧阈值（FedBuff 按轨道间隔自适应）
    tiered_staleness: bool = False  # 启用分层陈旧丢弃（轻度/中度/重度）
    adaptive_buffer: bool = False  # 启用自适应缓冲区大小（按在线卫星数）
    multi_dim_scoring: bool = False  # 启用多维度加权打分调度
    isl_relay: bool = False  # 启用 ISL 中继传输
    compute_time_per_epoch_min: float = 2.0  # 每 epoch 计算耗时（分钟）
    warning_threshold_min: float = 5.0  # 窗口预警阈值（分钟）
    base_interval_min: float = 30.0  # LEO 基准间隔（分钟）
    # 特殊场景轻量化
    small_constellation: bool = False  # 小星座模式（关闭 ISL 时序）
    early_convergence_stop: bool = False  # 提前收敛停止定时刷新
    # ── 组网优化（Round 12）──
    gs_cap: int = 10  # GS 上限锁定在 8~10，多余 GS 改为异地接力接收/备份冗余
    load_balance_weight: float = 0.15  # 负载均衡惩罚权重（0=禁用）
    temporal_smooth_weight: float = 0.10  # 时序稳定性正则权重（0=禁用）
    staged_training: bool = True  # 分段训练：前期快速拟合 + 后期微调
    stage1_lr: float = 0.05  # 第一阶段学习率（快速拟合）
    stage2_lr: float = 0.005  # 第二阶段学习率（微调固化）
    stage1_rounds_ratio: float = 0.5  # 第一阶段占比
    gs_sleep_enabled: bool = True  # GS≥10 时启用休眠调度（闲置站点关停）
    window_pre_merge: bool = True  # 窗口预合并预处理（降低优化复杂度）
    fragment_mining: bool = True  # 碎片窗口挖掘（多天线分时复用）
    temporal_smooth_filter: bool = True  # 轨道接轨窗口时序平滑滤波
    gs_config_zone: str = "auto"  # 组网配置区域: auto|efficient|stable|avoid_critical
    # ── 组网配置推荐表 (硬编码, 不可在 JSON 序列化中修改) ──
    # 高效区间: GS=3~5 + SAT≥18
    # 稳定区间: GS≥10 + SAT 任意
    # 劣势区间: GS=6~8 + SAT=12~18 (规避)

    @classmethod
    def from_dict(cls, config: dict) -> FLConfig:
        """
        从字典创建 FLConfig。

        自动过滤不存在的字段，仅使用有效键。

        Parameters
        ----------
        config : dict
            配置字典，键对应 FLConfig 字段名。

        Returns
        -------
        FLConfig
            配置实例。

        使用示例::

            config = FLConfig.from_dict({
                "algorithm": "fedprox",
                "num_rounds": 100,
                "mu": 0.1,
                "device": "cuda",
            })
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in config.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, filepath: str) -> FLConfig:
        """
        从 JSON 文件加载 FLConfig。

        Parameters
        ----------
        filepath : str
            JSON 配置文件路径。

        Returns
        -------
        FLConfig
            配置实例。

        使用示例::

            config = FLConfig.from_json("my_experiment.json")
        """
        import json

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """
        导出为字典（便于 JSON 序列化和保存）。

        Returns
        -------
        dict
            当前配置的字典表示。
        """
        from dataclasses import asdict

        return asdict(self)

    def to_json(self, filepath: str) -> None:
        """
        保存配置为 JSON 文件。

        Parameters
        ----------
        filepath : str
            输出 JSON 文件路径。
        """
        import json

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class FLServer:
    """
    FL 训练服务器。

    组合四个可插拔组件，编排完整训练流程。

    支持两种运行模式：
        - run_sync()：同步训练（FedAvg / FedProx）
        - run_async()：异步训练（FedBuff）

    Parameters
    ----------
    config : FLConfig
        FL 实验配置。
    selector : ClientSelector
        客户端选择策略。
    trainer : LocalTrainer
        本地训练策略。
    aggregator : Aggregator
        聚合策略。
    evaluator : Evaluator
        评估策略。
    scheduler : CommunicationScheduler | None
        通信调度器（可选，未提供时假定所有客户端始终可通信）。

    使用示例::

        from fl_space.fl.server import FLServer, FLConfig
        from fl_space.fl.fedavg import create_fedavg_components

        config = FLConfig(algorithm="fedavg", num_rounds=50)
        components = create_fedavg_components()
        server = FLServer(config, *components)

        history = server.run_sync(model, train_loaders, test_loader)
        print(f"最终准确率: {history[-1].eval_metrics['accuracy']}")
    """

    def __init__(
        self,
        config: FLConfig,
        selector: ClientSelector,
        trainer: LocalTrainer,
        aggregator: Aggregator,
        evaluator: Evaluator,
        scheduler: CommunicationScheduler | None = None,
        time_model: TimeModel | None = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("FLServer 需要 PyTorch，请运行: pip install torch")

        self.config = config
        self.selector = selector
        self.trainer = trainer
        self.aggregator = aggregator
        self.evaluator = evaluator
        self.scheduler = scheduler

        # 时间模型：显式传入 > 配置构建
        if time_model is not None:
            self.time_model = time_model
        else:
            self.time_model = self._build_time_model()

        # ── 定时缓存池（论文七大方向优化）──
        self._timing_cache: Any = None
        if config.timing_cache_enabled and scheduler is not None:
            from fl_space.fl.timing_cache import TimingCachePool

            sim = scheduler._sim
            self._timing_cache = TimingCachePool(
                scheduler=scheduler,
                timeslot_duration_min=getattr(sim, "timeslot_duration_min", 1.0),
                compute_time_per_epoch_min=getattr(config, "compute_time_per_epoch_min", 2.0),
            )
            # 绑定到 trainer（支持动态 Epoch / 动态 μ）
            if hasattr(self.trainer, "_timing_cache"):
                self.trainer._timing_cache = self._timing_cache
                self.trainer.max_epochs = config.local_epochs * 2
                self.trainer.warning_threshold_min = config.warning_threshold_min

        # 运行时状态
        self._global_model: Any = None
        self._clients: list[ClientState] = []
        self._history: list[FLRoundResult] = []
        self._sim_time_limit: int | None = None

        # ── ISL 中继缓存 ──
        self._isl_relay_map: dict[int, list[tuple[int, int, int]]] = {}
        if config.isl_relay and scheduler is not None:
            self._build_isl_relay_map()

    def _build_time_model(self) -> TimeModel:
        """根据 FLConfig 构建时间模型实例。"""
        kwargs = dict(self.config.time_model_kwargs)

        # 自动继承 timeslot 时长
        sim = self.scheduler._sim if self.scheduler is not None else None
        if sim is not None and "timeslot_duration_min" not in kwargs:
            kwargs["timeslot_duration_min"] = getattr(sim, "timeslot_duration_min", 1.0)

        return TimeModel.create(self.config.time_model, **kwargs)

    def _build_isl_relay_map(self) -> None:
        """预计算 ISL 中继映射表。

        为每颗卫星预计算其簇内最近地面站的中继卫星，
        无直连 GS 的卫星可通过中继上传模型。
        """
        sim = self.scheduler._sim
        if not sim.isl_config.enabled:
            return

        n_sats = sim.num_satellites
        n_slots = sim.num_timeslots

        from fl_space.simulator.orbit_cache import OrbitCacheCalculator

        cache_calc = OrbitCacheCalculator(
            orbit_period_min=sim.orbit_period_min,
            timeslot_duration_min=sim.timeslot_duration_min,
            num_satellites=n_sats,
            num_ground_stations=sim.num_ground_stations,
            num_timeslots=n_slots,
        )
        self._isl_relay_map = cache_calc.precompute_isl_relay_map(sim)

    # ── 定时优化辅助方法 ────────────────────────────────────
    def _get_dynamic_epochs(self, sat_id: int, timeslot: int) -> int:
        """获取卫星的动态 Epoch 数。

        结合定时缓存池和轨道间隔计算最优本地训练轮数。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        timeslot : int
            当前时隙。

        Returns
        -------
        int
            动态 epoch 数。
        """
        if not self.config.dynamic_epochs or self._timing_cache is None:
            return self.config.local_epochs

        return self._timing_cache.get_dynamic_epochs(
            sat_id, timeslot,
            max_epochs=self.config.local_epochs * 2,
            warning_threshold_min=self.config.warning_threshold_min,
        )

    def _get_dynamic_mu(self, sat_id: int, timeslot: int) -> float:
        """获取卫星的动态 μ 值。

        基于 T_wait 自适应调整 FedProx 近端约束。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        timeslot : int
            当前时隙。

        Returns
        -------
        float
            动态 μ 值。
        """
        if not self.config.dynamic_mu or self._timing_cache is None:
            return self.config.mu

        return self._timing_cache.get_dynamic_mu(
            sat_id, timeslot,
            base_mu=self.config.mu,
            base_interval_min=self.config.base_interval_min,
        )

    def _check_window_warning(self, sat_id: int, timeslot: int) -> bool:
        """检查窗口预警：距下一次GS接轨剩余时间是否不足。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        timeslot : int
            当前时隙。

        Returns
        -------
        bool
            True 表示需要立即停止训练。
        """
        if self._timing_cache is None:
            return False
        return self._timing_cache.check_window_warning(
            sat_id, timeslot, self.config.warning_threshold_min
        )

    def _get_isl_relay(self, sat_id: int, timeslot: int) -> tuple[bool, int]:
        """获取卫星的 ISL 中继信息。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        timeslot : int
            当前时隙。

        Returns
        -------
        tuple[bool, int]
            (is_available, relay_sat_id)。
        """
        if not self.config.isl_relay or self._timing_cache is None:
            return (False, -1)
        return self._timing_cache.get_isl_relay_info(sat_id, timeslot)

    @property
    def history(self) -> list[FLRoundResult]:
        """训练历史记录。"""
        return self._history

    @staticmethod
    def _get_client_data_sizes(
        train_loaders: dict[int, Any],
    ) -> dict[int, int]:
        """
        从 DataLoader 提取每个客户端的数据量。

        Parameters
        ----------
        train_loaders : dict[int, DataLoader]
            客户端 ID → DataLoader 映射。

        Returns
        -------
        dict[int, int]
            客户端 ID → 样本数映射。
        """
        sizes: dict[int, int] = {}
        for cid, loader in train_loaders.items():
            try:
                ds = loader.dataset
                if hasattr(ds, "__len__"):
                    sizes[cid] = len(ds)
                else:
                    sizes[cid] = 100
            except Exception:
                sizes[cid] = 100
        return sizes

    def _init_clients(self) -> None:
        """初始化客户端状态列表。"""
        self._clients = [
            ClientState(client_id=i, data_size=100) for i in range(self.config.num_clients)
        ]

    def _update_connectivity(self, timeslot: int) -> None:
        """
        根据通信调度器更新客户端连接状态。

        Parameters
        ----------
        timeslot : int
            当前时间槽。
        """
        if self.scheduler is None:
            # 无调度器：假设始终可通信
            for c in self._clients:
                c.is_connected = True
            return

        if self._sim_time_limit is not None and timeslot >= self._sim_time_limit:
            for c in self._clients:
                c.is_connected = False
            return

        connected = set(self.scheduler.get_connected_sats(timeslot))
        for c in self._clients:
            c.is_connected = c.client_id in connected

    def _train_client(
        self,
        client_id: int,
        train_loaders: dict[int, Any],
        round_num: int,
        global_weights: list[Any] | None = None,
    ) -> ClientUpdate | None:
        """
        训练单个客户端。

        Parameters
        ----------
        client_id : int
            客户端 ID。
        train_loaders : dict[int, DataLoader]
            客户端 ID → DataLoader 映射。
        round_num : int
            当前全局轮次。
        global_weights : list | None
            预克隆的全局权重，避免每客户端重复克隆。

        Returns
        -------
        ClientUpdate | None
            训练结果，失败时返回 None。
        """
        if client_id not in train_loaders:
            return None

        try:
            if global_weights is None:
                global_weights = [param.data.clone() for param in self._global_model.parameters()]
            return self.trainer.train(
                client_id=client_id,
                model=self._global_model,
                train_loader=train_loaders[client_id],
                global_weights=global_weights,
                round_num=round_num,
            )
        except Exception as e:
            print(f"  [警告] 客户端 {client_id} 训练失败: {e}")
            return None

    def _train_clients_parallel(
        self,
        selected_ids: list[int],
        train_loaders: dict[int, Any],
        round_num: int,
    ) -> list[ClientUpdate]:
        """
        并行训练多个客户端。

        使用线程池并行执行多个客户端的本地训练。
        trainer.train() 内部会 deepcopy 模型，因此多个线程
        同时读取 self._global_model 是安全的。

        性能收益：
            - GPU 训练：CUDA 操作释放 GIL，多线程可重叠 GPU 计算
            - CPU 训练：重叠数据加载和计算

        Parameters
        ----------
        selected_ids : list[int]
            选中的客户端 ID 列表。
        train_loaders : dict[int, DataLoader]
            客户端 ID → DataLoader 映射。
        round_num : int
            当前全局轮次。

        Returns
        -------
        list[ClientUpdate]
            成功训练的客户端更新列表。
        """
        import concurrent.futures

        n_workers = getattr(self.config, "num_train_workers", 1) or 1

        if n_workers <= 1 or len(selected_ids) <= 1:
            # 串行模式：预克隆一次全局权重，所有客户端复用
            global_weights = [param.data.clone() for param in self._global_model.parameters()]
            updates = []
            for cid in selected_ids:
                print(".", end="", flush=True)
                update = self._train_client(
                    cid,
                    train_loaders,
                    round_num,
                    global_weights=global_weights,
                )
                if update is not None:
                    updates.append(update)
            return updates

        # 并行模式：预克隆全局权重（只读，线程安全）
        global_weights = [param.data.clone() for param in self._global_model.parameters()]

        def _train_single(cid: int) -> ClientUpdate | None:
            if cid not in train_loaders:
                return None
            try:
                update = self.trainer.train(
                    client_id=cid,
                    model=self._global_model,
                    train_loader=train_loaders[cid],
                    global_weights=global_weights,
                    round_num=round_num,
                )
                return update
            except Exception as e:
                print(f"  [警告] 客户端 {cid} 训练失败: {e}")
                return None

        max_workers = min(n_workers, len(selected_ids))
        updates: list[ClientUpdate] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
        ) as executor:
            future_map = {executor.submit(_train_single, cid): cid for cid in selected_ids}
            for future in concurrent.futures.as_completed(future_map):
                result = future.result()
                if result is not None:
                    updates.append(result)

        return updates

    # ── 同步训练模式 (FedAvg / FedProx) ──────────────────────

    def run_sync(
        self,
        model: Any,
        train_loaders: dict[int, Any],
        test_loader: Any,
        verbose: bool = True,
    ) -> list[FLRoundResult]:
        """
        运行同步 FL 训练（适用于 FedAvg / FedProx）。

        通信驱动轮次推进：
            1. 找到当前时刻可通信的卫星 → 分发全局模型
            2. 卫星本地训练（不占虚拟时间）
            3. 等待各卫星下一次接触窗口 → 上传模型
            4. 聚合 → 新全局模型诞生 → 本轮完成
            5. 时间跳到聚合后下一次有卫星连接的时隙 → 下一轮

        Parameters
        ----------
        model : nn.Module
            初始全局模型。
        train_loaders : dict[int, DataLoader]
            客户端 ID → 本地训练数据 Loader。
        test_loader : DataLoader
            测试数据 Loader。
        verbose : bool
            是否打印进度。

        Returns
        -------
        list[FLRoundResult]
            每轮训练结果。
        """
        self._global_model = copy.deepcopy(model)
        self._init_clients()
        self._history = []

        # 获取模拟器引用（用于通信窗口查询）
        sim = self.scheduler._sim if self.scheduler is not None else None
        self._sim_time_limit = None
        if sim is not None and getattr(self.config, "limit_to_sim_window", True):
            self._sim_time_limit = int(getattr(sim, "num_timeslots_pre", sim.num_timeslots))

        # 计算模型大小（字节），供时间模型使用
        model_size_bytes = sum(
            p.numel() * p.element_size() for p in self._global_model.parameters()
        )

        # 预计算客户端数据量
        client_data_sizes = self._get_client_data_sizes(train_loaders)

        # ── 基线评估（训练前，随机初始化模型）──
        if verbose:
            print("  基线评估（随机模型）...", end="", flush=True)
        baseline_metrics = self.evaluator.evaluate(
            self._global_model, test_loader, -1,
        )
        self._history.append(FLRoundResult(
            round_num=-1,
            eval_metrics=baseline_metrics,
            num_clients=0,
            train_loss=baseline_metrics.get("loss", 0.0),
            timeslot_start=0,
            time_breakdown=None,
        ))
        if verbose:
            acc = baseline_metrics.get("accuracy", 0)
            print(f" 准确率 {acc:.2%}（随机基线 ≈ 10% for 10类）")

        current_ts = 0  # 当前虚拟时间槽
        completed_rounds = 0  # 已完成轮次计数

        while completed_rounds < self.config.num_rounds:
            # ── 时间分解记录 ──
            breakdown = TimeBreakdown()

            # ── 1. 找到下一个有卫星连接的时隙（分发窗口）──
            if self._sim_time_limit is not None and current_ts >= self._sim_time_limit:
                if verbose:
                    print("  Reached simulation time limit; stopping training.")
                break
            start_ts = self._advance_to_next_contact(current_ts, self._sim_time_limit)
            if start_ts is None:
                if verbose:
                    print(f"  轮次 {completed_rounds + 1}: 无更多通信窗口，训练终止")
                break

            breakdown.wait_distribution = start_ts - current_ts
            current_ts = start_ts

            # ── 1b. 模型下载耗时 ──
            download_slots = self.time_model.compute_download_slots(model_size_bytes)
            breakdown.download = download_slots
            current_ts += download_slots
            if self._sim_time_limit is not None and current_ts >= self._sim_time_limit:
                if verbose:
                    print("  Download would exceed simulation time limit; stopping training.")
                break

            # ── 2. 更新通信状态，选择客户端 ──
            self._update_connectivity(current_ts)
            selected_ids = self.selector.select(self._clients, completed_rounds)
            if len(selected_ids) < 1:
                # 无可选客户端，前进1个时隙重试
                current_ts += 1
                continue

            # ── 3. 本地训练（并行 + 时间模型决定耗时 + 动态Epoch）──
            if verbose:
                print(
                    f"  轮次 {completed_rounds + 1:3d}/{self.config.num_rounds} | "
                    f"训练 {len(selected_ids)} 客户端...",
                    end="",
                    flush=True,
                )
            train_start_ts = current_ts

            # ── 多维度打分重排序（按优先级训练）──
            if self.config.multi_dim_scoring and self._timing_cache is not None:
                staleness_map = {
                    cid: completed_rounds - self._clients[cid].last_update_round
                    for cid in selected_ids
                }
                scored = self._timing_cache.get_client_scores(
                    selected_ids, staleness_map, current_ts
                )
                selected_ids = [cid for cid, _ in scored]

            updates = self._train_clients_parallel(
                selected_ids,
                train_loaders,
                completed_rounds,
            )
            max_train_slots = 0
            for update in updates:
                cid = update.client_id
                n_samples = client_data_sizes.get(cid, 100)

                # ── 动态 Epoch：按轨道间隔自适应 ──
                if self.config.dynamic_epochs and self._timing_cache is not None:
                    dyn_epochs = self._timing_cache.get_dynamic_epochs(
                        cid, current_ts,
                        max_epochs=self.config.local_epochs * 2,
                        warning_threshold_min=self.config.warning_threshold_min,
                    )
                else:
                    dyn_epochs = self.config.local_epochs

                train_slots = self.time_model.compute_train_slots(
                    cid, n_samples, dyn_epochs,
                )
                breakdown.per_satellite[cid] = {
                    "train": train_slots,
                    "epochs": dyn_epochs,
                }
                max_train_slots = max(max_train_slots, train_slots)

            if not updates:
                current_ts += 1
                continue

            breakdown.train = max_train_slots
            current_ts = train_start_ts + max_train_slots

            # ── 4. 等待各卫星返回（下一次接触窗口 + 上传时间）──
            if sim is not None:
                return_times = []
                returned_client_ids: set[int] = set()
                return_start_ts = current_ts
                for cid in selected_ids:
                    next_contact = self._get_next_contact_for_client(
                        cid,
                        current_ts,
                        self._sim_time_limit,
                    )
                    if next_contact is not None:
                        contact_ts = next_contact[0]
                        # 上传时间叠加在接触窗口之后
                        upload_slots = self.time_model.compute_upload_slots(
                            model_size_bytes,
                        )
                        breakdown.per_satellite.setdefault(cid, {})
                        breakdown.per_satellite[cid]["upload"] = upload_slots
                        breakdown.per_satellite[cid]["wait_return"] = contact_ts - current_ts
                        arrival_ts = contact_ts + upload_slots
                        if self._sim_time_limit is not None and arrival_ts >= self._sim_time_limit:
                            continue
                        returned_client_ids.add(cid)
                        return_times.append(arrival_ts)
                if return_times:
                    current_ts = max(return_times)
                    breakdown.wait_return = current_ts - return_start_ts
                # 如果无返回窗口，模型更新丢失，但仍聚合已返回的
            else:
                # 无模拟器：固定步进
                current_ts += self.config.timeslots_per_round
                breakdown.wait_return = self.config.timeslots_per_round

            # 上传时间从 per_satellite 汇总
            breakdown.upload = max(
                (v.get("upload", 0) for v in breakdown.per_satellite.values()),
                default=0,
            )
            if sim is not None:
                updates = [u for u in updates if u.client_id in returned_client_ids]
                if not updates:
                    current_ts += 1
                    continue
            # 去掉上传时间重复计算（wait_return 已包含 upload 叠加）
            # 这里 wait_return 记录的是从 train_end 到 return_ts 的总等待时间
            # upload 单独记录以便分解展示

            # ── 5. 聚合 → 新全局模型诞生 ──
            global_weights = [param.data.clone() for param in self._global_model.parameters()]
            if self.aggregator.should_aggregate(updates, completed_rounds):
                new_weights = self.aggregator.aggregate(
                    global_weights,
                    updates,
                    completed_rounds,
                )
                with torch.no_grad():
                    for param, new_w in zip(self._global_model.parameters(), new_weights):
                        param.data.copy_(new_w)

            # 标记客户端参与
            for cid in selected_ids:
                client = self._clients[cid]
                client.last_update_round = completed_rounds

            # ── 6. 评估 ──
            eval_metrics = self.evaluator.evaluate(
                self._global_model,
                test_loader,
                completed_rounds,
            )

            # ── 6b. 自适应 μ 反馈 (FedProxSat) ──
            current_acc = eval_metrics.get("accuracy", 0)
            if hasattr(self.trainer, "update_accuracy"):
                _ = self.trainer.update_accuracy(current_acc)

            # 计算时间分解总计
            breakdown.total = current_ts - start_ts

            avg_loss = sum(u.train_loss for u in updates) / len(updates)
            result = FLRoundResult(
                round_num=completed_rounds,
                num_clients=len(updates),
                train_loss=round(avg_loss, 6),
                eval_metrics=eval_metrics,
                timeslot=current_ts,
                timeslot_start=start_ts,
                time_breakdown=breakdown.to_dict(),
            )
            self._history.append(result)

            completed_rounds += 1

            # ── 早停检查（含提前收敛定时中断）──
            early_stop_acc = getattr(self.config, "early_stop_acc", None)
            early_convergence = self.config.early_convergence_stop
            if early_stop_acc is not None and current_acc >= early_stop_acc:
                if verbose:
                    print(
                        f"\n  >>> 早停触发: 准确率 {current_acc:.4f} >= {early_stop_acc} "
                        f"(第 {completed_rounds} 轮)"
                    )
                break

            # ── 提前收敛定时中断：准确率达到阈值后停止轨道完整定时刷新 ──
            if early_convergence and current_acc >= 0.9:
                if self._timing_cache is not None:
                    self._timing_cache._refresh_layer1(current_ts)
                if verbose:
                    print(f"\n  >>> 提前收敛: acc={current_acc:.4f}, 停止完整定时刷新")

            # 准备下一轮：从聚合时刻之后开始
            current_ts += 1

            if verbose:
                acc = eval_metrics.get("accuracy", 0)
                conn_at_start = (
                    len(self.scheduler.get_connected_sats(start_ts))
                    if self.scheduler
                    else len(selected_ids)
                )
                # 构建时间分解显示
                tm_display = self.time_model.slots_to_display(
                    breakdown.total,
                    getattr(self.time_model, "timeslot_duration_min", 1.0),
                )
                parts = []
                if breakdown.download > 0:
                    parts.append(f"下载:{breakdown.download}")
                if breakdown.train > 0:
                    parts.append(f"训练:{breakdown.train}")
                if breakdown.upload > 0:
                    parts.append(f"上传:{breakdown.upload}")
                time_info = " | ".join(parts) if parts else "瞬时"
                print(
                    f"  \r  轮次 {completed_rounds:3d}/{self.config.num_rounds} | "
                    f"TS={start_ts:4d}→{result.timeslot:4d} "
                    f"({tm_display}) | "
                    f"在线:{conn_at_start} | "
                    f"选中:{len(updates):2d} | "
                    f"{time_info} | "
                    f"准确率:{acc:.4f}"
                )

        # ── 虚拟时间汇总 ──
        if sim is not None and self._history:
            final_ts = self._history[-1].timeslot
            total_hours = final_ts * sim.timeslot_duration_min / 60.0
            if verbose:
                print(
                    f"\n  [模拟时间] 总虚拟时间: {final_ts} timeslots = "
                    f"{total_hours:.1f} 小时 ({total_hours / 24:.1f} 天), "
                    f"预计算: {sim.num_timeslots_pre:.0f} slots"
                )

        return self._history

    def _advance_to_next_contact(
        self,
        from_ts: int,
        max_timeslot: int | None = None,
    ) -> int | None:
        """Find the next connected timeslot (numpy vectorized for max_timeslot path).

        When max_timeslot is set, uses vectorized numpy argmax instead of
        O(N) Python for-loop over timeslots.  Typically called once per FL round
        and can skip hundreds of empty timeslots in sparse contact regimes.
        """
        if self.scheduler is None:
            if max_timeslot is not None and from_ts >= max_timeslot:
                return None
            return from_ts

        sim = self.scheduler._sim
        cm = sim.contact_matrix

        if max_timeslot is None:
            earliest = None
            for sat_id in range(sim.num_satellites):
                nc = sim.get_next_contact(sat_id, from_ts - 1)
                if nc is not None:
                    ts, _ = nc
                    if earliest is None or ts < earliest:
                        earliest = ts
            return earliest

        start_ts = max(from_ts, 0)
        stop_ts = min(max_timeslot, getattr(sim, "num_timeslots", max_timeslot))
        if start_ts >= stop_ts:
            return None

        # ── numpy 向量化：O(1) 定位下一个有接触的 timeslot ──
        import numpy as np

        col_has_contact = np.any(cm.simple_matrix[:, start_ts:stop_ts] >= 0, axis=0)
        if not np.any(col_has_contact):
            return None
        offset = int(np.argmax(col_has_contact))
        ts = start_ts + offset
        return ts if ts < stop_ts else None

    def _get_next_contact_for_client(
        self,
        sat_id: int,
        after_ts: int,
        max_timeslot: int | None = None,
    ) -> tuple[int, int] | None:
        """Find one client's next contact (numpy vectorized for bounded range)."""
        if self.scheduler is None:
            return None

        sim = self.scheduler._sim
        if max_timeslot is None:
            return sim.get_next_contact(sat_id, after_ts)

        start_ts = max(after_ts + 1, 0)
        stop_ts = min(max_timeslot, getattr(sim, "num_timeslots", max_timeslot))
        if start_ts >= stop_ts:
            return None
        # ── numpy 向量化：直接在接触矩阵行上 argmax ──
        import numpy as np

        row_slice = sim.contact_matrix.simple_matrix[sat_id, start_ts:stop_ts]
        in_contact = row_slice >= 0
        if not np.any(in_contact):
            return None
        offset = int(np.argmax(in_contact))
        return (start_ts + offset, int(row_slice[offset]))

    def run_async(
        self,
        model: Any,
        train_loaders: dict[int, Any],
        test_loader: Any,
        verbose: bool = True,
    ) -> list[FLRoundResult]:
        """
        运行异步 FL 训练（适用于 FedBuff）。

        流程：
            与同步不同，客户端独立训练并随时提交更新。
            服务端在每个 timeslot：
                1. 更新通信状态
                2. 让所有可通信的客户端分别训练
                3. 将更新放入缓冲区（分层陈旧过滤）
                4. 缓冲区满时触发聚合
                5. 周期性评估
                6. 动态调整缓冲区大小 / 陈旧阈值

        Parameters
        ----------
        model : nn.Module
            初始全局模型。
        train_loaders : dict[int, DataLoader]
            客户端 ID → 本地训练数据 Loader。
        test_loader : DataLoader
            测试数据 Loader。
        verbose : bool
            是否打印进度。

        Returns
        -------
        list[FLRoundResult]
            聚合事件记录。
        """
        from fl_space.fl.fedbuff import BufferAggregator

        self._global_model = copy.deepcopy(model)
        self._init_clients()
        self._history = []

        if not isinstance(self.aggregator, BufferAggregator):
            raise TypeError(
                f"异步模式需要 BufferAggregator，当前聚合器: {type(self.aggregator).__name__}"
            )

        buffer_agg: BufferAggregator = self.aggregator
        sim = self.scheduler._sim if self.scheduler is not None else None
        self._sim_time_limit = None
        if sim is not None and getattr(self.config, "limit_to_sim_window", True):
            self._sim_time_limit = int(getattr(sim, "num_timeslots_pre", sim.num_timeslots))

        total_timeslots = self.config.num_rounds * self.config.timeslots_per_round
        if self._sim_time_limit is not None:
            total_timeslots = min(total_timeslots, self._sim_time_limit)
        training_clients: set[int] = set()
        global_round = 0

        eval_interval = max(1, total_timeslots // 20)  # 约 20 次评估
        last_eval_metrics: dict[str, float] = {}

        # ── 增量更新定时缓存 ──
        if self._timing_cache is not None:
            self._timing_cache.incremental_update(0)

        for ts in range(total_timeslots):
            # ── 增量定时更新 ──
            if self._timing_cache is not None:
                self._timing_cache.incremental_update(ts)

            # 1. 更新通信状态
            self._update_connectivity(ts)

            # 2. 选择可参与训练的客户端
            available = self.selector.select(
                self._clients,
                global_round,
                already_training=training_clients,
            )

            # ── 动态缓冲区调整 ──
            if self.config.adaptive_buffer:
                n_connected = sum(1 for c in self._clients if c.is_connected)
                buffer_agg.update_buffer_size(n_connected)

            # 3. 每个可用客户端独立训练（含 ISL 中继）
            for cid in available:
                # ── ISL 中继：无直连 GS 的卫星通过中继上传 ──
                if self.config.isl_relay and self._timing_cache is not None:
                    has_relay, _relay_sat = self._timing_cache.get_isl_relay_info(cid, ts)
                    if not has_relay:
                        # 无直连也无中继，跳过
                        continue

                update = self._train_client(cid, train_loaders, global_round)
                if update is not None:
                    # add_update 内部处理分层陈旧过滤
                    accepted = buffer_agg.add_update(update)
                    if accepted:
                        self._clients[cid].last_update_round = global_round

            # 4. 检查缓冲区是否触发聚合
            if buffer_agg.should_aggregate([], global_round):
                global_weights = [param.data.clone() for param in self._global_model.parameters()]
                new_weights = buffer_agg.aggregate(
                    global_weights,
                    [],
                    global_round,
                )
                with torch.no_grad():
                    for param, new_w in zip(self._global_model.parameters(), new_weights):
                        param.data.copy_(new_w)

                global_round += 1

                # 5. 周期性评估
                if ts % eval_interval == 0:
                    last_eval_metrics = self.evaluator.evaluate(
                        self._global_model,
                        test_loader,
                        global_round,
                    )

                status = buffer_agg.buffer_status()
                result = FLRoundResult(
                    round_num=global_round,
                    num_clients=status.get("last_aggregate_count", status["current_count"]),
                    train_loss=0.0,  # 异步模式下难以精确计算
                    eval_metrics=dict(last_eval_metrics),
                    timeslot=ts,
                )
                self._history.append(result)

                if verbose:
                    acc = last_eval_metrics.get("accuracy", 0)
                    drop_rate = status.get("drop_rate", 0)
                    print(
                        f"  Timeslot {ts:4d} | 聚合 #{global_round:3d} | "
                        f"缓冲区: {status['current_count']}/{status['buffer_size']} | "
                        f"丢弃率: {drop_rate:.2%} | "
                        f"准确率: {acc:.4f}"
                    )

        # 最终评估
        final_metrics = self.evaluator.evaluate(
            self._global_model,
            test_loader,
            global_round,
        )
        self._history.append(FLRoundResult(
            round_num=global_round,
            num_clients=0,
            train_loss=0.0,
            eval_metrics=dict(final_metrics),
            timeslot=total_timeslots,
        ))
        if verbose:
            acc = final_metrics.get("accuracy", 0)
            print(f"\n  最终准确率: {acc:.4f}")
            print(f"  总聚合次数: {global_round}")

        return self._history

    # ── 通用接口 ──────────────────────────────────────────────

    def run(
        self,
        model: Any,
        train_loaders: dict[int, Any],
        test_loader: Any,
        verbose: bool = True,
    ) -> list[FLRoundResult]:
        """
        根据配置自动选择同步或异步模式运行。

        Parameters
        ----------
        model : nn.Module
            初始全局模型。
        train_loaders : dict[int, DataLoader]
            客户端 ID → 本地训练数据 Loader。
        test_loader : DataLoader
            测试数据 Loader。
        verbose : bool
            是否打印进度。

        Returns
        -------
        list[FLRoundResult]
            训练历史。

        Raises
        ------
        ValueError
            当 algorithm 未知时。
        """
        algo = self.config.algorithm.lower()
        if algo in ("fedavg", "fedprox"):
            return self.run_sync(model, train_loaders, test_loader, verbose)
        elif algo == "fedbuff":
            return self.run_async(model, train_loaders, test_loader, verbose)
        else:
            raise ValueError(f"未知算法: '{algo}'，支持: fedavg, fedprox, fedbuff")

    def get_global_model(self) -> Any:
        """返回当前全局模型。"""
        return self._global_model

    def get_history_dict(self) -> list[dict[str, Any]]:
        """
        将训练历史导出为字典列表（便于 JSON 序列化）。

        Returns
        -------
        list[dict]
            每轮结果字典。
        """
        results = []
        for r in self._history:
            entry: dict[str, Any] = {
                "round": r.round_num,
                "timeslot": r.timeslot,
                "timeslot_start": r.timeslot_start,
                "num_clients": r.num_clients,
                "train_loss": r.train_loss,
            }
            entry.update(r.eval_metrics)
            if r.time_breakdown:
                entry["time_breakdown"] = r.time_breakdown
            results.append(entry)
        return results
