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
    server_learning_rate: float = 1.0
    async_eval_every: int = 1
    protocol_mode: str = "standard"  # standard | paper_approx
    selection_strategy: str = "random"  # random | earliest_return
    contact_adaptive_epochs: bool = False
    max_contact_epochs: int = 50
    fedbuff_mu: float = 0.0
    max_staleness: int | None = None
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
    satellite_data_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    data_dir: str = "./data"
    limit_to_sim_window: bool = True

    def __post_init__(self) -> None:
        self.protocol_mode = str(self.protocol_mode).lower()
        self.selection_strategy = str(self.selection_strategy).lower()
        if self.protocol_mode not in {"standard", "paper_approx"}:
            raise ValueError("protocol_mode must be 'standard' or 'paper_approx'")
        if self.selection_strategy not in {"random", "earliest_return"}:
            raise ValueError("selection_strategy must be 'random' or 'earliest_return'")
        if self.protocol_mode == "paper_approx":
            if self.algorithm.lower() in {"fedavg", "fedprox"}:
                self.selection_strategy = "earliest_return"
            if self.algorithm.lower() == "fedprox":
                self.contact_adaptive_epochs = True
            if self.algorithm.lower() == "fedbuff":
                if self.fedbuff_mu <= 0:
                    self.fedbuff_mu = self.mu
                if self.max_staleness is None:
                    self.max_staleness = 4
        self.max_contact_epochs = max(1, int(self.max_contact_epochs))
        if self.max_staleness is not None:
            self.max_staleness = max(0, int(self.max_staleness))

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

        # 运行时状态
        self._global_model: Any = None
        self._clients: list[ClientState] = []
        self._history: list[FLRoundResult] = []
        self._events: list[dict[str, Any]] = []
        self._sim_time_limit: int | None = None

    def _build_time_model(self) -> TimeModel:
        """根据 FLConfig 构建时间模型实例。"""
        kwargs = dict(self.config.time_model_kwargs)

        # 自动继承 timeslot 时长
        sim = self.scheduler._sim if self.scheduler is not None else None
        if sim is not None and "timeslot_duration_min" not in kwargs:
            kwargs["timeslot_duration_min"] = getattr(sim, "timeslot_duration_min", 1.0)

        return TimeModel.create(self.config.time_model, **kwargs)

    @property
    def history(self) -> list[FLRoundResult]:
        """训练历史记录。"""
        return self._history

    @property
    def events(self) -> list[dict[str, Any]]:
        """Return a copy of the communication and aggregation event log."""
        return list(self._events)

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
        **trainer_kwargs: Any,
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
                **trainer_kwargs,
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
        """Run paper-faithful synchronous FedAvg/FedProx rounds.

        The selected client set is fixed for the whole round. With an orbital
        scheduler, every selected satellite receives the same global version at
        its next contact and the server waits until every selected update has
        returned before aggregating.
        """
        self._global_model = copy.deepcopy(model)
        self._init_clients()
        self._history = []
        self._events = []

        sim = self.scheduler._sim if self.scheduler is not None else None
        self._sim_time_limit = None
        if sim is not None and getattr(self.config, "limit_to_sim_window", True):
            self._sim_time_limit = int(getattr(sim, "num_timeslots_pre", sim.num_timeslots))

        model_size_bytes = sum(
            p.numel() * p.element_size() for p in self._global_model.parameters()
        )
        client_data_sizes = self._get_client_data_sizes(train_loaders)

        baseline_metrics = self.evaluator.evaluate(self._global_model, test_loader, -1)
        self._history.append(
            FLRoundResult(
                round_num=-1,
                eval_metrics=baseline_metrics,
                num_clients=0,
                train_loss=baseline_metrics.get("loss", 0.0),
                timeslot_start=0,
                time_breakdown=TimeBreakdown().to_dict(),
                extra={"phase": "baseline"},
            )
        )

        current_ts = 0
        for completed_rounds in range(self.config.num_rounds):
            round_start = current_ts

            # 尝试推进到有客户端在线的时隙
            max_advance = 200  # 安全阀：最多跳过 200 个时隙
            skip_count = 0
            self._update_connectivity(round_start)
            while skip_count < max_advance:
                if any(c.is_connected for c in self._clients):
                    break
                round_start += 1
                skip_count += 1
                self._update_connectivity(round_start)
            current_ts = round_start

            if skip_count >= max_advance:
                break  # 仿真窗口内无可接触客户端

            breakdown = TimeBreakdown()
            download_slots = self.time_model.compute_download_slots(model_size_bytes)
            selection_scores: dict[int, int] = {}
            if self.scheduler is not None:
                upload_slots = self.time_model.compute_upload_slots(model_size_bytes)
                for client in self._clients:
                    first_contact = self._get_next_contact_for_client(
                        client.client_id,
                        round_start - 1,
                        self._sim_time_limit,
                    )
                    if first_contact is None:
                        continue
                    candidate_samples = client_data_sizes.get(client.client_id, 100)
                    candidate_train_slots = self.time_model.compute_train_slots(
                        client.client_id,
                        candidate_samples,
                        self.config.local_epochs,
                    )
                    candidate_end = first_contact[0] + download_slots + candidate_train_slots
                    return_contact = self._get_next_contact_for_client(
                        client.client_id,
                        candidate_end - 1,
                        self._sim_time_limit,
                    )
                    if return_contact is not None:
                        selection_scores[client.client_id] = return_contact[0] + upload_slots

            selected_ids = self.selector.select(
                self._clients,
                completed_rounds,
                completion_times=selection_scores,
            )
            if not selected_ids:
                # 有连接但选择器返回空（如 CappedSelector.min_clients 不满足）
                current_ts += 1
                continue

            global_weights = [param.data.clone() for param in self._global_model.parameters()]
            updates: list[ClientUpdate] = []
            arrival_times: list[int] = []
            download_times: list[int] = []
            train_end_times: list[int] = []
            round_complete = True

            for cid in selected_ids:
                if self.scheduler is None:
                    download_ts = round_start
                else:
                    contact = self._get_next_contact_for_client(
                        cid,
                        round_start - 1,
                        self._sim_time_limit,
                    )
                    if contact is None:
                        round_complete = False
                        break
                    download_ts = contact[0]

                n_samples = client_data_sizes.get(cid, 100)
                actual_epochs = self.config.local_epochs
                train_start_ts = download_ts + download_slots
                if (
                    self.config.algorithm.lower() == "fedprox"
                    and self.config.contact_adaptive_epochs
                    and self.scheduler is not None
                ):
                    next_contact = self._get_next_contact_window_start(
                        cid,
                        train_start_ts,
                        self._sim_time_limit,
                    )
                    one_epoch_slots = max(
                        1,
                        self.time_model.compute_train_slots(cid, n_samples, 1),
                    )
                    if next_contact is not None:
                        available_slots = max(1, next_contact[0] - train_start_ts)
                        actual_epochs = max(
                            1,
                            min(
                                self.config.max_contact_epochs,
                                available_slots // one_epoch_slots,
                            ),
                        )

                train_slots = self.time_model.compute_train_slots(
                    cid,
                    n_samples,
                    actual_epochs,
                )
                train_end_ts = train_start_ts + train_slots
                upload_slots = self.time_model.compute_upload_slots(model_size_bytes)

                if self.scheduler is None:
                    arrival_ts = train_end_ts + upload_slots
                else:
                    upload_contact = self._get_next_contact_for_client(
                        cid,
                        train_end_ts - 1,
                        self._sim_time_limit,
                    )
                    if upload_contact is None:
                        round_complete = False
                        break
                    arrival_ts = upload_contact[0] + upload_slots

                if self._sim_time_limit is not None and arrival_ts >= self._sim_time_limit:
                    round_complete = False
                    break

                update = self._train_client(
                    cid,
                    train_loaders,
                    completed_rounds,
                    global_weights=global_weights,
                    local_epochs_override=actual_epochs,
                )
                if update is None:
                    round_complete = False
                    break

                update.started_at = train_start_ts
                update.completed_at = arrival_ts
                update.metadata.update(
                    {
                        "download_timeslot": download_ts,
                        "train_end_timeslot": train_end_ts,
                        "upload_timeslot": arrival_ts - upload_slots,
                        "actual_local_epochs": actual_epochs,
                    }
                )
                updates.append(update)
                arrival_times.append(arrival_ts)
                download_times.append(download_ts)
                train_end_times.append(train_end_ts)
                breakdown.per_satellite[cid] = {
                    "download_at": download_ts,
                    "train": train_slots,
                    "upload": upload_slots,
                    "arrive_at": arrival_ts,
                    "local_epochs": actual_epochs,
                }
                self._events.append(
                    {
                        "event": "client_update",
                        "algorithm": self.config.algorithm,
                        "round": completed_rounds,
                        "client_id": cid,
                        "base_version": completed_rounds,
                        "download_timeslot": download_ts,
                        "train_start_timeslot": train_start_ts,
                        "train_end_timeslot": train_end_ts,
                        "arrival_timeslot": arrival_ts,
                        "actual_local_epochs": actual_epochs,
                    }
                )

            if not round_complete or len(updates) != len(selected_ids):
                if verbose:
                    print(
                        f"  Round {completed_rounds + 1}: selected set could not finish "
                        "inside the simulation window"
                    )
                break

            current_ts = max(arrival_times, default=round_start)
            if self.aggregator.should_aggregate(updates, completed_rounds):
                new_weights = self.aggregator.aggregate(
                    global_weights,
                    updates,
                    completed_rounds,
                )
                with torch.no_grad():
                    for param, new_w in zip(self._global_model.parameters(), new_weights):
                        param.data.copy_(new_w)

            for cid in selected_ids:
                self._clients[cid].last_update_round = completed_rounds

            eval_metrics = self.evaluator.evaluate(
                self._global_model,
                test_loader,
                completed_rounds,
            )
            current_acc = eval_metrics.get("accuracy", 0)
            if hasattr(self.trainer, "update_accuracy"):
                _ = self.trainer.update_accuracy(current_acc)

            breakdown.download = download_slots
            breakdown.train = max(
                (item["train"] for item in breakdown.per_satellite.values()),
                default=0,
            )
            breakdown.upload = max(
                (item["upload"] for item in breakdown.per_satellite.values()),
                default=0,
            )
            breakdown.wait_distribution = max(download_times, default=round_start) - round_start
            breakdown.wait_return = current_ts - min(train_end_times, default=current_ts)
            breakdown.total = current_ts - round_start
            avg_loss = sum(u.train_loss for u in updates) / len(updates)
            result = FLRoundResult(
                round_num=completed_rounds,
                num_clients=len(updates),
                train_loss=round(avg_loss, 6),
                eval_metrics=eval_metrics,
                timeslot=current_ts,
                timeslot_start=round_start,
                time_breakdown=breakdown.to_dict(),
                extra={
                    "selected_client_ids": selected_ids,
                    "base_version": completed_rounds,
                    "client_updates": len(updates),
                },
            )
            self._history.append(result)
            self._events.append(
                {
                    "event": "server_aggregate",
                    "algorithm": self.config.algorithm,
                    "round": completed_rounds,
                    "timeslot": current_ts,
                    "client_ids": selected_ids,
                }
            )

            early_stop_acc = getattr(self.config, "early_stop_acc", None)
            if early_stop_acc is not None and current_acc >= early_stop_acc:
                break
            current_ts += max(1, self.config.timeslots_per_round if sim is None else 1)
            if verbose:
                acc = eval_metrics.get("accuracy", 0)
                print(
                    f"  Round {completed_rounds + 1:3d}/{self.config.num_rounds} | "
                    f"TS={round_start}->{result.timeslot} | clients={len(updates)} | "
                    f"accuracy={acc:.4f}"
                )

        return self._history

    def _advance_to_next_contact(
        self,
        from_ts: int,
        max_timeslot: int | None = None,
    ) -> int | None:
        """Find the next connected timeslot without extending the simulator window."""
        if self.scheduler is None:
            if max_timeslot is not None and from_ts >= max_timeslot:
                return None
            return from_ts

        sim = self.scheduler._sim
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
        for ts in range(start_ts, stop_ts):
            if self.scheduler.get_connected_sats(ts):
                return ts
        return None

    def _get_next_contact_for_client(
        self,
        sat_id: int,
        after_ts: int,
        max_timeslot: int | None = None,
    ) -> tuple[int, int] | None:
        """Find one client's next contact without crossing the simulation cap."""
        if self.scheduler is None:
            return None

        sim = self.scheduler._sim
        if max_timeslot is None:
            return sim.get_next_contact(sat_id, after_ts)

        start_ts = max(after_ts + 1, 0)
        stop_ts = min(max_timeslot, getattr(sim, "num_timeslots", max_timeslot))
        for ts in range(start_ts, stop_ts):
            gs_id = sim.contact_matrix.get_first_contact(sat_id, ts)
            if gs_id >= 0:
                return (ts, int(gs_id))
        return None

    def _get_next_contact_window_start(
        self,
        sat_id: int,
        after_ts: int,
        max_timeslot: int | None = None,
    ) -> tuple[int, int] | None:
        """Find the next disconnected-to-connected transition for one satellite."""
        if self.scheduler is None:
            return None
        sim = self.scheduler._sim
        stop_ts = int(getattr(sim, "num_timeslots", 0))
        if max_timeslot is not None:
            stop_ts = min(stop_ts, max_timeslot)
        start_ts = max(0, after_ts + 1)
        was_connected = (
            sim.contact_matrix.get_first_contact(sat_id, max(0, start_ts - 1)) >= 0
        )
        for ts in range(start_ts, stop_ts):
            gs_id = sim.contact_matrix.get_first_contact(sat_id, ts)
            connected = gs_id >= 0
            if connected and not was_connected:
                return (ts, int(gs_id))
            was_connected = connected
        return None

    def run_async(
        self,
        model: Any,
        train_loaders: dict[int, Any],
        test_loader: Any,
        verbose: bool = True,
    ) -> list[FLRoundResult]:
        """Run event-driven FedBuff with versioned client deltas."""
        from fl_space.fl.fedbuff import BufferAggregator

        self._global_model = copy.deepcopy(model)
        self._init_clients()
        self._history = []
        self._events = []

        if not isinstance(self.aggregator, BufferAggregator):
            raise TypeError(
                f"异步模式需要 BufferAggregator，当前聚合器: {type(self.aggregator).__name__}"
            )

        buffer_agg: BufferAggregator = self.aggregator
        sim = self.scheduler._sim if self.scheduler is not None else None
        self._sim_time_limit = None
        if sim is not None and getattr(self.config, "limit_to_sim_window", True):
            self._sim_time_limit = int(getattr(sim, "num_timeslots_pre", sim.num_timeslots))

        total_timeslots = self.config.num_rounds * max(1, self.config.timeslots_per_round)
        if self._sim_time_limit is not None:
            total_timeslots = min(total_timeslots, self._sim_time_limit)

        model_size_bytes = sum(
            p.numel() * p.element_size() for p in self._global_model.parameters()
        )
        client_data_sizes = self._get_client_data_sizes(train_loaders)
        pending_updates: dict[int, tuple[int, ClientUpdate]] = {}
        global_version = 0
        total_arrivals = 0
        eval_every = max(1, self.config.async_eval_every)

        baseline_metrics = self.evaluator.evaluate(self._global_model, test_loader, -1)
        self._history.append(
            FLRoundResult(
                round_num=-1,
                num_clients=0,
                train_loss=baseline_metrics.get("loss", 0.0),
                eval_metrics=baseline_metrics,
                timeslot=0,
                timeslot_start=0,
                extra={"phase": "baseline"},
            )
        )

        for ts in range(total_timeslots):
            self._update_connectivity(ts)

            ready_clients = sorted(
                cid
                for cid, (ready_ts, _update) in pending_updates.items()
                if ready_ts <= ts and self._clients[cid].is_connected
            )
            for cid in ready_clients:
                _ready_ts, update = pending_updates.pop(cid)
                update.completed_at = ts
                update.metadata["arrival_timeslot"] = ts
                buffer_agg.add_update(update)
                total_arrivals += 1
                self._clients[cid].last_update_round = global_version
                self._events.append(
                    {
                        "event": "client_arrival",
                        "algorithm": "fedbuff",
                        "timeslot": ts,
                        "client_id": cid,
                        "base_version": update.base_version,
                        "arrival_version": global_version,
                        "staleness": global_version - update.base_version,
                    }
                )

                while (
                    global_version < self.config.num_rounds
                    and buffer_agg.should_aggregate([], global_version)
                ):
                    before_version = global_version
                    global_weights = [
                        param.data.clone() for param in self._global_model.parameters()
                    ]
                    new_weights = buffer_agg.aggregate(
                        global_weights,
                        [],
                        global_version,
                    )
                    with torch.no_grad():
                        for param, new_w in zip(self._global_model.parameters(), new_weights):
                            param.data.copy_(new_w)
                    global_version += 1

                    status = buffer_agg.buffer_status()
                    should_evaluate = (
                        global_version % eval_every == 0
                        or global_version == self.config.num_rounds
                    )
                    eval_metrics = (
                        self.evaluator.evaluate(
                            self._global_model,
                            test_loader,
                            global_version,
                        )
                        if should_evaluate
                        else {}
                    )
                    staleness_values = status["last_staleness"]
                    result = FLRoundResult(
                        round_num=global_version,
                        num_clients=status["last_aggregate_count"],
                        train_loss=0.0,
                        eval_metrics=eval_metrics,
                        timeslot=ts,
                        timeslot_start=ts,
                        extra={
                            "base_server_version": before_version,
                            "client_ids": status["last_client_ids"],
                            "staleness": staleness_values,
                            "mean_staleness": (
                                sum(staleness_values) / len(staleness_values)
                                if staleness_values
                                else 0.0
                            ),
                            "buffer_remaining": status["current_count"],
                            "total_arrivals": total_arrivals,
                        },
                    )
                    self._history.append(result)
                    self._events.append(
                        {
                            "event": "server_aggregate",
                            "algorithm": "fedbuff",
                            "timeslot": ts,
                            "version": global_version,
                            "client_ids": status["last_client_ids"],
                            "staleness": staleness_values,
                            "buffer_remaining": status["current_count"],
                        }
                    )
                    if verbose:
                        acc = eval_metrics.get("accuracy")
                        acc_text = f"{acc:.4f}" if acc is not None else "not evaluated"
                        print(
                            f"  FedBuff update {global_version:3d}/{self.config.num_rounds} | "
                            f"TS={ts} | clients={status['last_client_ids']} | "
                            f"staleness={staleness_values} | accuracy={acc_text}"
                        )

            if global_version >= self.config.num_rounds:
                break

            busy_clients = set(pending_updates)
            available = self.selector.select(
                self._clients,
                global_version,
                already_training=busy_clients,
            )
            global_weights = [param.data.clone() for param in self._global_model.parameters()]
            download_slots = self.time_model.compute_download_slots(model_size_bytes)
            for cid in available:
                n_samples = client_data_sizes.get(cid, 100)
                train_slots = self.time_model.compute_train_slots(
                    cid,
                    n_samples,
                    self.config.local_epochs,
                )
                ready_ts = ts + max(1, download_slots + train_slots)
                update = self._train_client(
                    cid,
                    train_loaders,
                    global_version,
                    global_weights=global_weights,
                )
                if update is None:
                    continue
                update.base_version = global_version
                update.round_num = global_version
                update.started_at = ts
                update.metadata.update(
                    {
                        "ready_timeslot": ready_ts,
                        "download_slots": download_slots,
                        "train_slots": train_slots,
                    }
                )
                pending_updates[cid] = (ready_ts, update)
                self._events.append(
                    {
                        "event": "client_train_start",
                        "algorithm": "fedbuff",
                        "timeslot": ts,
                        "ready_timeslot": ready_ts,
                        "client_id": cid,
                        "base_version": global_version,
                    }
                )

        if self._history and self._history[-1].round_num >= 0:
            final_metrics = self.evaluator.evaluate(
                self._global_model,
                test_loader,
                global_version,
            )
            self._history[-1].eval_metrics = final_metrics

        self._events.append(
            {
                "event": "run_complete",
                "algorithm": "fedbuff",
                "timeslot": self._history[-1].timeslot if self._history else 0,
                "server_updates": global_version,
                "arrivals": total_arrivals,
                "buffered": buffer_agg.buffer_status()["current_count"],
                "pending": len(pending_updates),
            }
        )

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
            if r.extra:
                entry.update(r.extra)
            results.append(entry)
        return results

    def get_event_history(self) -> list[dict[str, Any]]:
        """Return communication and aggregation events for reproducibility."""
        return list(self._events)
