"""
FL 实验运行器 — 一站式 FL 实验入口

职责：
    - 加载数据集并分配到客户端
    - 创建模型
    - 组装组件 → FLServer
    - 执行训练 → 返回结果

设计原则：
    - 数据加载与算法逻辑解耦
    - 支持 IID 和 non-IID 数据分布
    - 结果以字典形式返回，便于后续可视化和分析

使用示例::

    from fl_space.fl.runner import FLRunner

    # 使用预设
    runner = FLRunner.from_preset("fedavg", "small", "mnist")
    history = runner.run(verbose=True)

    # 使用自定义配置
    from fl_space.fl.config import fedavg_config
    runner = FLRunner(config=fedavg_config(num_clients=15, local_epochs=10))
    history = runner.run()
"""

from __future__ import annotations

from typing import Any

from fl_space.fl.config import DATASET_PRESETS, FLConfig, get_preset_config
from fl_space.fl.core import (
    Aggregator,
    ClientSelector,
    Evaluator,
    FLRoundResult,
    LocalTrainer,
)
from fl_space.fl.models import get_model
from fl_space.fl.scheduler import CommunicationScheduler
from fl_space.fl.server import FLServer

# PyTorch 可选依赖
try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FLRunner:
    """
    FL 实验运行器。

    负责：数据准备 → 模型创建 → 组件组装 → 训练执行

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
        通信调度器（可选）。
    """

    def __init__(
        self,
        config: FLConfig,
        selector: ClientSelector,
        trainer: LocalTrainer,
        aggregator: Aggregator,
        evaluator: Evaluator,
        scheduler: CommunicationScheduler | None = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "FLRunner 需要 PyTorch 和 torchvision。"
                "请运行: pip install fl-space[full]"
            )

        self.config = config
        self.selector = selector
        self.trainer = trainer
        self.aggregator = aggregator
        self.evaluator = evaluator
        self.scheduler = scheduler

        # 运行时状态
        self._server: FLServer | None = None
        self._train_loaders: dict[int, DataLoader] = {}
        self._test_loader: DataLoader | None = None
        self._model: Any = None
        self._dataset_model_kwargs: dict[str, Any] = {}

    @classmethod
    def from_preset(
        cls,
        algorithm: str = "fedavg",
        scale: str = "small",
        dataset: str = "mnist",
        scheduler: CommunicationScheduler | None = None,
        device: str = "cpu",
        **overrides: Any,
    ) -> FLRunner:
        """
        从预设配置创建 Runner。

        Parameters
        ----------
        algorithm : str
            算法: "fedavg", "fedprox", "fedbuff"。
        scale : str
            规模: "small", "medium", "large"。
        dataset : str
            数据集: "mnist", "fashion_mnist", "cifar10"。
        scheduler : CommunicationScheduler | None
            通信调度器。
        device : str
            计算设备。
        **overrides
            其他覆盖参数。

        Returns
        -------
        FLRunner
            配置好的 Runner 实例。

        使用示例::

            runner = FLRunner.from_preset("fedprox", "medium", "cifar10", device="cuda")
            history = runner.run()
        """
        config = get_preset_config(
            algorithm=algorithm,
            scale=scale,
            dataset=dataset,
            device=device,
            **overrides,
        )

        # 根据算法创建组件
        algo = algorithm.lower()
        if algo == "fedavg":
            from fl_space.fl.fedavg import create_fedavg_components
            components = create_fedavg_components(
                fraction=config.fraction,
                min_clients=max(1, int(config.num_clients * config.fraction)),
                local_epochs=config.local_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                device=config.device,
                seed=config.seed,
                selection_strategy=config.selection_strategy,
            )
        elif algo == "fedprox":
            from fl_space.fl.fedprox import create_fedprox_components
            components = create_fedprox_components(
                fraction=config.fraction,
                min_clients=max(1, int(config.num_clients * config.fraction)),
                local_epochs=config.local_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                mu=config.mu,
                device=config.device,
                seed=config.seed,
                selection_strategy=config.selection_strategy,
            )
        elif algo == "fedbuff":
            from fl_space.fl.fedbuff import create_fedbuff_components
            components = create_fedbuff_components(
                min_clients=config.num_clients,
                local_epochs=config.local_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                buffer_size=config.buffer_size,
                staleness_weight=config.staleness_weight,
                server_learning_rate=config.server_learning_rate,
                mu=config.fedbuff_mu,
                max_staleness=config.max_staleness,
                device=config.device,
            )
        else:
            raise ValueError(f"未知算法: '{algo}'")

        return cls(config, *components, scheduler=scheduler)

    # ── 数据准备 ──────────────────────────────────────────────

    def prepare_data(
        self,
        dataset_name: str = "mnist",
        iid: bool = True,
        alpha: float = 0.5,
        classes_per_client: int | None = None,
        max_samples_per_client: int = 0,
        data_dir: str = "./data",
        partition_strategy: str = "probability",
        class_probability: float = 0.8,
        preference_mode: str = "class_balanced",
        preferred_clients_per_class: int = 1,
        sample_cap_strategy: str = "preserve",
        satellite_data_profiles: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """
        加载数据集并分配到客户端。

        支持 IID（均匀随机分配）和 non-IID（Dirichlet 或固定类别数）。

        Parameters
        ----------
        dataset_name : str
            数据集名称: "mnist", "fashion_mnist", "cifar10"。
        iid : bool
            True 表示 IID 分配，False 表示 non-IID。
        alpha : float
            Dirichlet 分布参数（仅 non-IID 时有效）。
            alpha 越小分布越不均匀。
        classes_per_client : int | None
            每个客户端限定的类别数，None 使用 Dirichlet。
        max_samples_per_client : int
            每个客户端样本数上限，0 表示不限制。
        data_dir : str
            数据下载/缓存目录。

        Raises
        ------
        ImportError
            torchvision 未安装时。
        """
        if not TORCH_AVAILABLE:
            raise ImportError("数据加载需要 torchvision")

        dataset_name = dataset_name.lower()

        # 选择数据集和变换
        if dataset_name in ("mnist", "fashion_mnist"):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            ds_cls = (
                datasets.MNIST if dataset_name == "mnist"
                else datasets.FashionMNIST
            )
            train_ds = ds_cls(data_dir, train=True, download=True, transform=transform)
            test_ds = ds_cls(data_dir, train=False, download=True, transform=transform)
            self._dataset_model_kwargs = {}

        elif dataset_name == "cifar10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ])
            train_ds = datasets.CIFAR10(
                data_dir, train=True, download=True, transform=transform_train,
            )
            test_ds = datasets.CIFAR10(
                data_dir, train=False, download=True, transform=transform_test,
            )
            self._dataset_model_kwargs = {}
        elif dataset_name == "femnist":
            # FEMNIST writer-level (LEAF benchmark)
            # 使用项目内置 loader，自动下载/缓存
            from fl_space.fl.femnist_loader import get_femnist_split

            # max_writers 由 num_clients 决定（每 writer = 1 client）
            max_writers = self.config.num_clients
            # samples_per_writer 对齐论文 200~350
            samples_per_writer = (
                max_samples_per_client if max_samples_per_client > 0 else 300
            )
            train_ds, test_ds, writer_ids = get_femnist_split(
                data_dir=data_dir,
                max_writers=max_writers,
                samples_per_writer=samples_per_writer,
                download=True,
            )
            self._dataset_model_kwargs = {}
            # 保存 writer_ids 用于自定义 partition
            self._femnist_writer_ids = writer_ids
            self._femnist_train_ds = train_ds

        elif dataset_name in ("imagefolder", "custom"):
            train_dir = f"{data_dir}/train"
            test_dir = f"{data_dir}/test"
            transform_train = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ])
            transform_test = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ])
            train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
            test_ds = datasets.ImageFolder(test_dir, transform=transform_test)
            self._dataset_model_kwargs = {
                "num_classes": len(train_ds.classes),
                "in_channels": 3,
            }
        else:
            raise ValueError(f"不支持的数据集: '{dataset_name}'")

        # 分配数据到客户端
        n_clients = self.config.num_clients
        profiles = satellite_data_profiles or getattr(
            self.config,
            "satellite_data_profiles",
            {},
        )
        if dataset_name == "femnist":
            # FEMNIST: 按 writer 划分 (1 writer = 1 client)，已经是天然 non-IID
            client_data_indices = self._partition_femnist_writers(
                train_ds, n_clients
            )
        elif profiles:
            targets = np.array([train_ds[i][1] for i in range(len(train_ds))])
            client_data_indices = self._partition_satellite_profiles(
                targets=targets,
                n_clients=n_clients,
                profiles=profiles,
                default_probability=class_probability,
            )
        else:
            client_data_indices = self._partition_data(
                train_ds, n_clients, iid=iid, alpha=alpha,
                classes_per_client=classes_per_client,
                partition_strategy=partition_strategy,
                class_probability=class_probability,
                preference_mode=preference_mode,
                preferred_clients_per_class=preferred_clients_per_class,
            )

        # ── 样本数截断（模拟太空 FL 小数据集）──
        targets = np.array([train_ds[i][1] for i in range(len(train_ds))])
        if max_samples_per_client > 0 and dataset_name != "femnist":
            # FEMNIST 在 loader 中已经截断
            client_data_indices = self._cap_client_samples(
                client_data_indices,
                targets,
                max_samples_per_client,
                sample_cap_strategy,
            )
        if profiles:
            client_data_indices = self._cap_satellite_profiles(
                client_data_indices,
                profiles,
            )

        self._client_label_distribution = self._build_label_distribution(
            client_data_indices,
            targets,
        )

        client_class_dist = {}
        for cid, indices in enumerate(client_data_indices):
            classes_present = sorted({train_ds[i][1] for i in indices})
            client_class_dist[cid] = (len(indices), classes_present)
        self._client_data_summary = client_class_dist
        self._satellite_data_profiles = profiles

        # 创建 DataLoader
        self._train_loaders = {}
        num_workers = getattr(self.config, 'num_workers', 0) or 0
        for cid, indices in enumerate(client_data_indices):
            subset = Subset(train_ds, indices)
            generator = None
            if getattr(self.config, "seed", None) is not None:
                generator = torch.Generator()
                generator.manual_seed(int(self.config.seed) + cid)
            self._train_loaders[cid] = DataLoader(
                subset,
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=(self.config.device == "cuda"),
                generator=generator,
            )

        self._test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.config.device == "cuda"),
        )

    def _partition_femnist_writers(
        self, dataset: Dataset, n_clients: int
    ) -> list[list[int]]:
        """FEMNIST writer-level partition: 1 writer = 1 client.

        这是最自然的 non-IID 划分，每个 writer 的笔迹风格独特，
        类别分布相对均衡但 feature 漂移大（论文 FEMNIST 语义）。

        Parameters
        ----------
        dataset : FEMNISTWriterDataset
            已按 max_writers 截断的数据集，含 writer_to_indices 字典。
        n_clients : int
            客户端数 (应与 max_writers 一致)。
        """
        writer_to_indices = getattr(dataset, "writer_to_indices", None)
        if not writer_to_indices:
            raise ValueError("FEMNIST 数据集未提供 writer_to_indices 映射")

        writer_ids = sorted(writer_to_indices.keys())[:n_clients]
        client_indices: list[list[int]] = []
        for wid in writer_ids:
            client_indices.append(list(writer_to_indices[wid]))

        # 若 writer 数 < n_clients，剩余客户端分空 (后续 selector 会跳过)
        while len(client_indices) < n_clients:
            client_indices.append([])

        return client_indices

    def _partition_data(
        self,
        dataset: Dataset,
        n_clients: int,
        iid: bool = True,
        alpha: float = 0.5,
        classes_per_client: int | None = None,
        partition_strategy: str = "probability",
        class_probability: float = 0.8,
        preference_mode: str = "class_balanced",
        preferred_clients_per_class: int = 1,
    ) -> list[list[int]]:
        """Split a dataset into client shards for IID and non-IID experiments."""
        n_samples = len(dataset)
        targets = np.array([dataset[i][1] for i in range(n_samples)])
        classes = np.unique(targets)
        n_classes = len(classes)
        rng = np.random.default_rng(getattr(self.config, "seed", None))

        strategy = (partition_strategy or "probability").lower()
        if iid:
            strategy = "iid"
        elif classes_per_client is not None and classes_per_client <= 0:
            classes_per_client = None
            if strategy in ("probability", "shard"):
                strategy = "dirichlet"

        if strategy == "iid":
            indices = rng.permutation(n_samples)
            split_size = n_samples // n_clients
            client_indices = [
                indices[i * split_size : (i + 1) * split_size].tolist()
                for i in range(n_clients)
            ]
            remainder = indices[n_clients * split_size :]
            if len(remainder) > 0:
                client_indices[-1].extend(remainder.tolist())
            return client_indices

        if strategy == "shard" and classes_per_client is not None:
            client_indices = [[] for _ in range(n_clients)]
            k = min(classes_per_client, n_classes)
            class_to_clients: dict[int, list[int]] = {int(c): [] for c in classes}
            for cid in range(n_clients):
                for offset in range(k):
                    cls = int(classes[(cid + offset) % n_classes])
                    class_to_clients[cls].append(cid)

            for c in classes:
                c_int = int(c)
                class_indices = np.where(targets == c)[0]
                rng.shuffle(class_indices)
                holders = class_to_clients[c_int]
                if not holders:
                    continue
                split_sz = len(class_indices) // len(holders)
                start_idx = 0
                for h_idx, cid in enumerate(holders):
                    extra = 1 if h_idx < len(class_indices) % len(holders) else 0
                    end_idx = start_idx + split_sz + extra
                    client_indices[cid].extend(class_indices[start_idx:end_idx].tolist())
                    start_idx = end_idx
            return client_indices

        if strategy == "probability" and classes_per_client is not None:
            return self._partition_probability(
                targets=targets,
                classes=classes,
                n_clients=n_clients,
                class_probability=class_probability,
                preference_mode=preference_mode,
                classes_per_client=classes_per_client,
                preferred_clients_per_class=preferred_clients_per_class,
                rng=rng,
            )

        client_indices = [[] for _ in range(n_clients)]
        for c in classes:
            class_indices = np.where(targets == c)[0]
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.repeat(alpha, n_clients))
            proportions = (proportions * len(class_indices)).astype(int)
            proportions[-1] += len(class_indices) - proportions.sum()

            start_idx = 0
            for cid in range(n_clients):
                end_idx = start_idx + proportions[cid]
                client_indices[cid].extend(class_indices[start_idx:end_idx].tolist())
                start_idx = end_idx

        return client_indices

    def _partition_probability(
        self,
        *,
        targets: np.ndarray,
        classes: np.ndarray,
        n_clients: int,
        class_probability: float,
        preference_mode: str,
        classes_per_client: int,
        preferred_clients_per_class: int,
        rng: np.random.Generator,
    ) -> list[list[int]]:
        client_indices = [[] for _ in range(n_clients)]
        p_preferred = min(max(float(class_probability), 0.0), 1.0)
        mode = (preference_mode or "class_balanced").lower()

        if mode == "class_balanced":
            holders = max(1, min(int(preferred_clients_per_class), n_clients))
            for class_pos, c in enumerate(classes):
                preferred = [(class_pos + offset) % n_clients for offset in range(holders)]
                class_indices = np.where(targets == c)[0]
                rng.shuffle(class_indices)
                for sample_idx in class_indices:
                    if rng.random() < p_preferred:
                        cid = int(rng.choice(preferred))
                    else:
                        cid = int(rng.integers(0, n_clients))
                    client_indices[cid].append(int(sample_idx))
            return client_indices

        k = min(classes_per_client, len(classes))
        class_to_clients: dict[int, list[int]] = {int(c): [] for c in classes}
        for cid in range(n_clients):
            for offset in range(k):
                cls = int(classes[(cid + offset) % len(classes)])
                class_to_clients[cls].append(cid)

        for c in classes:
            c_int = int(c)
            class_indices = np.where(targets == c)[0]
            rng.shuffle(class_indices)
            preferred = class_to_clients.get(c_int) or list(range(n_clients))
            for sample_idx in class_indices:
                if rng.random() < p_preferred:
                    cid = int(rng.choice(preferred))
                else:
                    cid = int(rng.integers(0, n_clients))
                client_indices[cid].append(int(sample_idx))
        return client_indices

    def _partition_satellite_profiles(
        self,
        *,
        targets: np.ndarray,
        n_clients: int,
        profiles: dict[str, dict[str, Any]],
        default_probability: float,
    ) -> list[list[int]]:
        """Partition samples with independently configurable satellite preferences."""
        rng = np.random.default_rng(getattr(self.config, "seed", None))
        classes = sorted(int(value) for value in np.unique(targets))
        result = [[] for _ in range(n_clients)]

        normalized: dict[int, tuple[set[int], float]] = {}
        for raw_id, raw_profile in profiles.items():
            try:
                client_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if not 0 <= client_id < n_clients or not isinstance(raw_profile, dict):
                continue
            preferred: set[int] = set()
            for value in raw_profile.get("preferred_classes", []):
                try:
                    class_id = int(value)
                except (TypeError, ValueError):
                    continue
                if class_id in classes:
                    preferred.add(class_id)
            probability = float(
                raw_profile.get("preference_probability", default_probability)
            )
            normalized[client_id] = (preferred, min(max(probability, 0.0), 1.0))

        for class_id in classes:
            class_indices = np.where(targets == class_id)[0]
            rng.shuffle(class_indices)
            preferred_clients = [
                client_id
                for client_id, (preferred, _probability) in normalized.items()
                if class_id in preferred
            ]
            fallback_clients = [
                client_id for client_id in range(n_clients)
                if client_id not in preferred_clients
            ] or list(range(n_clients))
            class_preference_probability = (
                sum(normalized[client_id][1] for client_id in preferred_clients)
                / len(preferred_clients)
                if preferred_clients
                else 0.0
            )
            for sample_index in class_indices:
                if preferred_clients and rng.random() < class_preference_probability:
                    client_id = int(rng.choice(preferred_clients))
                else:
                    client_id = int(rng.choice(fallback_clients))
                result[client_id].append(int(sample_index))
        return result

    @staticmethod
    def _cap_satellite_profiles(
        client_indices: list[list[int]],
        profiles: dict[str, dict[str, Any]],
    ) -> list[list[int]]:
        """Apply optional per-satellite sample limits after preference routing."""
        capped: list[list[int]] = []
        for client_id, indices in enumerate(client_indices):
            profile = profiles.get(str(client_id), profiles.get(client_id, {}))
            raw_limit = profile.get("max_samples", 0) if isinstance(profile, dict) else 0
            try:
                limit = max(0, int(raw_limit))
            except (TypeError, ValueError):
                limit = 0
            capped.append(indices[:limit] if limit else indices)
        return capped

    def _cap_client_samples(
        self,
        client_indices: list[list[int]],
        targets: np.ndarray,
        max_samples_per_client: int,
        sample_cap_strategy: str,
    ) -> list[list[int]]:
        rng = np.random.default_rng(getattr(self.config, "seed", None))
        strategy = (sample_cap_strategy or "preserve").lower()
        capped: list[list[int]] = []

        for indices in client_indices:
            if len(indices) <= max_samples_per_client:
                capped.append(indices)
                continue

            if strategy == "balanced":
                unique_cls = np.unique(targets[indices])
                truncated: list[int] = []
                samples_per_class = max_samples_per_client // len(unique_cls)
                remainder = max_samples_per_client % len(unique_cls)
                for cls in unique_cls:
                    cls_idx = [idx for idx in indices if targets[idx] == cls]
                    rng.shuffle(cls_idx)
                    take = min(samples_per_class + (1 if remainder > 0 else 0), len(cls_idx))
                    if remainder > 0:
                        remainder -= 1
                    truncated.extend(cls_idx[:take])
                capped.append(truncated)
                continue

            shuffled = np.array(indices)
            rng.shuffle(shuffled)
            capped.append(shuffled[:max_samples_per_client].astype(int).tolist())

        return capped

    def _build_label_distribution(
        self,
        client_indices: list[list[int]],
        targets: np.ndarray,
    ) -> dict[str, Any]:
        labels = sorted(int(x) for x in np.unique(targets))
        clients: dict[str, Any] = {}
        entropies: list[float] = []

        for cid, indices in enumerate(client_indices):
            counts = dict.fromkeys(labels, 0)
            for idx in indices:
                counts[int(targets[idx])] += 1

            total = sum(counts.values())
            ratios = {label: count / total for label, count in counts.items() if total > 0}
            entropy = -sum(ratio * float(np.log2(ratio)) for ratio in ratios.values() if ratio > 0)
            entropies.append(entropy)
            top_classes = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
            clients[str(cid)] = {
                "num_samples": total,
                "num_classes": sum(1 for count in counts.values() if count > 0),
                "entropy": round(entropy, 6),
                "class_counts": {str(label): int(count) for label, count in counts.items()},
                "top_classes": [
                    {
                        "label": int(label),
                        "count": int(count),
                        "ratio": round(count / total, 6) if total > 0 else 0.0,
                    }
                    for label, count in top_classes
                ],
            }

        return {
            "num_clients": len(client_indices),
            "num_classes": len(labels),
            "avg_entropy": round(float(np.mean(entropies)), 6) if entropies else 0.0,
            "clients": clients,
        }

    def prepare_model(
        self,
        model_name: str = "mlp",
        **model_kwargs: Any,
    ) -> None:
        """
        创建模型实例。

        Parameters
        ----------
        model_name : str
            模型名称: "mlp" 或 "simplecnn"。
        **model_kwargs
            传递给模型构造函数的参数。
        """
        self._model = get_model(model_name, **model_kwargs)

    # ── 执行训练 ──────────────────────────────────────────────

    def run(
        self,
        dataset_name: str = "mnist",
        iid: bool | None = None,
        alpha: float | None = None,
        classes_per_client: int | None = None,
        max_samples_per_client: int | None = None,
        data_dir: str = "./data",
        partition_strategy: str | None = None,
        class_probability: float | None = None,
        preference_mode: str | None = None,
        preferred_clients_per_class: int | None = None,
        sample_cap_strategy: str | None = None,
        satellite_data_profiles: dict[str, dict[str, Any]] | None = None,
        verbose: bool = True,
    ) -> list[FLRoundResult]:
        """
        执行完整的 FL 训练流程。

        自动完成：数据加载 → 模型创建 → 训练执行。

        数据划分参数优先从 config 读取，方法参数仅作覆盖用途。

        Parameters
        ----------
        dataset_name : str
            数据集名称。
        iid : bool | None
            是否 IID 分配（None 则从 config 推断）。
        alpha : float | None
            non-IID Dirichlet 参数。
        classes_per_client : int | None
            每客户端类别数（None 则从 config 读取）。
        max_samples_per_client : int | None
            每客户端样本上限（None 则从 config 读取，0=不限）。
        data_dir : str
            数据目录。
        verbose : bool
            是否打印进度。

        Returns
        -------
        list[FLRoundResult]
            每轮训练结果。

        使用示例::

            runner = FLRunner.from_preset("fedavg", "small", "mnist")
            history = runner.run(dataset_name="mnist", iid=True, verbose=True)

            # 查看结果
            for r in history:
                print(f"轮次 {r.round_num}: 准确率 {r.eval_metrics['accuracy']}")
        """
        # 从 config 读取默认值
        _iid = iid if iid is not None else (getattr(self.config, 'classes_per_client', 0) == 0)
        _alpha = alpha if alpha is not None else 0.5
        _cpc = (
            classes_per_client if classes_per_client is not None
            else getattr(self.config, 'classes_per_client', None)
        )
        _max_spc = (
            max_samples_per_client if max_samples_per_client is not None
            else getattr(self.config, 'max_samples_per_client', 0)
        )
        _data_dir = data_dir if data_dir != "./data" else getattr(self.config, "data_dir", data_dir)
        _partition_strategy = (
            partition_strategy if partition_strategy is not None
            else getattr(self.config, "partition_strategy", "probability")
        )
        _class_probability = (
            class_probability if class_probability is not None
            else getattr(self.config, "class_probability", 0.8)
        )
        _preference_mode = (
            preference_mode if preference_mode is not None
            else getattr(self.config, "preference_mode", "class_balanced")
        )
        _preferred_clients_per_class = (
            preferred_clients_per_class if preferred_clients_per_class is not None
            else getattr(self.config, "preferred_clients_per_class", 1)
        )
        _sample_cap_strategy = (
            sample_cap_strategy if sample_cap_strategy is not None
            else getattr(self.config, "sample_cap_strategy", "preserve")
        )
        _satellite_data_profiles = (
            satellite_data_profiles
            if satellite_data_profiles is not None
            else getattr(self.config, "satellite_data_profiles", {})
        )

        if verbose:
            print("=== SpaceFL 实验 ===")
            print(f"  算法: {self.config.algorithm}")
            print(f"  数据集: {dataset_name}")
            print(f"  客户端: {self.config.num_clients}")
            print(f"  轮次: {self.config.num_rounds}")
            print(f"  设备: {self.config.device}")
            if _cpc:
                print(f"  数据: non-IID (每客户端 {_cpc} 类, 上限 {_max_spc} 样本)")
            elif not _iid:
                print(f"  数据: non-IID (Dirichlet α={_alpha}, 上限 {_max_spc} 样本)")
            else:
                print(f"  数据: IID (上限 {_max_spc} 样本)" if _max_spc else "  数据: IID")
            print()

        # 1. 数据准备
        if verbose:
            print("[1/3] 加载数据...")
        self.prepare_data(
            dataset_name=dataset_name,
            iid=_iid,
            alpha=_alpha,
            classes_per_client=_cpc,
            max_samples_per_client=_max_spc,
            data_dir=_data_dir,
            partition_strategy=_partition_strategy,
            class_probability=_class_probability,
            preference_mode=_preference_mode,
            preferred_clients_per_class=_preferred_clients_per_class,
            sample_cap_strategy=_sample_cap_strategy,
            satellite_data_profiles=_satellite_data_profiles,
        )

        # 2. 模型准备
        if verbose:
            print("[2/3] 创建模型...")
        # 固定 PyTorch 随机种子，确保模型初始化可复现
        if TORCH_AVAILABLE and hasattr(self.config, 'seed') and self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        ds_preset = DATASET_PRESETS.get(dataset_name, DATASET_PRESETS["mnist"])
        model_kwargs = dict(ds_preset.get("model_kwargs", {}))
        model_kwargs.update(self._dataset_model_kwargs)
        self.prepare_model(
            model_name=ds_preset["model"],
            **model_kwargs,
        )

        # 3. 训练
        if verbose:
            print("[3/3] 开始训练...")
            print()

        self._server = FLServer(
            config=self.config,
            selector=self.selector,
            trainer=self.trainer,
            aggregator=self.aggregator,
            evaluator=self.evaluator,
            scheduler=self.scheduler,
        )

        history = self._server.run(
            model=self._model,
            train_loaders=self._train_loaders,
            test_loader=self._test_loader,
            verbose=verbose,
        )

        if verbose:
            print()
            print("=== 训练完成 ===")
            if history:
                final_acc = history[-1].eval_metrics.get("accuracy", 0)
                print(f"  最终准确率: {final_acc:.4f}")
                first_ts = history[0].timeslot_start
                last_ts = history[-1].timeslot
                total_slots = last_ts - first_ts
                # 使用 time_model 显示可读时间
                tm = self._server.time_model
                ts_dur = getattr(tm, "timeslot_duration_min", 1.0)
                time_str = tm.slots_to_display(total_slots, ts_dur) if tm else f"{total_slots} slots"
                print(f"  虚拟时间范围: TS{first_ts} → TS{last_ts} "
                      f"({time_str}, {len(history)} 轮)")
                print(f"  时间模型: {tm.name}")

        return history

    @property
    def client_label_distribution(self) -> dict[str, Any]:
        """Client-side label distribution generated during data preparation."""
        return getattr(self, "_client_label_distribution", {})

    @property
    def history_dict(self) -> list[dict[str, Any]]:
        """训练历史（字典格式，便于 JSON 序列化）。"""
        if self._server is None:
            return []
        return self._server.get_history_dict()

    @property
    def event_history(self) -> list[dict[str, Any]]:
        """Detailed client and server events from the latest run."""
        if self._server is None:
            return []
        return self._server.get_event_history()
