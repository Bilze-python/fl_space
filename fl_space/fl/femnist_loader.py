"""FEMNIST writer-level 数据集加载器（LEAF benchmark）.

支持两种来源（自动选择，按优先级）：
    1. HuggingFace `flwrlabs/femnist` (推荐，一行下载，无需手动操作)
    2. 本地 LEAF JSON 目录 (用户已下载的 LEAF 数据)

设计目标
--------
- 按 writer (user_id) 划分客户端，模拟论文
  "Bringing Federated Learning to Space" 中的 FEMNIST writer-level 异构。
- 自动缓存到 `<data_dir>/femnist_cache.pkl`，避免重复下载/解析。
- 暴露与 torchvision.datasets 兼容的 `__getitem__`/`__len__` 接口，
  可直接被 `fl_space.fl.runner.FLRunner.prepare_data` 调用。

参考
----
- LEAF repo: https://github.com/TalwalkarLab/leaf/tree/master/data/femnist
- HF dataset: https://huggingface.co/datasets/flwrlabs/femnist
- 论文 FEMNIST: 62 类 (10 数字 + 26 小写 + 26 大写), 每 writer 200~350 样本
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

# torchvision 仅用于 transform / Dataset 基类
try:
    from torch.utils.data import Dataset
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # type: ignore[assignment, misc]


# 62 类标签：0-9 数字, 10-35 大写字母, 36-61 小写字母
NUM_CLASSES = 62
IMG_SIZE = 28 * 28  # 784 像素


class FEMNISTWriterDataset(Dataset):
    """FEMNIST writer-level 数据集 (torchvision 兼容接口).

    `__getitem__` 返回 `(tensor_image, label)`，label 已是 0~61 的整数索引。
    """

    def __init__(
        self,
        data_dir: str = "./data",
        train: bool = True,
        download: bool = True,
        transform: Any = None,
        max_writers: int | None = None,
        samples_per_writer: int | None = None,
        cache_override: str | None = None,
    ) -> None:
        """初始化 FEMNIST writer-level 数据集.

        Parameters
        ----------
        data_dir : str
            数据缓存根目录 (会在此目录下创建 `femnist_cache.pkl`).
        train : bool
            True 加载训练集, False 加载测试集.
        download : bool
            若本地缓存不存在，是否自动下载。
        transform : callable, optional
            torchvision transform pipeline.
        max_writers : int, optional
            仅保留前 N 个 writer (按 user_id 排序)，用于小规模实验。
        samples_per_writer : int, optional
            每个 writer 最多保留 N 个样本 (按比例缩放，对齐论文 200~350 样本).
        cache_override : str, optional
            指定缓存文件路径，覆盖默认的 `<data_dir>/femnist_cache.pkl`.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "FEMNIST 数据加载需要 torchvision + torch. "
                "请运行: pip install fl-space[full]"
            )

        self.data_dir = data_dir
        self.train = train
        self.transform = transform

        cache_path = Path(cache_override) if cache_override else Path(data_dir) / "femnist_cache.pkl"

        # 加载或构建缓存
        cache = self._load_or_build_cache(cache_path, download)

        # train/test 切分：80% / 20% per writer (LEAF 默认)
        split_key = "train" if train else "test"
        writers_data = cache.get(split_key, {})

        # 可选过滤
        if max_writers is not None and max_writers > 0:
            sorted_writers = sorted(writers_data.keys())[:max_writers]
            writers_data = {w: writers_data[w] for w in sorted_writers}
        if samples_per_writer is not None and samples_per_writer > 0:
            rng = np.random.default_rng(42)
            new_data = {}
            for w, items in writers_data.items():
                if len(items) > samples_per_writer:
                    idx = rng.choice(len(items), samples_per_writer, replace=False)
                    new_data[w] = [items[i] for i in idx]
                else:
                    new_data[w] = items
            writers_data = new_data

        # 扁平化为列表：[(image_array, label, writer_id), ...]
        self.samples: list[tuple[np.ndarray, int, str]] = []
        for writer_id, items in writers_data.items():
            for img, label in items:
                self.samples.append((img, label, writer_id))

        # 维护 writer -> index 映射 (用于 partition)
        self.writer_to_indices: dict[str, list[int]] = {}
        for idx, (_, _, wid) in enumerate(self.samples):
            self.writer_to_indices.setdefault(wid, []).append(idx)

        # 标签统计 (用于 _partition_data 兼容)
        self._targets = np.array([s[1] for s in self.samples])

    def _load_or_build_cache(
        self, cache_path: Path, download: bool
    ) -> dict[str, dict[str, list[tuple[np.ndarray, int]]]]:
        """加载或构建 FEMNIST 缓存.

        返回结构: {"train": {writer_id: [(img, label), ...]}, "test": {...}}
        """
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"[FEMNIST] 缓存文件损坏，将重建: {e}", file=sys.stderr)

        if not download:
            raise FileNotFoundError(
                f"FEMNIST 缓存不存在且 download=False: {cache_path}"
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[FEMNIST] 首次加载，开始下载/解析到 {cache_path} ...")

        # 优先尝试 HuggingFace datasets
        cache = None
        try:
            cache = self._build_from_huggingface()
        except Exception as e:
            print(f"[FEMNIST] HuggingFace 加载失败: {e}", file=sys.stderr)
            print("[FEMNIST] 尝试从本地 LEAF JSON 加载...", file=sys.stderr)
            cache = None

        if cache is None:
            cache = self._build_from_leaf_json(self.data_dir)

        with cache_path.open("wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        total_train = sum(len(v) for v in cache.get("train", {}).values())
        total_test = sum(len(v) for v in cache.get("test", {}).values())
        print(
            f"[FEMNIST] 缓存构建完成: "
            f"{len(cache.get('train', {}))} writers, "
            f"{total_train} train / {total_test} test samples"
        )
        return cache

    def _build_from_huggingface(self) -> dict[str, dict[str, list[tuple[np.ndarray, int]]]]:
        """从 HuggingFace flwrlabs/femnist 加载.

        依赖: `pip install datasets`
        国内可用: HF_ENDPOINT=https://hf-mirror.com 环境变量加速。

        flwrlabs/femnist 字段:
          - 'image' (PIL.Image, 28x28 灰度)
          - 'writer_id' (str, writer 标识, e.g. 'f0000_14')
          - 'character' (int, 0-61 标签, 已是整数无需转换)
          - 'hsf_id' (int, 手写风格 ID)
        """
        from datasets import load_dataset

        ds_train = load_dataset("flwrlabs/femnist", split="train")

        cache: dict[str, dict[str, list[tuple[np.ndarray, int]]]] = {
            "train": {},
            "test": {},
        }

        # 一次遍历构建 train，同时收集 per-writer 数据用于 train/test 切分
        all_writer_data: dict[str, list[tuple[np.ndarray, int]]] = {}
        for sample in ds_train:
            img_pil = sample["image"]
            label = sample["character"]  # flwrlabs/femnist 已是 0-61 整数
            writer_id = sample["writer_id"]
            if not isinstance(label, int):
                continue
            img_arr = np.array(img_pil, dtype=np.uint8)  # (28, 28)
            all_writer_data.setdefault(writer_id, []).append((img_arr, label))

        # 按 writer 80/20 切分 (LEAF 默认)
        rng = np.random.default_rng(42)
        for writer_id, items in all_writer_data.items():
            n = len(items)
            if n <= 1:
                cache["train"][writer_id] = items
            else:
                n_test = max(1, int(n * 0.2))
                idx = rng.choice(n, n_test, replace=False)
                idx_set = set(idx)
                test_items = [items[i] for i in idx]
                train_items = [items[i] for i in range(n) if i not in idx_set]
                cache["train"][writer_id] = train_items
                cache["test"][writer_id] = test_items

        return cache

    def _build_from_leaf_json(
        self, data_dir: str
    ) -> dict[str, dict[str, list[tuple[np.ndarray, int]]]]:
        """从本地 LEAF JSON 目录加载 (用户已 download.sh 后的场景).

        期望目录结构:
            <data_dir>/femnist/data/train/all_data_*.json
            <data_dir>/femnist/data/test/all_data_*.json
        """
        leaf_root = Path(data_dir) / "femnist" / "data"
        if not leaf_root.exists():
            raise FileNotFoundError(
                f"未找到 LEAF FEMNIST 数据: {leaf_root}\n"
                f"请执行以下步骤之一:\n"
                f"  1. pip install datasets  (推荐, 自动从 HF 下载)\n"
                f"  2. git clone https://github.com/TalwalkarLab/leaf.git && "
                f"cd leaf/data/femnist && bash download.sh && "
                f"./preprocess.sh -s niid --sf 1.0 -k 0 --t user --tf 0.8"
            )

        cache: dict[str, dict[str, list[tuple[np.ndarray, int]]]] = {
            "train": {},
            "test": {},
        }
        for split in ("train", "test"):
            split_dir = leaf_root / split
            if not split_dir.exists():
                continue
            for json_file in sorted(split_dir.glob("all_data_*.json")):
                with json_file.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                for user_data in payload.get("user_data", {}).values():
                    # LEAF 格式: {'x': [[像素列表], ...], 'y': [字符, ...]}
                    xs = user_data.get("x", [])
                    ys = user_data.get("y", [])
                    for x, y in zip(xs, ys):
                        label = _char_to_label(str(y))
                        if label is None:
                            continue
                        # x 是 784 长度的 list of int (0~255)
                        img_arr = np.array(x, dtype=np.uint8).reshape(28, 28)
                        # writer_id 在 LEAF 中是 user_data 的 key，这里用文件名+index
                        writer_id = json_file.stem + "_" + str(len(cache[split]))
                        cache[split].setdefault(writer_id, []).append((img_arr, label))
        return cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        img_arr, label, _ = self.samples[idx]
        # 转为 PIL-like (HWC) 以兼容 torchvision transform
        if self.transform is not None:
            from PIL import Image
            img = Image.fromarray(img_arr, mode="L")
            return self.transform(img), label
        # 默认返回 tensor (与 MNIST 行为一致)
        import torch
        return torch.from_numpy(img_arr).float().unsqueeze(0) / 255.0, label

    @property
    def targets(self) -> np.ndarray:
        """兼容 FLRunner._partition_data 中 `targets = np.array([dataset[i][1] ...])`."""
        return self._targets

    @property
    def classes(self) -> np.ndarray:
        return np.arange(NUM_CLASSES)


# ── 辅助函数 ────────────────────────────────────────────────────

_CHAR_TO_LABEL: dict[str, int] = {}


def _build_char_map() -> None:
    """构建 0-9 / A-Z / a-z → 0-61 的映射."""
    if _CHAR_TO_LABEL:
        return
    # 数字 0-9
    for i, c in enumerate("0123456789"):
        _CHAR_TO_LABEL[c] = i
    # 大写 A-Z (10-35)
    for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        _CHAR_TO_LABEL[c] = 10 + i
    # 小写 a-z (36-61)
    for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
        _CHAR_TO_LABEL[c] = 36 + i
    # LEAF 有时会用数字字符串表示数字
    for i in range(10):
        _CHAR_TO_LABEL[str(i)] = i


def _char_to_label(char: str) -> int | None:
    """字符 -> 0~61 标签索引. 未知字符返回 None."""
    _build_char_map()
    return _CHAR_TO_LABEL.get(char)


def get_default_transform():
    """返回 FEMNIST 标准 transform (与 MNIST 一致)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def get_femnist_split(
    data_dir: str = "./data",
    max_writers: int | None = None,
    samples_per_writer: int = 300,
    download: bool = True,
) -> tuple[FEMNISTWriterDataset, FEMNISTWriterDataset, list[str]]:
    """获取 FEMNIST train/test 数据集 + writer_id 列表.

    返回: (train_ds, test_ds, writer_ids)
        - train_ds/test_ds: FEMNISTWriterDataset (torchvision 兼容)
        - writer_ids: 排序后的 writer_id 列表，长度 = max_writers 或全部

    使用示例
    --------
    >>> train_ds, test_ds, writers = get_femnist_split("./data", max_writers=50)
    >>> len(writers), len(train_ds), len(test_ds)
    (50, ~15000, ~3750)
    """
    transform = get_default_transform()
    train_ds = FEMNISTWriterDataset(
        data_dir=data_dir, train=True, download=download,
        transform=transform, max_writers=max_writers,
        samples_per_writer=samples_per_writer,
    )
    test_ds = FEMNISTWriterDataset(
        data_dir=data_dir, train=False, download=download,
        transform=transform, max_writers=max_writers,
        samples_per_writer=None,  # test 不截断
    )
    writer_ids = sorted(train_ds.writer_to_indices.keys())
    return train_ds, test_ds, writer_ids
