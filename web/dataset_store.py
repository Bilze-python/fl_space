"""Validated ImageFolder ZIP imports for local SpaceFL experiments."""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import BinaryIO
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from PIL import Image, UnidentifiedImageError

MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024
MAX_FILES = 50_000
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DATASET_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
WINDOWS_RESERVED_NAMES = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{index}" for prefix in ("com", "lpt") for index in range(1, 10)
}


def _portable_component(part: str) -> bool:
    """Avoid Windows device names and aliases so archives behave consistently."""
    return (
        bool(part) and part not in {".", ".."} and part[-1] not in {".", " "}
        and not any(ord(char) < 32 or char in '\\:<>"|?*' for char in part)
        and part.split(".", 1)[0].rstrip(" .").casefold() not in WINDOWS_RESERVED_NAMES
    )


def list_datasets(root: Path) -> list[dict]:
    datasets = []
    if not root.is_dir():
        return datasets
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not DATASET_ID.fullmatch(folder.name):
            continue
        try:
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            if (folder / "train").is_dir() and (folder / "test").is_dir():
                datasets.append({**metadata, "data_dir": str(folder.resolve())})
        except (OSError, ValueError, TypeError):
            continue
    return datasets


def import_imagefolder(root: Path, name: str, archive: bytes | BinaryIO) -> dict:
    name = name.strip()
    if isinstance(archive, bytes):
        archive_size = len(archive)
        source = BytesIO(archive)
    else:
        archive.seek(0, 2)
        archive_size = archive.tell()
        archive.seek(0)
        source = archive
    if not 1 <= len(name) <= 100 or not 0 < archive_size <= MAX_ARCHIVE_BYTES:
        raise ValueError("数据集名称或 ZIP 大小无效（上限 200 MB）")
    try:
        with ZipFile(source) as zf:
            entries = [item for item in zf.infolist() if not item.is_dir()]
            if not entries or len(entries) > MAX_FILES:
                raise ValueError("ZIP 图片数量必须为 1 到 50000")
            if sum(item.file_size for item in entries) > MAX_EXTRACTED_BYTES:
                raise ValueError("解压后数据不能超过 1 GB")
            counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            seen: set[str] = set()
            for item in entries:
                path = PurePosixPath(item.filename)
                parts = path.parts
                if (item.filename != path.as_posix() or len(parts) != 3
                        or parts[0] not in {"train", "test"}
                        or any(not _portable_component(part) for part in parts)
                        or not parts[1].strip() or not parts[2].strip()
                        or path.suffix.lower() not in IMAGE_SUFFIXES
                        or item.filename.casefold() in seen or item.flag_bits & 0x1
                        or (item.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError("ZIP 只能包含 train/类别/图片 和 test/类别/图片 的 JPG/PNG 文件")
                seen.add(item.filename.casefold())
                counts[parts[0]][parts[1]] += 1
            classes = set(counts["train"])
            if (len(classes) < 2 or classes != set(counts["test"])
                    or len({category.casefold() for category in classes}) != len(classes)):
                raise ValueError("训练和测试目录必须包含相同的至少两个类别")

            root.mkdir(parents=True, exist_ok=True)
            dataset_id = uuid4().hex[:12]
            staging = root / f".staging-{dataset_id}"
            target = root / dataset_id
            staging.mkdir()
            try:
                for item in entries:
                    destination = staging.joinpath(*PurePosixPath(item.filename).parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(item) as source, destination.open("wb") as output:
                        signature = source.read(8)
                        if item.filename.lower().endswith(".png"):
                            valid = signature == b"\x89PNG\r\n\x1a\n"
                        else:
                            valid = signature.startswith(b"\xff\xd8\xff")
                        if not valid:
                            raise ValueError(f"图片内容与扩展名不符: {item.filename}")
                        output.write(signature)
                        shutil.copyfileobj(source, output)
                    try:
                        with Image.open(destination) as image:
                            width, height = image.size
                            if width * height > 32_000_000 or min(width, height) < 1:
                                raise ValueError(f"图片尺寸超出限制: {item.filename}")
                            image.verify()
                    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                        raise ValueError(f"图片无法解码: {item.filename}") from exc
                metadata = {
                    "id": dataset_id,
                    "name": name,
                    "classes": sorted(classes),
                    "train_images": sum(counts["train"].values()),
                    "test_images": sum(counts["test"].values()),
                }
                (staging / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                staging.rename(target)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            return {**metadata, "data_dir": str(target.resolve())}
    except BadZipFile as exc:
        raise ValueError("文件不是有效 ZIP 压缩包") from exc
