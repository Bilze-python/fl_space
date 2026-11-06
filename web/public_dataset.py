"""Import a pinned public classification release into the managed dataset store."""

from __future__ import annotations

from hashlib import md5
import json
from pathlib import Path, PurePosixPath
import random
import time
from tempfile import SpooledTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

from web.dataset_store import MAX_ARCHIVE_BYTES, import_imagefolder

EUROSAT_RGB_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
EUROSAT_RGB_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"
EUROSAT_SPLIT_SEED = 42
EUROSAT_CLASSES = 10
EUROSAT_IMAGES = 27000


def import_eurosat_rgb(root: Path, progress=None, source=None) -> dict:
    """Download the official RGB release and make a reproducible stratified 80/20 split."""
    report = progress or (lambda message: None)
    report("正在从 Zenodo 下载 EuroSAT RGB（约 95 MB）")
    with SpooledTemporaryFile(max_size=8 * 1024 * 1024) as original:
        digest = md5()
        response = source if source is not None else urlopen(EUROSAT_RGB_URL, timeout=45)
        with response:
            size = 0
            last_report = time.monotonic()
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_ARCHIVE_BYTES:
                    raise ValueError("EuroSAT 下载文件超过 200 MB")
                digest.update(chunk)
                original.write(chunk)
                if time.monotonic() - last_report >= 15:
                    report(f"正在下载 EuroSAT RGB（已接收 {size // (1024 * 1024)} MB）")
                    last_report = time.monotonic()
        if digest.hexdigest() != EUROSAT_RGB_MD5:
            raise ValueError("EuroSAT RGB 校验失败，请稍后重试")
        original.seek(0)
        report("已校验官方文件，正在按类别生成固定 80/20 划分")
        with ZipFile(original) as archive, SpooledTemporaryFile(max_size=8 * 1024 * 1024) as prepared:
            groups: dict[str, list[str]] = {}
            for member in archive.infolist():
                if member.is_dir():
                    continue
                parts = PurePosixPath(member.filename).parts
                if (len(parts) != 3 or parts[0] != "EuroSAT_RGB"
                        or PurePosixPath(member.filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}):
                    raise ValueError("EuroSAT RGB 文件结构不符合预期")
                groups.setdefault(parts[1], []).append(member.filename)
            if len(groups) != EUROSAT_CLASSES or sum(map(len, groups.values())) != EUROSAT_IMAGES:
                raise ValueError("EuroSAT RGB 类别或图片数量不符合官方版本")
            with ZipFile(prepared, "w") as output:
                written = 0
                for category, filenames in sorted(groups.items()):
                    filenames.sort()
                    random.Random(f"{EUROSAT_SPLIT_SEED}:{category}").shuffle(filenames)
                    test_count = len(filenames) // 5
                    for index, filename in enumerate(filenames):
                        split = "test" if index < test_count else "train"
                        output.writestr(f"{split}/{category}/{PurePosixPath(filename).name}",
                                        archive.read(filename))
                        written += 1
                        if written % 3000 == 0:
                            report(f"正在划分 EuroSAT RGB（{written}/{EUROSAT_IMAGES} 张）")
            prepared.seek(0)
            result = import_imagefolder(root, "EuroSAT RGB", prepared, progress=report)
    metadata_file = Path(result["data_dir"]) / "metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    metadata["provenance"] = {"source": EUROSAT_RGB_URL, "record": "Zenodo 7711810",
                              "checksum": f"md5:{EUROSAT_RGB_MD5}", "split": "stratified 80/20",
                              "seed": EUROSAT_SPLIT_SEED, "license": "MIT"}
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report("EuroSAT RGB 已导入")
    return {**result, "provenance": metadata["provenance"]}
