"""Local image dataset import contract and archive safety checks."""

from io import BytesIO
from hashlib import md5
from pathlib import Path
from tempfile import SpooledTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from torchvision.datasets import ImageFolder

from web.dataset_store import import_imagefolder, list_datasets
from web import server
from web import public_dataset


image_buffer = BytesIO()
Image.new("RGB", (2, 2), (25, 80, 160)).save(image_buffer, format="PNG")
PNG = image_buffer.getvalue()


def make_zip(files: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def valid_files() -> dict[str, bytes]:
    return {
        "train/urban/a.png": PNG,
        "train/forest/b.png": PNG,
        "test/urban/c.png": PNG,
        "test/forest/d.png": PNG,
    }


def test_imported_dataset_is_listed_and_trainable_layout(tmp_path: Path):
    root = tmp_path / "imported"
    imported = import_imagefolder(root, "Satellite scenes", make_zip(valid_files()))
    assert imported["classes"] == ["forest", "urban"]
    assert imported["train_images"] == 2
    assert imported["test_images"] == 2
    assert (Path(imported["data_dir"]) / "train" / "urban" / "a.png").is_file()
    train = ImageFolder(Path(imported["data_dir"]) / "train")
    test = ImageFolder(Path(imported["data_dir"]) / "test")
    assert train.classes == test.classes == ["forest", "urban"]
    assert train[0][0].size == (2, 2)
    assert list_datasets(root) == [imported]


def test_import_accepts_spooled_upload(tmp_path: Path):
    with SpooledTemporaryFile(max_size=10) as upload:
        upload.write(make_zip(valid_files()))
        upload.seek(0)
        imported = import_imagefolder(tmp_path / "imported", "Buffered", upload)
    assert imported["train_images"] == 2


@pytest.mark.parametrize("bad_name", [
    "../outside.png", "train/urban/../../outside.png", "train/urban/a.txt", "train/./urban/a.png",
    "train/urban/NUL.png", "train/urban /a.png", "train/urban/A.png", "train/urban/a?.png",
])
def test_import_rejects_unexpected_members(tmp_path: Path, bad_name: str):
    files = valid_files()
    files[bad_name] = PNG
    with pytest.raises(ValueError):
        import_imagefolder(tmp_path / "imported", "Bad", make_zip(files))
    assert list_datasets(tmp_path / "imported") == []


def test_import_rejects_mismatched_classes_and_false_images(tmp_path: Path):
    files = valid_files()
    del files["test/forest/d.png"]
    with pytest.raises(ValueError, match="相同"):
        import_imagefolder(tmp_path / "imported", "Bad", make_zip(files))
    files = valid_files()
    files["test/forest/d.png"] = b"not an image"
    with pytest.raises(ValueError, match="图片内容"):
        import_imagefolder(tmp_path / "imported", "Bad", make_zip(files))
    files["test/forest/d.png"] = PNG[:25]
    with pytest.raises(ValueError, match="图片无法解码"):
        import_imagefolder(tmp_path / "imported", "Bad", make_zip(files))
    assert list_datasets(tmp_path / "imported") == []


def test_dataset_http_import_and_public_demo_guard(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(server, "IMPORTED_DATASET_DIR", tmp_path / "imported")
    monkeypatch.setattr(server, "PUBLIC_DEMO", False)
    with TestClient(server.create_app(), base_url="http://127.0.0.1:8721", client=("127.0.0.1", 50000)) as client:
        imported = client.post(
            "/api/datasets/import?name=EuroSAT%20sample",
            content=make_zip(valid_files()),
            headers={"content-type": "application/zip"},
        )
        assert imported.status_code == 200
        catalog = client.get("/api/datasets").json()
        assert catalog["imported"][0]["id"] == imported.json()["id"]
        assert len(catalog["sources"]) == 4
        assert [item["id"] for item in catalog["sources"]] == ["eurosat_rgb", "bigearthnet", "spacenet", "satnogs"]
        assert [item["importable"] for item in catalog["sources"]] == [True, False, False, False]
        assert client.post(
            "/api/datasets/import?name=blocked",
            content=make_zip(valid_files()),
            headers={"content-type": "application/zip", "Origin": "https://other.example"},
        ).status_code == 403

    monkeypatch.setattr(server, "PUBLIC_DEMO", True)
    with TestClient(server.create_app()) as client:
        assert client.post("/api/datasets/import?name=blocked", content=b"ZIP").status_code == 403
        assert client.post("/api/datasets/public/eurosat_rgb").status_code == 403
        assert client.get("/api/datasets").json()["imported"] == []


def test_server_public_eurosat_import_is_idempotent(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(server, "IMPORTED_DATASET_DIR", tmp_path / "imported")
    monkeypatch.setattr(server, "PUBLIC_DEMO", False)
    calls = []

    def fake_import(root):
        calls.append(root)
        return import_imagefolder(root, "EuroSAT RGB", make_zip(valid_files()))

    monkeypatch.setattr(server, "import_eurosat_rgb", fake_import)
    with TestClient(server.create_app(), base_url="http://127.0.0.1:8721", client=("127.0.0.1", 50000)) as client:
        first = client.post("/api/datasets/public/eurosat_rgb")
        second = client.post("/api/datasets/public/eurosat_rgb")
        assert first.status_code == second.status_code == 200
        assert first.json()["id"] == second.json()["id"]
        assert calls == [tmp_path / "imported"]


def test_eurosat_rgb_download_checksum_and_fixed_split(tmp_path: Path, monkeypatch):
    files = {f"EuroSAT_RGB/{category}/{index}.png": PNG
             for category in ("forest", "urban") for index in range(5)}
    archive = make_zip(files)
    monkeypatch.setattr(public_dataset, "EUROSAT_RGB_MD5", md5(archive).hexdigest())
    monkeypatch.setattr(public_dataset, "EUROSAT_CLASSES", 2)
    monkeypatch.setattr(public_dataset, "EUROSAT_IMAGES", 10)
    imported = public_dataset.import_eurosat_rgb(tmp_path / "datasets", source=BytesIO(archive))
    assert imported["train_images"] == 8
    assert imported["test_images"] == 2
    assert imported["provenance"]["seed"] == 42
    assert list_datasets(tmp_path / "datasets")[0]["provenance"]["record"] == "Zenodo 7711810"
    with pytest.raises(ValueError, match="校验失败"):
        public_dataset.import_eurosat_rgb(tmp_path / "datasets", source=BytesIO(archive + b"x"))
