"""Local image dataset import contract and archive safety checks."""

from io import BytesIO
from pathlib import Path
from tempfile import SpooledTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from torchvision.datasets import ImageFolder

from web.dataset_store import import_imagefolder, list_datasets
from web import server


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
        assert client.post(
            "/api/datasets/import?name=blocked",
            content=make_zip(valid_files()),
            headers={"content-type": "application/zip", "Origin": "https://other.example"},
        ).status_code == 403

    monkeypatch.setattr(server, "PUBLIC_DEMO", True)
    with TestClient(server.create_app()) as client:
        assert client.post("/api/datasets/import?name=blocked", content=b"ZIP").status_code == 403
        assert client.get("/api/datasets").json()["imported"] == []
