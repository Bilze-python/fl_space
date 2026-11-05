"""literature 模块单元测试（全部离线，使用 tmp_path）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fl_space.research import literature

MD_TEXT = """# 卫星联邦学习综述

本文综述低地球轨道卫星星座中的联邦学习聚合与数据卸载问题，
并重点讨论通信调度对收敛速度的影响。地面站可见窗口有限，星地链路间歇性明显。

## 背景

星间链路可以显著减少聚合轮次时间，但会引入额外的能耗开销。
"""

TXT_TEXT = """地面站辅助的联邦学习调度笔记

摘要：地面站数量是训练时延的主要瓶颈，调度优化最多可带来 9 倍加速。

Ground station scheduling is the main bottleneck of satellite federated learning.

关键词：联邦学习；地面站；调度
"""

HOME_PAGE = "\n".join(
    [
        "Federated Learning for LEO Satellite Constellations",
        "1st Alice Zhang, 2nd Bob Li",
        "School of Engineering, Tsinghua University",
        "Abstract\u2014Federated learning over low earth orbit satellites is challenging",
        "because ground station contacts are intermittent and energy budgets are tight.",
        "Index Terms\u2014federated learning, LEO satellite, scheduling",
        "arXiv:2511.14889v1 [cs.NI] 6 Nov 2026",
    ]
)


def _make_index(tmp_path: Path, files: dict[str, str]) -> tuple[literature.LiteratureIndex, Path]:
    """构造临时文献根目录与索引实例。"""
    root = tmp_path / "library"
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    index = literature.LiteratureIndex(
        [("默认文献", root)],
        cache_path=tmp_path / "cache" / "index.json",
        extract_dir=tmp_path / "extract",
    )
    return index, root


def _build_pdf(lines: list[str]) -> bytes:
    """构造一个最小的单页 PDF（Helvetica 文本），用于离线测试 pypdf 抽取。"""
    content = "BT /F1 14 Tf 72 720 Td 18 TL\n" + "".join(
        f"({line}) Tj T*\n" for line in lines
    ) + "ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
    ]
    body = "%PDF-1.4\n"
    offsets: list[int] = []
    for number, item in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{number} 0 obj\n{item}\nendobj\n"
    xref_at = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for offset in offsets:
        body += f"{offset:010d} 00000 n \n"
    body += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    return body.encode("latin-1")


# --------------------------------------------------------------------- 词元化


def test_tokenize_mixed_chinese_english() -> None:
    """中英混排词元化：英文小写化、中文二元组、停用词过滤。"""
    tokens = literature.tokenize("Federated Learning 在卫星星座中的应用 the of")
    assert "federated" in tokens
    assert "learning" in tokens
    assert "卫星" in tokens
    assert "星座" in tokens
    assert "应用" in tokens
    assert "the" not in tokens
    assert "of" not in tokens


def test_tokenize_single_char_and_digits() -> None:
    """单字汉字保留，单字虚词过滤，数字保留。"""
    assert literature.tokenize("星") == ["星"]
    assert literature.tokenize("的") == []
    assert "2024" in literature.tokenize("LEO 2024 dataset")
    assert "3" in literature.tokenize("top 3")


# ------------------------------------------------------------------------ BM25


def test_bm25_ranks_higher_term_frequency_first() -> None:
    """BM25 排序：词频更高、长度相近的文档得分更高。"""
    docs = [
        ["satellite", "satellite", "satellite"],
        ["satellite", "ground", "station"],
        ["scheduling", "delay", "window"],
    ]
    engine = literature.BM25(docs)
    scores = engine.scores(["satellite"])
    assert scores[0] > scores[1] > 0.0
    assert scores[2] == 0.0


def test_bm25_unknown_query_returns_zeros() -> None:
    """未知查询词元不产生得分，也不抛异常。"""
    engine = literature.BM25([["federated", "learning"]])
    assert engine.scores(["unknown"]) == [0.0]
    assert literature.BM25([]).scores(["anything"]) == []


# ------------------------------------------------------------------ 分块边界


def test_chunk_pages_overlap_and_page_boundaries() -> None:
    """长文本按块大小切开并保留重叠，页码跟随原页。"""
    chunks = literature.chunk_pages(
        [{"page": 3, "text": "甲" * 3000}], chunk_chars=1000, overlap=100
    )
    assert len(chunks) == 4
    assert all(len(chunk["text"]) <= 1000 for chunk in chunks)
    assert all(chunk["page"] == 3 for chunk in chunks)
    assert chunks[1]["text"][:50] == chunks[0]["text"][-50:]


def test_chunk_pages_edge_cases() -> None:
    """空输入返回空列表；空文本页跳过；重叠不小于块长时仍能终止。"""
    assert literature.chunk_pages([]) == []
    assert literature.chunk_pages([{"page": 1, "text": "   "}]) == []
    chunks = literature.chunk_pages(
        [{"page": 1, "text": "x" * 1000}], chunk_chars=200, overlap=200
    )
    assert len(chunks) >= 2
    assert all(len(chunk["text"]) <= 200 for chunk in chunks)


def test_chunk_pages_prefers_sentence_boundary() -> None:
    """切分点优先落在句号等语义边界上。"""
    text = "First sentence about satellites. " * 80
    chunks = literature.chunk_pages([{"page": 1, "text": text}], chunk_chars=300, overlap=50)
    assert chunks
    assert chunks[0]["text"].endswith(".")


# ------------------------------------------------------------------ 元数据猜测


def test_guess_metadata_from_pdf_homepage() -> None:
    """从 PDF 首页文本抽取标题、作者、年份、venue、摘要与关键词。"""
    meta = literature.guess_metadata(
        [{"page": 1, "text": HOME_PAGE}], Path("2511.14889v1.pdf"), "pdf"
    )
    assert meta["title"] == "Federated Learning for LEO Satellite Constellations"
    assert meta["authors"] == ["Alice Zhang", "Bob Li"]
    assert meta["year"] == 2026
    assert meta["venue"] == "arXiv [cs.NI]"
    assert meta["abstract"].startswith("Federated learning over low earth orbit")
    assert meta["keywords"] == ["federated learning", "LEO satellite", "scheduling"]


def test_guess_metadata_markdown_and_fallback() -> None:
    """Markdown 取一级标题与首段；无可用信息时回退文件名。"""
    meta = literature.guess_metadata(
        [{"page": 1, "text": MD_TEXT}], Path("note.md"), "md"
    )
    assert meta["title"] == "卫星联邦学习综述"
    assert "联邦学习聚合与数据卸载" in meta["abstract"]
    fallback = literature.guess_metadata([], Path("[23]_DFedSat2025_分析.md"), "md")
    assert fallback["title"] == "DFedSat2025 分析"


# --------------------------------------------------------------- 文件读取/缓存


def test_read_document_text_markdown_and_txt(tmp_path: Path) -> None:
    """md/txt 读取为单页文本，标记为 markdown。"""
    md_path = tmp_path / "note.md"
    md_path.write_text(MD_TEXT, encoding="utf-8")
    pages, note = literature.read_document_text(md_path, tmp_path / "extract")
    assert note == "markdown"
    assert len(pages) == 1 and pages[0]["page"] == 1
    assert "卫星联邦学习综述" in pages[0]["text"]

    txt_path = tmp_path / "note.txt"
    txt_path.write_text(TXT_TEXT, encoding="utf-8")
    pages, note = literature.read_document_text(txt_path, tmp_path / "extract")
    assert note == "markdown"
    assert pages[0]["text"].startswith("地面站辅助的联邦学习调度笔记")


def test_read_document_text_pdf_uses_cache(tmp_path: Path) -> None:
    """PDF 首次用 pypdf 抽取并写缓存，第二次命中缓存。"""
    pytest.importorskip("pypdf")
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(_build_pdf(["Federated Learning for LEO Satellites"]))
    extract_dir = tmp_path / "extract"
    pages, note = literature.read_document_text(pdf_path, extract_dir)
    assert note == "pypdf"
    assert pages and pages[0]["page"] == 1
    assert "Federated" in pages[0]["text"]
    cache_file = extract_dir / "sample.txt"
    assert cache_file.is_file()
    assert cache_file.read_text(encoding="utf-8").startswith("===== PAGE 1 =====")
    pages_again, note_again = literature.read_document_text(pdf_path, extract_dir)
    assert note_again == "cache"
    assert pages_again == pages


def test_read_document_text_missing_file(tmp_path: Path) -> None:
    """不存在或类型不支持时返回空列表与 missing，不抛异常。"""
    pages, note = literature.read_document_text(tmp_path / "nope.md", tmp_path / "extract")
    assert pages == [] and note == "missing"
    other = tmp_path / "note.bin"
    other.write_bytes(b"\x00\x01")
    pages, note = literature.read_document_text(other, tmp_path / "extract")
    assert pages == [] and note == "missing"


# ----------------------------------------------------------------- 索引与检索


def test_rebuild_items_get_stats(tmp_path: Path) -> None:
    """重建后能取到条目、单篇详情与统计。"""
    index, _root = _make_index(tmp_path, {"综述.md": MD_TEXT, "笔记.txt": TXT_TEXT})
    report = index.rebuild()
    assert report["indexed"] == 2
    assert report["parsed"] == 2
    assert report["reused"] == 0
    assert report["errors"] == []

    items = index.items()
    assert len(items) == 2
    doc = next(item for item in items if item.doc_type == "md")
    assert doc.title == "卫星联邦学习综述"
    assert doc.source == "默认文献"
    assert doc.path.startswith("0:")
    assert doc.id == literature.make_doc_id(doc.path)
    assert doc.pages == 1 and doc.chunks >= 1
    assert doc.extracted is True and doc.extract_note == "markdown"
    assert doc.modified_at.endswith("+00:00")
    public = doc.to_public()
    assert "abs_path" not in public
    assert public["type"] == "md"
    assert public.get("tags")

    assert index.get(doc.id) is not None
    assert index.get("lib_missing") is None

    stats = index.stats()
    assert stats["total"] == 2
    assert stats["md"] == 1 and stats["txt"] == 1
    assert stats["parsed"] == 2
    assert stats["chars"] > 100
    assert stats["roots"][0]["exists"] is True


def test_search_and_context(tmp_path: Path) -> None:
    """中英文检索均能命中，上下文与引用编号一致且不超长。"""
    index, _root = _make_index(tmp_path, {"综述.md": MD_TEXT, "笔记.txt": TXT_TEXT})
    index.rebuild()

    hits = index.search("数据卸载 通信调度")
    assert hits
    assert hits[0]["title"] == "卫星联邦学习综述"
    assert hits[0]["score"] > 0
    assert hits[0]["snippets"]
    assert hits[0]["snippets"][0]["page"] == 1
    assert "数据卸载" in "".join(item["text"] for item in hits[0]["snippets"])

    english = index.search("scheduling ground station")
    assert english and english[0]["title"] == "地面站辅助的联邦学习调度笔记"

    assert index.search("") == []
    assert index.search("zzzz-unknown-term") == []

    context_text, citations = index.context("联邦学习", max_chars=400, limit=5)
    assert len(context_text) <= 400
    assert citations
    assert context_text.startswith("[1] 《")
    assert "[1] " in context_text and " p.1:" in context_text
    assert citations[0]["index"] == 1
    assert citations[0]["page"] == 1
    assert set(citations[0]) == {"index", "doc_id", "title", "page", "quote"}
    assert index.get(citations[0]["doc_id"]) is not None


def test_search_title_boost(tmp_path: Path) -> None:
    """标题命中查询词的文档获得加权，排在仅正文命中的文档之前。"""
    index, _root = _make_index(
        tmp_path,
        {
            "联邦学习综述.md": "# 联邦学习综述\n\n本文讨论星地链路的训练流程。\n",
            "工程笔记.md": "# 工程笔记\n\n联邦学习 联邦学习 联邦学习 联邦学习 联邦学习 讨论。\n",
        },
    )
    index.rebuild()
    hits = index.search("联邦学习")
    assert next(item["title"] for item in hits) == "联邦学习综述"


def test_pages_returns_text_for_known_doc(tmp_path: Path) -> None:
    """pages 返回逐页文本；未知 id 返回空列表。"""
    index, _root = _make_index(tmp_path, {"综述.md": MD_TEXT})
    index.rebuild()
    doc = index.items()[0]
    pages = index.pages(doc.id)
    assert pages and pages[0]["page"] == 1
    assert "地面站可见窗口" in pages[0]["text"]
    assert index.pages("lib_unknown") == []


# --------------------------------------------------------------- 增量与容错


def test_rebuild_incremental_reuse(tmp_path: Path) -> None:
    """未修改文件走复用，修改后重新解析。"""
    index, root = _make_index(tmp_path, {"综述.md": MD_TEXT, "笔记.txt": TXT_TEXT})
    first = index.rebuild()
    assert first["parsed"] == 2

    second = index.rebuild()
    assert second["parsed"] == 0
    assert second["reused"] == 2
    assert second["indexed"] == 2

    (root / "综述.md").write_text(f"{MD_TEXT}\n新增段落：星间链路可以减少聚合轮次时间。\n", encoding="utf-8")
    third = index.rebuild()
    assert third["parsed"] == 1
    assert third["reused"] == 1

    forced = index.rebuild(force=True)
    assert forced["parsed"] == 2 and forced["reused"] == 0


def test_rebuild_broken_pdf_and_missing_root(tmp_path: Path) -> None:
    """损坏 PDF 与不存在的根目录都不崩溃，错误被记录。"""
    index, root = _make_index(tmp_path, {"正常.md": MD_TEXT})
    (root / "broken.pdf").write_bytes(b"this is not a real pdf file")
    report = index.rebuild()
    assert report["indexed"] == 2
    assert any("broken.pdf" in item["path"] for item in report["errors"])
    broken = next(
        doc for doc in index.items() if doc.path.endswith("broken.pdf")
    )
    assert broken.doc_type == "pdf"
    assert broken.extracted is False
    assert broken.extract_note == "missing"
    assert broken.title

    missing = literature.LiteratureIndex(
        [("缺失", tmp_path / "nope")], tmp_path / "cache2" / "index.json", tmp_path / "extract2"
    )
    empty = missing.rebuild()
    assert empty["indexed"] == 0 and empty["errors"] == []
    assert missing.stats()["roots"][0]["exists"] is False


def test_cache_roundtrip_and_corrupt_cache(tmp_path: Path) -> None:
    """save/load 可还原索引；损坏的缓存文件被忽略。"""
    index, _root = _make_index(tmp_path, {"综述.md": MD_TEXT, "笔记.txt": TXT_TEXT})
    index.rebuild()
    cache_path = tmp_path / "cache" / "index.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["version"] == literature.INDEX_VERSION
    assert len(payload["docs"]) == 2
    assert "text" not in payload["docs"][0]
    assert payload["docs"][0]["chunks"] >= 1

    restored = literature.LiteratureIndex(
        [("默认文献", tmp_path / "library")], cache_path, tmp_path / "extract"
    )
    restored.load()
    assert len(restored.items()) == 2
    assert {doc.title for doc in restored.items()} == {
        "卫星联邦学习综述",
        "地面站辅助的联邦学习调度笔记",
    }
    assert restored.search("数据卸载")[0]["title"] == "卫星联邦学习综述"
    assert restored.stats()["parsed"] == 2

    cache_path.write_text("{not json", encoding="utf-8")
    broken = literature.LiteratureIndex(
        [("默认文献", tmp_path / "library")], cache_path, tmp_path / "extract"
    )
    broken.load()
    assert broken.items() == []


def test_extract_cache_directory_not_indexed(tmp_path: Path) -> None:
    """根目录下的 extracted 缓存目录不会被当成文献重复索引。"""
    index, root = _make_index(tmp_path, {"综述.md": MD_TEXT})
    (root / "extracted").mkdir()
    (root / "extracted" / "stale.txt").write_text("缓存内容", encoding="utf-8")
    report = index.rebuild()
    assert report["indexed"] == 1
    assert index.items()[0].path == "0:综述.md"


def test_threaded_search_is_stable(tmp_path: Path) -> None:
    """并发检索不破坏索引状态（RLock 保护）。"""
    import threading

    index, _root = _make_index(tmp_path, {"综述.md": MD_TEXT, "笔记.txt": TXT_TEXT})
    index.rebuild()
    results: list[int] = []

    def worker() -> None:
        """并发检索一次，记录命中条数（异常记为 -1）。"""
        try:
            results.append(len(index.search("联邦学习")))
        except Exception:  # pragma: no cover - 并发异常即便失败也会让断言不成立
            results.append(-1)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [2] * 6
