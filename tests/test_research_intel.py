"""intel 模块单元测试（全部离线：抓取入口与下载入口都被 monkeypatch）。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from fl_space.research import intel

ATOM_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>arXiv Query: satellite federated learning</title>
  <entry>
    <id>http://arxiv.org/abs/2511.14889v1</id>
    <updated>2026-11-01T10:00:00Z</updated>
    <published>2026-11-01T10:00:00Z</published>
    <title>Bringing Federated Learning to Space</title>
    <summary>We study &lt;b&gt;federated learning&lt;/b&gt; for LEO satellite
      constellations with intermittent ground contacts.</summary>
    <author><name>Grace Ra Kim</name></author>
    <author><name>Filip Svoboda</name></author>
    <link href="http://arxiv.org/abs/2511.14889v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2511.14889v1" rel="related"
          type="application/pdf"/>
    <arxiv:primary_category term="cs.NI"/>
    <category term="cs.NI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2511.14890v2</id>
    <published>2026-10-20T08:30:00Z</published>
    <title>Satellite Scheduling for Federated Aggregation</title>
    <summary>Ground station scheduling for satellite federated learning.</summary>
    <author><name>Alice Zhang</name></author>
    <link href="http://arxiv.org/abs/2511.14890v2" rel="alternate"/>
    <category term="eess.SP"/>
  </entry>
</feed>
"""

RSS2_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>SpaceNews</title>
    <item>
      <title>Satellite operators test federated learning in orbit</title>
      <link>https://spacenews.com/federated-learning-in-orbit/</link>
      <description><![CDATA[<p>Operators run <b>federated learning</b> payloads on
        LEO satellites.</p>]]></description>
      <pubDate>Mon, 02 Nov 2026 08:00:00 +0000</pubDate>
      <category>satellite</category>
    </item>
    <item>
      <title>Constellation edge computing moves to orbit</title>
      <link>https://spacenews.com/edge-computing-orbit/</link>
      <description>Edge computing on satellites matures.</description>
      <pubDate>Sun, 01 Nov 2026 12:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""

RDF_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns="http://purl.org/rss/1.0/"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel rdf:about="https://www.nasa.gov/rss/dyn/breaking_news.rss">
    <title>NASA Breaking News</title>
  </channel>
  <item rdf:about="https://www.nasa.gov/press-release/leo-demo">
    <title>NASA demonstrates onboard computing payload</title>
    <link>https://www.nasa.gov/press-release/leo-demo</link>
    <description>An onboard computing payload was demonstrated in low earth orbit.</description>
    <dc:date>2026-10-30T15:00:00Z</dc:date>
    <dc:creator>NASA Newsroom</dc:creator>
    <category>onboard computing</category>
  </item>
</rdf:RDF>
"""


def _build_pdf(lines: list[str]) -> bytes:
    """构造一个最小的单页 PDF（Helvetica 文本），用于离线测试 PDF 归档。"""
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


class _FakeNetwork:
    """按 URL 分发静态响应，并记录调用次数。"""

    def __init__(self, *, fail_arxiv_rss: bool = True) -> None:
        """记录调用列表；可选择让 arXiv cs.NI 源失败。"""
        self.calls: list[str] = []
        self.fail_arxiv_rss = fail_arxiv_rss

    def fetch_url(self, url: str) -> str:
        """按 URL 返回静态 Atom/RSS 文本，模拟一次 HTTP 抓取。"""
        self.calls.append(url)
        if "export.arxiv.org" in url:
            return ATOM_SAMPLE
        if "spacenews.com" in url:
            return RSS2_SAMPLE
        if "nasa.gov" in url:
            return RDF_SAMPLE
        if self.fail_arxiv_rss:
            raise OSError("network unreachable")
        return ATOM_SAMPLE


def _make_feed(tmp_path: Path, network: _FakeNetwork, **kwargs: Any) -> intel.IntelFeed:
    """构造只依赖静态响应的 IntelFeed（_fetch_url 被测试替身替换）。"""
    defaults: dict[str, Any] = {
        "timeout": 3.0,
        "refresh_interval_sec": 1800,
        "max_items": 12,
        "arxiv_queries": ["satellite federated learning"],
        "rss_feeds": [
            ("SpaceNews", "https://spacenews.com/feed/"),
            ("arXiv cs.NI", "https://rss.arxiv.org/rss/cs.NI"),
        ],
    }
    defaults.update(kwargs)
    feed = intel.IntelFeed(tmp_path / "intel.json", **defaults)
    feed._fetch_url = network.fetch_url
    return feed


# ------------------------------------------------------------------ 解析器


def test_parse_atom_fields() -> None:
    """Atom 解析出完整条目字段，arXiv id / pdf 链接 / 标签正确。"""
    items = intel.parse_atom(ATOM_SAMPLE, "arXiv")
    assert len(items) == 2
    first = items[0]
    assert set(first) == set(intel.ITEM_FIELDS)
    assert first["id"] == "arxiv_2511.14889"
    assert first["kind"] == "paper"
    assert first["title"] == "Bringing Federated Learning to Space"
    assert first["url"] == "http://arxiv.org/abs/2511.14889v1"
    assert first["pdf_url"] == "http://arxiv.org/pdf/2511.14889v1"
    assert first["authors"] == ["Grace Ra Kim", "Filip Svoboda"]
    assert first["published"] == "2026-11-01T10:00:00+00:00"
    assert "<b>" not in first["summary"] and "federated learning" in first["summary"]
    assert "cs.NI" in first["tags"] and "arxiv" in first["tags"]
    assert first["ingested"] is False
    second = items[1]
    assert second["id"] == "arxiv_2511.14890"
    assert second["pdf_url"] == "https://arxiv.org/pdf/2511.14890"


def test_parse_rss_variants() -> None:
    """RSS 2.0 与 RSS 1.0（RDF）都能解析；id 为 rss_<sha1>[:8]。"""
    items = intel.parse_rss(RSS2_SAMPLE, "SpaceNews")
    assert len(items) == 2
    first = items[0]
    assert set(first) == set(intel.ITEM_FIELDS)
    assert first["kind"] == "news"
    assert first["id"] == f"rss_{hashlib.sha1(first['url'].encode()).hexdigest()[:8]}"
    assert first["published"] == "2026-11-02T08:00:00+00:00"
    assert "<p>" not in first["summary"]
    assert "satellite" in first["tags"]

    rdf_items = intel.parse_rss(RDF_SAMPLE, "NASA")
    assert len(rdf_items) == 1
    rdf_item = rdf_items[0]
    assert rdf_item["title"] == "NASA demonstrates onboard computing payload"
    assert rdf_item["authors"] == ["NASA Newsroom"]
    assert rdf_item["published"] == "2026-10-30T15:00:00+00:00"
    assert rdf_item["url"] == "https://www.nasa.gov/press-release/leo-demo"


def test_parse_feed_sniffing_and_invalid_xml() -> None:
    """parse_feed 自动识别 Atom/RSS/RDF；非法 XML 一律返回空列表。"""
    assert len(intel.parse_feed(ATOM_SAMPLE, "arXiv")) == 2
    assert len(intel.parse_feed(RSS2_SAMPLE, "SpaceNews")) == 2
    assert len(intel.parse_feed(RDF_SAMPLE, "NASA")) == 1
    assert intel.parse_atom("<not-xml", "arXiv") == []
    assert intel.parse_rss("", "SpaceNews") == []
    assert intel.parse_feed("   ", "x") == []
    assert intel.parse_atom("<feed xmlns='http://www.w3.org/2005/Atom'><entry/></feed>", "x") == []


def test_safe_filename() -> None:
    """非法字符被清理、长度受限、空名回退。"""
    cleaned = intel.safe_filename('a/b:c*d?"e<f>g|h.pdf')
    assert not set(cleaned) & set('<>:"/\\|?*')
    long_name = intel.safe_filename("联邦学习" * 60 + ".pdf")
    assert len(long_name) <= 80
    assert long_name.endswith(".pdf")
    assert intel.safe_filename("   ") == "untitled"
    assert intel.safe_filename("卫星联邦学习-综述") == "卫星联邦学习-综述"


def test_topic_keyword_mapping() -> None:
    """内置 CN→EN 关键词映射可用，且能匹配到英文文本。"""
    keywords = intel.topic_keywords("太空联邦学习")
    assert "federated learning" in keywords
    assert "satellite" in intel.topic_keywords("卫星")
    assert "offloading" in intel.topic_keywords("卸载")
    assert "scheduling" in intel.topic_keywords("通信调度")
    assert "federated learning" in intel.keywords_in_text("A study on Federated Learning")
    assert "federated learning" not in intel.keywords_in_text("federated")


# -------------------------------------------------------------------- 抓取


def test_fetch_aggregates_dedupes_and_caches(tmp_path: Path) -> None:
    """抓取聚合多个来源、去重、按时间倒序，并写缓存；TTL 内不重复拉取。"""
    network = _FakeNetwork()
    feed = _make_feed(tmp_path, network)
    payload = feed.fetch(topics=["卫星", "联邦学习"])

    assert set(payload) == {"updated_at", "refresh_interval_sec", "sources", "items"}
    assert payload["refresh_interval_sec"] == 1800
    names = [source["name"] for source in payload["sources"]]
    assert names == ["arXiv: satellite federated learning", "SpaceNews", "arXiv cs.NI"]
    statuses = {source["name"]: source["status"] for source in payload["sources"]}
    assert statuses["SpaceNews"] == "ok"
    assert statuses["arXiv cs.NI"] == "error"
    counts = {source["name"]: source["count"] for source in payload["sources"]}
    assert counts["arXiv: satellite federated learning"] == 2
    assert counts["arXiv cs.NI"] == 0
    assert any(source["error"] for source in payload["sources"] if source["status"] == "error")

    items = payload["items"]
    assert len(items) == 4
    assert len({item["id"] for item in items}) == 4
    assert [item["published"] for item in items] == sorted(
        (item["published"] for item in items), reverse=True
    )
    atom_item = next(item for item in items if item["id"] == "arxiv_2511.14889")
    assert "federated learning" in atom_item["tags"]
    assert "satellite" in atom_item["tags"]
    assert all(item["ingested"] is False for item in items)

    assert (tmp_path / "intel.json").is_file()
    cached = feed.cached()
    assert cached is not None
    assert len(cached["items"]) == 4
    assert feed.cached_is_fresh() is True

    calls_before = len(network.calls)
    again = feed.fetch(topics=["卫星", "联邦学习"])
    assert len(network.calls) == calls_before
    assert again["updated_at"] == payload["updated_at"]

    feed.fetch(force=True)
    assert len(network.calls) > calls_before


def test_fetch_dedupes_same_entry_across_queries(tmp_path: Path) -> None:
    """同一篇论文被多个查询命中时只保留一条。"""
    network = _FakeNetwork()
    feed = _make_feed(
        tmp_path,
        network,
        arxiv_queries=["satellite federated learning", "leo satellite federated learning"],
        rss_feeds=[],
    )
    payload = feed.fetch()
    assert len(payload["items"]) == 2
    assert [source["count"] for source in payload["sources"]] == [2, 2]


def test_fetch_ttl_expiry_refetches(tmp_path: Path) -> None:
    """refresh_interval_sec=0 时缓存立即过期，第二次调用重新抓取。"""
    network = _FakeNetwork()
    feed = _make_feed(tmp_path, network, refresh_interval_sec=0)
    feed.fetch()
    calls_before = len(network.calls)
    feed.fetch()
    assert len(network.calls) > calls_before
    assert feed.cached_is_fresh() is False


def test_fetch_all_sources_failing_degrades(tmp_path: Path) -> None:
    """全部来源失败时返回空条目与错误来源，不抛异常。"""

    def broken(url: str) -> str:
        """模拟所有来源都抓取失败。"""
        raise OSError(f"boom: {url}")

    feed = intel.IntelFeed(
        tmp_path / "intel.json",
        arxiv_queries=["satellite"],
        rss_feeds=[("SpaceNews", "https://spacenews.com/feed/")],
    )
    feed._fetch_url = broken
    payload = feed.fetch()
    assert payload["items"] == []
    assert [source["status"] for source in payload["sources"]] == ["error", "error"]
    assert all(source["error"] for source in payload["sources"])


def test_cached_without_file_returns_none(tmp_path: Path) -> None:
    """没有缓存文件时 cached() 返回 None，不抛异常。"""
    feed = intel.IntelFeed(tmp_path / "missing.json")
    assert feed.cached() is None
    assert feed.cached_is_fresh() is False


# -------------------------------------------------------------------- 入库


def test_ingest_non_arxiv_item_writes_markdown(tmp_path: Path) -> None:
    """非 arXiv 条目：写入 YAML 头 + 摘要的 Markdown，并标记已入库。"""
    network = _FakeNetwork()
    feed = _make_feed(tmp_path, network, arxiv_queries=[])
    payload = feed.fetch(topics=["卫星"])
    item = next(entry for entry in payload["items"] if entry["kind"] == "news")
    target_dir = tmp_path / "文献"
    result = feed.ingest(item["id"], target_dir, tmp_path / "extract")

    assert result["ingested"] is True
    assert result["title"] == item["title"]
    path = Path(result["path"])
    assert path.is_file() and result["name"] == path.name
    assert path.parent == target_dir
    content = path.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert 'title: "' in content
    assert 'source: "SpaceNews"' in content
    assert f'url: "{item["url"]}"' in content
    assert "tags: [" in content
    assert "## 摘要" in content
    assert "## 正文" not in content
    cached = feed.cached() or {}
    stored = next(entry for entry in cached["items"] if entry["id"] == item["id"])
    assert stored["ingested"] is True


def test_ingest_arxiv_download_failure_keeps_metadata(tmp_path: Path) -> None:
    """arXiv 条目下载失败时只归档元数据与摘要，不抛异常。"""

    def broken_download(url: str) -> bytes:
        """模拟 PDF 下载失败。"""
        raise OSError(f"timeout: {url}")

    network = _FakeNetwork()
    feed = _make_feed(tmp_path, network, rss_feeds=[])
    payload = feed.fetch()
    item = next(entry for entry in payload["items"] if entry["kind"] == "paper")
    feed._fetch_bytes = broken_download
    result = feed.ingest(item["id"], tmp_path / "文献", tmp_path / "extract")
    content = Path(result["path"]).read_text(encoding="utf-8")
    assert "PDF 下载失败" in content
    assert "## 正文" not in content


def test_ingest_arxiv_pdf_extracts_body(tmp_path: Path) -> None:
    """arXiv 条目下载到 PDF 时抽取正文写入 Markdown。"""
    pytest.importorskip("pypdf")
    network = _FakeNetwork()
    feed = _make_feed(tmp_path, network, rss_feeds=[])
    payload = feed.fetch()
    item = next(entry for entry in payload["items"] if entry["kind"] == "paper")
    pdf_bytes = _build_pdf(
        [
            "Bringing Federated Learning to Space",
            "Grace Ra Kim",
            "We study federated learning for LEO satellites.",
        ]
    )
    feed._fetch_bytes = lambda _url: pdf_bytes
    result = feed.ingest(item["id"], tmp_path / "文献", tmp_path / "extract")
    content = Path(result["path"]).read_text(encoding="utf-8")
    assert "## 正文" in content
    assert "federated learning for LEO satellites" in content
    assert not list((tmp_path / "extract").glob("*.pdf"))


def test_ingest_unknown_item_raises(tmp_path: Path) -> None:
    """id 不存在时抛出 ValueError。"""
    feed = intel.IntelFeed(tmp_path / "intel.json", arxiv_queries=[], rss_feeds=[])
    with pytest.raises(ValueError):
        feed.ingest("rss_missing", tmp_path / "文献", tmp_path / "extract")
