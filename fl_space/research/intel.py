"""前沿资讯抓取：arXiv Atom API + 多家 RSS 聚合、缓存与文献库归档。

设计原则：

- 导入本模块不产生任何网络请求；只有显式调用 :meth:`IntelFeed.fetch` 才联网；
- 所有网络请求走 ``urllib.request``，统一 UA ``SpaceFL-Research/1.0``、超时 15s；
- 任一来源失败只降级该来源（``sources[].status = "error"``），不影响整体返回；
- 解析器对非法 XML 返回空列表，绝不抛出异常。

示例::

    feed = IntelFeed(Path(".spacefl_intel_cache.json"))
    payload = feed.fetch(topics=["卫星联邦学习", "数据卸载"])
    feed.ingest(payload["items"][0]["id"], Path("文献"), Path("文献/extracted"))
"""

from __future__ import annotations

import contextlib
import copy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import html
import json
import os
from pathlib import Path
import re
import threading
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# 归档条目字段顺序（契约要求 11 个字段齐全）
ITEM_FIELDS = (
    "id",
    "source",
    "kind",
    "title",
    "url",
    "published",
    "summary",
    "authors",
    "tags",
    "pdf_url",
    "ingested",
)

# 缓存结构版本
CACHE_VERSION = 1

# 网络标识与超时
USER_AGENT = "SpaceFL-Research/1.0"
DEFAULT_TIMEOUT_SEC = 15.0

# arXiv Atom API 与单查询返回条数
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_MAX_RESULTS = 10

# 两条默认英文检索式：卫星联邦学习 / LEO FL / 星上计算 / 星座边缘智能
DEFAULT_ARXIV_QUERIES: list[str] = [
    "satellite federated learning",
    "LEO satellite federated learning",
    "onboard computing satellite",
    "constellation edge intelligence",
]

# 默认 RSS 源：(标签, 地址)
DEFAULT_RSS_FEEDS: list[tuple[str, str]] = [
    ("SpaceNews", "https://spacenews.com/feed/"),
    ("NASA", "https://www.nasa.gov/rss/dyn/breaking_news.rss"),
    ("arXiv cs.NI", "https://rss.arxiv.org/rss/cs.NI"),
]

# 内置 中文主题 → 英文关键词 映射，用于给情报条目打标签与排序加权
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "太空联邦学习": ["federated learning", "space federated learning", "satellite"],
    "联邦学习": ["federated learning"],
    "卫星": ["satellite"],
    "低轨卫星": ["leo satellite", "low earth orbit"],
    "LEO 星座": ["leo constellation", "low earth orbit"],
    "LEO星座": ["leo constellation", "low earth orbit"],
    "星座": ["constellation"],
    "数据卸载": ["offloading", "data offloading"],
    "卸载": ["offloading"],
    "通信调度": ["scheduling", "communication scheduling"],
    "边缘计算": ["edge computing"],
    "星上计算": ["onboard computing", "in-orbit computing"],
    "星间链路": ["inter-satellite link", "inter-satellite links"],
    "资源分配": ["resource allocation"],
    "通信压缩": ["communication compression", "quantization"],
    "模型聚合": ["model aggregation"],
    "空天地一体化": ["space-air-ground", "satellite-terrestrial"],
}

# 关键词匹配用的全量英文词表（去重、按长度倒序便于优先命中长词）
_ALL_EN_KEYWORDS: list[str] = sorted(
    {keyword for values in TOPIC_KEYWORDS.values() for keyword in values},
    key=len,
    reverse=True,
)

# 文件名非法字符（与 Windows 限制一致）
_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# arXiv id（新版 2511.14889 / 旧版 cs.NI/0601001）
_ARXIV_ID_RE = re.compile(
    r"(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", re.IGNORECASE
)
_TAG_RE = re.compile(r"<[^>]*>")
_RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"


def _collapse_ws(text: str) -> str:
    """把任意空白压缩为单个空格并去除首尾。"""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_markup(text: str) -> str:
    """去掉 HTML 标签、反转义实体并压缩空白。"""
    unescaped = html.unescape(str(text or ""))
    return _collapse_ws(_TAG_RE.sub(" ", unescaped))


def _truncate(text: str, limit: int) -> str:
    """按字符数截断（含省略号），保证结果长度不超过 limit。"""
    cleaned = str(text or "")
    if limit <= 0 or len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(1, limit - 1)].rstrip()}\u2026"


def _iso(moment: datetime) -> str:
    """datetime → 带时区的 UTC ISO8601 字符串。"""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def now_iso() -> str:
    """当前 UTC 时间 → ISO8601 字符串。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_iso(value: str) -> str:
    """把 Atom/RFC822 时间字符串统一成 UTC ISO8601；失败返回空串。"""
    text = _collapse_ws(value)
    if not text:
        return ""
    candidate = text
    if candidate.endswith(("Z", "z")):
        candidate = f"{candidate[:-1]}+00:00"
    with contextlib.suppress(ValueError):
        return _iso(datetime.fromisoformat(candidate))
    with contextlib.suppress(TypeError, ValueError, OverflowError):
        parsed = parsedate_to_datetime(text)
        if parsed is not None:
            return _iso(parsed)
    return ""


def _local_name(tag: str) -> str:
    """取 XML 标签的本地名（去掉命名空间）并小写。"""
    return str(tag).rsplit("}", 1)[-1].lower()


def _child_text(node: ET.Element, *names: str) -> str:
    """取第一个指定本地名子节点的文本（含嵌套文本）。"""
    wanted = {name.lower() for name in names}
    for child in node:
        if _local_name(child.tag) in wanted:
            return _collapse_ws("".join(child.itertext()))
    return ""


def _children(node: ET.Element, *names: str) -> list[ET.Element]:
    """取全部指定本地名子节点。"""
    wanted = {name.lower() for name in names}
    return [child for child in node if _local_name(child.tag) in wanted]


def _entry_nodes(xml_text: str, names: str) -> list[ET.Element]:
    """解析 XML 并按本地名取出条目节点；非法 XML 返回空列表。"""
    try:
        root = ET.fromstring(xml_text)
    except (ET.ParseError, ValueError):
        return []
    return [node for node in root.iter() if _local_name(node.tag) == names]


def _extract_arxiv_id(value: str) -> str:
    """从 abs/pdf 链接或裸 id 中提取 arXiv id（去掉版本号）。"""
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"(?:abs|pdf)/([^/?#]+)", text)
    candidate = match.group(1) if match else text
    if candidate.lower().endswith(".pdf"):
        candidate = candidate[:-4]
    if not _ARXIV_ID_RE.fullmatch(candidate):
        found = _ARXIV_ID_RE.search(text)
        if found is None:
            return ""
        candidate = found.group(0)
    return re.sub(r"v\d+$", "", candidate, flags=re.IGNORECASE)


def _make_item(
    *,
    item_id: str,
    source: str,
    kind: str,
    title: str,
    url: str,
    published: str,
    summary: str,
    authors: list[str],
    tags: list[str],
    pdf_url: str,
) -> dict[str, Any]:
    """按契约字段顺序组装资讯条目。"""
    return {
        "id": item_id,
        "source": source,
        "kind": kind if kind in {"paper", "news"} else "news",
        "title": title,
        "url": url,
        "published": published,
        "summary": summary,
        "authors": authors,
        "tags": tags,
        "pdf_url": pdf_url,
        "ingested": False,
    }


def _rss_id(*candidates: str) -> str:
    """非 arXiv 条目 id：``rss_`` + sha1(首个可用标识) 前 8 位。"""
    seed = next((str(item).strip() for item in candidates if str(item).strip()), "")
    return f"rss_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:8]}"


def parse_atom(xml_text: str, source: str) -> list[dict[str, Any]]:
    """解析 Atom feed（arXiv API / arXiv RSS / 通用 Atom）为资讯条目列表。

    非法 XML 或缺少必需字段时返回 ``[]``；arXiv 条目 id 为 ``arxiv_<id>``，
    其余 Atom 条目退化为 ``rss_<sha1>``。
    """
    items: list[dict[str, Any]] = []
    for node in _entry_nodes(xml_text, "entry"):
        raw_id = _child_text(node, "id")
        title = _clean_markup(_child_text(node, "title"))
        if not title:
            continue
        arxiv_id = _extract_arxiv_id(raw_id) or _extract_arxiv_id(_child_text(node, "link"))
        url = raw_id if raw_id.startswith("http") else ""
        pdf_url = ""
        for link in _children(node, "link"):
            href = _collapse_ws(link.get("href") or "")
            if not href:
                continue
            rel = (link.get("rel") or "").strip().lower()
            link_title = (link.get("title") or "").strip().lower()
            link_type = (link.get("type") or "").strip().lower()
            if link_title == "pdf" or link_type == "application/pdf":
                pdf_url = pdf_url or href
            elif rel in {"alternate", ""} and not url:
                url = href
        if arxiv_id:
            url = url or f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
        authors = [
            _clean_markup(_child_text(author, "name")) for author in _children(node, "author")
        ]
        authors = [name for name in authors if name]
        tags: list[str] = []
        for child in node:
            if _local_name(child.tag) not in {"category", "primary_category"}:
                continue
            term = _collapse_ws(child.get("term") or "")
            if term and term not in tags:
                tags.append(term)
        if arxiv_id and "arxiv" not in tags:
            tags.append("arxiv")
        published = _to_iso(_child_text(node, "published", "updated", "date"))
        summary = _truncate(_clean_markup(_child_text(node, "summary", "content")), 600)
        item_id = f"arxiv_{arxiv_id}" if arxiv_id else _rss_id(raw_id, url, title)
        items.append(
            _make_item(
                item_id=item_id,
                source=source,
                kind="paper" if arxiv_id else "news",
                title=title,
                url=url or pdf_url,
                published=published,
                summary=summary,
                authors=authors,
                tags=tags,
                pdf_url=pdf_url,
            )
        )
    return items


def parse_rss(xml_text: str, source: str) -> list[dict[str, Any]]:
    """解析 RSS 2.0 或 RSS 1.0（RDF）为资讯条目列表。

    非法 XML 返回 ``[]``；条目 id 统一为 ``rss_<sha1(link)>[:8]``，kind 为 ``news``。
    """
    items: list[dict[str, Any]] = []
    for node in _entry_nodes(xml_text, "item"):
        title = _clean_markup(_child_text(node, "title"))
        if not title:
            continue
        rdf_about = _collapse_ws(node.get(_RDF_ABOUT) or "")
        link = _child_text(node, "link") or rdf_about or _child_text(node, "guid")
        description = _child_text(node, "description", "encoded", "summary", "content")
        published = _to_iso(
            _child_text(node, "pubdate", "date", "published", "updated", "issued")
        )
        authors = [
            _clean_markup(_child_text(node, name))
            for name in ("creator", "author")
        ]
        authors = [author for author in authors if author]
        tags: list[str] = []
        for child in node:
            if _local_name(child.tag) not in {"category", "subject", "tag"}:
                continue
            term = _clean_markup("".join(child.itertext())) or _collapse_ws(child.get("term") or "")
            if term and term not in tags:
                tags.append(term)
        pdf_url = ""
        for enclosure in _children(node, "enclosure"):
            enclosure_url = _collapse_ws(enclosure.get("url") or "")
            enclosure_type = (enclosure.get("type") or "").lower()
            is_pdf = "pdf" in enclosure_type or enclosure_url.lower().endswith(".pdf")
            if enclosure_url and is_pdf:
                pdf_url = enclosure_url
                break
        items.append(
            _make_item(
                item_id=_rss_id(link, title),
                source=source,
                kind="news",
                title=title,
                url=link,
                published=published,
                summary=_truncate(_clean_markup(description), 600),
                authors=authors,
                tags=tags,
                pdf_url=pdf_url,
            )
        )
    return items


def parse_feed(xml_text: str, source: str) -> list[dict[str, Any]]:
    """按根节点自动识别 Atom / RSS 2.0 / RSS 1.0 并解析。"""
    if not str(xml_text or "").strip():
        return []
    head = str(xml_text)[:4000].lower()
    if "<rss" in head or "<rdf" in head:
        return parse_rss(xml_text, source)
    if "<feed" in head:
        return parse_atom(xml_text, source)
    return parse_atom(xml_text, source) or parse_rss(xml_text, source)


def safe_filename(name: str, *, limit: int = 80) -> str:
    """把任意标题清理成安全的文件名（去 Windows 非法字符、限长 80）。"""
    cleaned = _INVALID_NAME_CHARS.sub("_", str(name or ""))
    cleaned = _collapse_ws(cleaned).strip(" ._-")
    if not cleaned:
        return "untitled"
    if len(cleaned) <= limit:
        return cleaned
    suffix = ""
    match = re.search(r"(\.[A-Za-z0-9]{1,10})$", cleaned)
    if match:
        suffix = match.group(1)
        cleaned = cleaned[: -len(suffix)]
    keep = max(1, limit - len(suffix))
    return f"{cleaned[:keep].rstrip(' ._-') or 'untitled'}{suffix}"


def _contains(haystack: str, keyword: str) -> bool:
    """子串匹配；长度 ≤4 的英文短词要求词边界，避免误命中。"""
    if not haystack or not keyword:
        return False
    if keyword.isascii() and len(keyword) <= 4:
        return re.search(rf"\b{re.escape(keyword)}\b", haystack) is not None
    return keyword in haystack


def keywords_in_text(text: str) -> list[str]:
    """返回文本中出现的全部内置英文关键词（长词优先）。"""
    haystack = str(text or "").casefold()
    if not haystack:
        return []
    return [keyword for keyword in _ALL_EN_KEYWORDS if _contains(haystack, keyword)]


def topic_keywords(topic: str) -> list[str]:
    """中文主题 → 英文关键词；未收录时回退原文匹配或原样返回。"""
    cleaned = _collapse_ws(topic)
    if not cleaned:
        return []
    mapped = TOPIC_KEYWORDS.get(cleaned)
    if mapped:
        return list(mapped)
    matched = keywords_in_text(cleaned)
    if matched:
        return matched
    return [cleaned]


def _add_tags(item: dict[str, Any], keywords: list[str]) -> None:
    """把关键词并入条目 tags（大小写不敏感去重）。"""
    tags = [str(tag) for tag in item.get("tags") or [] if str(tag).strip()]
    seen = {tag.casefold() for tag in tags}
    for keyword in keywords:
        key = keyword.casefold()
        if key and key not in seen:
            seen.add(key)
            tags.append(keyword)
    item["tags"] = tags


def _apply_topics(item: dict[str, Any], topics: list[str]) -> int:
    """按主题关键词给条目打标签，返回命中的主题数量（用于排序加权）。"""
    haystack = " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("summary") or ""),
            " ".join(str(tag) for tag in item.get("tags") or []),
        ]
    ).casefold()
    hits = 0
    for topic in topics:
        matched = [
            keyword for keyword in topic_keywords(topic) if _contains(haystack, keyword.casefold())
        ]
        if matched:
            hits += 1
            _add_tags(item, matched)
    return hits


def _arxiv_api_url(query: str) -> str:
    """构造 arXiv Atom API 查询地址（多个词用 AND 连接）。"""
    terms = [term for term in re.split(r"\s+", _collapse_ws(query)) if term]
    if not terms:
        return f"{ARXIV_API_URL}?search_query=all:satellite&max_results={ARXIV_MAX_RESULTS}"
    expression = "+AND+".join(
        f"all:{urllib.parse.quote(term, safe='')}" for term in terms
    )
    return (
        f"{ARXIV_API_URL}?search_query={expression}"
        f"&start=0&max_results={ARXIV_MAX_RESULTS}"
        "&sortBy=submittedDate&sortOrder=descending"
    )


def _error_text(exc: BaseException) -> str:
    """把异常压缩成单行错误文本（最多 200 字）。"""
    reason = getattr(exc, "reason", None) if isinstance(exc, urllib.error.URLError) else None
    detail = _collapse_ws(str(reason)) if reason else _collapse_ws(str(exc))
    detail = detail or exc.__class__.__name__
    return f"{type(exc).__name__}: {detail}"[:200]


def _is_fresh(cache: dict[str, Any] | None, interval_sec: int) -> bool:
    """缓存是否在有效期内（有内容且 updated_at 距今小于间隔）。"""
    if not isinstance(cache, dict):
        return False
    items = cache.get("items")
    if not isinstance(items, list) or not items:
        return False
    stamp = str(cache.get("updated_at") or "")
    if not stamp:
        return False
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - parsed).total_seconds()
    return 0 <= age < max(0, int(interval_sec))


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """在目录内生成不冲突的文件路径。"""
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        candidate = directory / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
    return directory / f"{stem}_{os.getpid()}{suffix}"


def _yaml_text(value: str) -> str:
    """YAML 双引号字符串转义。"""
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"')


def _yaml_list(values: Any) -> str:
    """把字符串列表渲染成 YAML 行内列表。"""
    if not isinstance(values, (list, tuple)):
        return "[]"
    rendered = [f'"{_yaml_text(str(value))}"' for value in values if str(value).strip()]
    return f"[{', '.join(rendered)}]"


def _published_date(item: dict[str, Any]) -> str:
    """资讯发布日期（``YYYY-MM-DD``），无日期时用 ``undated``。"""
    stamp = str(item.get("published") or "")
    return stamp[:10] if len(stamp) >= 10 else "undated"


def _render_markdown(item: dict[str, Any], body: str, note: str) -> str:
    """渲染归档 Markdown：YAML 头 + 摘要 + 正文。"""
    title = _collapse_ws(str(item.get("title") or "未命名资讯")) or "未命名资讯"
    lines = [
        "---",
        f'title: "{_yaml_text(title)}"',
        f'source: "{_yaml_text(str(item.get("source") or ""))}"',
        f'url: "{_yaml_text(str(item.get("url") or ""))}"',
        f'published: "{_yaml_text(str(item.get("published") or ""))}"',
        f"tags: {_yaml_list(item.get('tags'))}",
        f'kind: "{_yaml_text(str(item.get("kind") or "news"))}"',
        f'ingested_at: "{now_iso()}"',
        "---",
        "",
        f"# {title}",
        "",
    ]
    if item.get("authors"):
        lines += [f"**作者**：{', '.join(str(name) for name in item['authors'])}", ""]
    lines += ["## 摘要", "", str(item.get("summary") or "（无摘要）").strip(), ""]
    if note:
        lines += [f"> {note}", ""]
    if body:
        lines += ["## 正文", "", body.strip(), ""]
    return "\n".join(lines).rstrip() + "\n"


class IntelFeed:
    """前沿资讯聚合器：arXiv 检索 + RSS 抓取 + 本地缓存 + 归档入库。"""

    def __init__(
        self,
        cache_path: Path,
        *,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        refresh_interval_sec: int = 1800,
        max_items: int = 48,
        arxiv_queries: list[str] | None = None,
        rss_feeds: list[tuple[str, str]] | None = None,
    ) -> None:
        """保存配置与缓存路径；构造时不联网、不读大文件（首次访问再读缓存）。"""
        self.cache_path = Path(cache_path)
        self.timeout = max(1.0, float(timeout))
        self.refresh_interval_sec = int(refresh_interval_sec)
        self.max_items = max(1, int(max_items))
        self.arxiv_queries = list(
            DEFAULT_ARXIV_QUERIES if arxiv_queries is None else arxiv_queries
        )
        self.rss_feeds = list(DEFAULT_RSS_FEEDS if rss_feeds is None else rss_feeds)
        self._lock = threading.RLock()
        self._cache: dict[str, Any] | None = None
        self._loaded = False

    # ------------------------------------------------------------------ 缓存

    def cached(self) -> dict[str, Any] | None:
        """返回本地缓存的聚合结果（不校验有效期），无缓存时返回 None。"""
        with self._lock:
            if not self._loaded:
                self._cache = self._read_cache()
                self._loaded = True
            if self._cache is None:
                return None
            return copy.deepcopy(self._cache)

    def cached_is_fresh(self) -> bool:
        """当前缓存是否仍在刷新间隔内。"""
        cache = self.cached()
        return _is_fresh(cache, self.refresh_interval_sec)

    def _read_cache(self) -> dict[str, Any] | None:
        """读缓存文件；缺失、损坏或版本不符时返回 None。"""
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            return None
        try:
            version = int(payload.get("version") or 0)
        except (TypeError, ValueError):
            return None
        if version != CACHE_VERSION:
            return None
        payload.pop("version", None)
        return payload

    def _write_cache(self, payload: dict[str, Any]) -> None:
        """原子写入缓存文件，失败静默（不影响调用方）。"""
        record = {**payload, "version": CACHE_VERSION}
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.cache_path.with_name(f"{self.cache_path.name}.tmp")
            temp_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            os.replace(temp_path, self.cache_path)
        except OSError:
            with contextlib.suppress(OSError):
                self.cache_path.with_name(f"{self.cache_path.name}.tmp").unlink()

    # ------------------------------------------------------------------ 抓取

    def fetch(
        self, *, topics: list[str] | None = None, force: bool = False
    ) -> dict[str, Any]:
        """抓取所有来源并返回聚合结果；缓存未过期且 ``force=False`` 时直接返回缓存。"""
        topic_list = [_collapse_ws(topic) for topic in (topics or []) if _collapse_ws(topic)]
        cache = self.cached()
        if not force and cache is not None and _is_fresh(cache, self.refresh_interval_sec):
            return cache
        sources: list[dict[str, Any]] = []
        collected: list[dict[str, Any]] = []
        for query in self.arxiv_queries:
            entries, status, error = self._fetch_source(
                f"arXiv: {query}", _arxiv_api_url(query), "arXiv"
            )
            for item in entries:
                _add_tags(item, keywords_in_text(query))
            collected.extend(entries)
            sources.append(
                {
                    "name": f"arXiv: {query}",
                    "status": status,
                    "count": len(entries),
                    "error": error,
                }
            )
        for name, url in self.rss_feeds:
            entries, status, error = self._fetch_source(name, url, name)
            collected.extend(entries)
            sources.append({"name": name, "status": status, "count": len(entries), "error": error})
        items = self._merge_items(collected, topic_list)
        payload = {
            "updated_at": now_iso(),
            "refresh_interval_sec": self.refresh_interval_sec,
            "sources": sources,
            "items": items,
        }
        with self._lock:
            self._cache = {**payload, "version": CACHE_VERSION}
            self._loaded = True
            self._write_cache(payload)
        return payload

    def _fetch_source(
        self, name: str, url: str, source: str
    ) -> tuple[list[dict[str, Any]], str, str]:
        """抓取单个来源；失败时返回空列表与错误文本，不抛出异常。"""
        try:
            xml_text = self._fetch_url(url)
            entries = parse_feed(xml_text, source)
        except Exception as exc:
            return [], "error", _error_text(exc)
        return entries, ("ok" if entries else "empty"), ""

    def _merge_items(self, items: list[dict[str, Any]], topics: list[str]) -> list[dict[str, Any]]:
        """按 id/标题去重、打主题标签、按发布时间倒序并截断。"""
        seen_ids: set[str] = set()
        seen_titles: set[str] = set()
        merged: list[dict[str, Any]] = []
        for item in items:
            item_id = str(item.get("id") or "")
            title_key = _collapse_ws(str(item.get("title") or "")).casefold()
            if not item_id or item_id in seen_ids:
                continue
            if title_key and title_key in seen_titles:
                continue
            seen_ids.add(item_id)
            if title_key:
                seen_titles.add(title_key)
            if topics:
                _apply_topics(item, topics)
            merged.append(item)
        merged.sort(
            key=lambda entry: (
                str(entry.get("published") or ""),
                _apply_sort_weight(entry, topics),
            ),
            reverse=True,
        )
        return merged[: self.max_items]

    # ------------------------------------------------------------------ 网络

    def _fetch_url(self, url: str) -> str:
        """内部下载入口：子类或测试可覆写，默认 urllib + 固定 UA + 超时。"""
        return self._fetch_bytes(url).decode("utf-8", "replace")

    def _fetch_bytes(self, url: str) -> bytes:
        """下载二进制内容（PDF 归档用），UA ``SpaceFL-Research/1.0``。"""
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "*/*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return bytes(response.read())

    # ------------------------------------------------------------------ 入库

    def ingest(self, item_id: str, target_dir: Path, extract_dir: Path) -> dict[str, Any]:
        """把资讯归档为 Markdown 文献。

        arXiv 条目会尝试下载 ``pdf_url`` 并抽取正文前 20000 字；下载或解析失败时
        只归档元数据与摘要。返回 ``{"ingested","path","name","title"}``。
        """
        item = self.find_item(item_id)
        if item is None:
            raise ValueError(f"未找到资讯条目：{item_id}")
        target_root = Path(target_dir)
        extract_root = Path(extract_dir)
        target_root.mkdir(parents=True, exist_ok=True)
        extract_root.mkdir(parents=True, exist_ok=True)
        title = _collapse_ws(str(item.get("title") or "")) or "未命名资讯"
        stem = safe_filename(f"{_published_date(item)}-{title}")
        path = _unique_path(target_root, stem, ".md")
        body = ""
        note = ""
        pdf_url = str(item.get("pdf_url") or "").strip()
        if pdf_url:
            payload = None
            try:
                payload = self._fetch_bytes(pdf_url)
            except Exception as exc:
                note = f"PDF 下载失败：{_error_text(exc)}，仅归档元数据与摘要"
            if payload:
                body, note = self._extract_pdf_body(payload, item, extract_root)
        path.write_text(_render_markdown(item, body, note), encoding="utf-8")
        self._mark_ingested(str(item.get("id") or item_id))
        return {"ingested": True, "path": str(path), "name": path.name, "title": title}

    def _extract_pdf_body(
        self, payload: bytes, item: dict[str, Any], extract_root: Path
    ) -> tuple[str, str]:
        """把下载到的 PDF 落盘抽取后删除临时文件，返回（正文, 备注）。"""
        from fl_space.research.literature import read_document_text

        temp_path = extract_root / f"{safe_filename(str(item.get('id') or 'intel'))}.pdf"
        try:
            temp_path.write_bytes(payload)
            pages, note = read_document_text(temp_path, extract_root)
            text = "\n\n".join(str(page.get("text") or "") for page in pages)
            if not text.strip():
                return "", "PDF 已下载但未抽取到文本，仅归档元数据与摘要"
            return text[:20000], f"正文由 {note} 抽取，最多保留 20000 字"
        except Exception as exc:
            return "", f"PDF 解析失败：{_error_text(exc)}，仅归档元数据与摘要"
        finally:
            with contextlib.suppress(OSError):
                temp_path.unlink()

    def find_item(self, item_id: str) -> dict[str, Any] | None:
        """在缓存中按 id 查找条目（必要时先读缓存文件）。"""
        wanted = str(item_id or "").strip()
        if not wanted:
            return None
        cache = self.cached()
        if cache is None:
            cache = self.fetch()
        for item in cache.get("items") or []:
            if str(item.get("id")) == wanted:
                return copy.deepcopy(item)
        return None

    def _mark_ingested(self, item_id: str) -> None:
        """把条目标记为已入库并写回缓存。"""
        with self._lock:
            cache = self._cache if self._cache is not None else self._read_cache()
            self._loaded = True
            if not isinstance(cache, dict):
                return
            changed = False
            for item in cache.get("items") or []:
                if str(item.get("id")) == item_id and not item.get("ingested"):
                    item["ingested"] = True
                    changed = True
            self._cache = cache
            if changed:
                payload = {key: value for key, value in cache.items() if key != "version"}
                self._write_cache(payload)


def _apply_sort_weight(entry: dict[str, Any], topics: list[str]) -> int:
    """排序加权：条目已带主题标签越多权重越高（同一时间戳时优先）。"""
    if not topics:
        return 0
    tags = {str(tag).casefold() for tag in entry.get("tags") or []}
    weight = 0
    for topic in topics:
        if any(keyword.casefold() in tags for keyword in topic_keywords(topic)):
            weight += 1
    return weight
