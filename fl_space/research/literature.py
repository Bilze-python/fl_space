"""文献知识库：目录扫描、PDF 文本抽取、分块、BM25 全文检索与 RAG 上下文。

本模块只依赖标准库；``pypdf`` 为可选依赖：未安装时 PDF 仍然以文件名元数据进入索引，
抽取文本标记为 ``missing``，不会向调用方抛出 ImportError。

约定与 ``web/server.py`` 保持一致：

- 文献根目录为 ``[(标签, Path)]`` 列表，逻辑路径形如 ``0:文献/xxx.pdf``；
- PDF 抽取文本缓存沿用 ``文献/extracted`` 的 ``===== PAGE n =====`` 分节格式。

示例::

    index = LiteratureIndex(roots, cache_path, extract_dir)
    index.load()
    index.rebuild()
    index.search("卫星 联邦学习 卸载")
"""

from collections import Counter
from collections.abc import Sequence
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any

# 索引缓存结构版本；不匹配时视为无效缓存（元数据抽取规则变更时递增）
INDEX_VERSION = 2

# 支持索引的文献后缀（与 web/server.py 的 LIBRARY_SUFFIXES 对齐）
SUPPORTED_SUFFIXES = frozenset({".pdf", ".md", ".markdown", ".txt"})

# 扫描时跳过的目录名（extracted 是抽取文本缓存目录，避免与原文重复入库）
SKIP_DIR_NAMES = frozenset({"extracted", "__pycache__", ".git", "node_modules", ".venv"})

# 抽取文本缓存的分页标记，与项目既有 文献/extracted 约定一致
PAGE_MARKER_TEMPLATE = "===== PAGE {page} ====="
PAGE_MARKER_RE = re.compile(r"^===== PAGE (\d+) =====[ \t]*$", re.MULTILINE)

# 英文/数字检索词（ASCII \w+，排除中文，中文走二元组分支）
_EN_TOKEN_RE = re.compile(r"[a-z0-9_]+")
# 连续汉字（含扩展 A / 兼容区）
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")

# Windows 文件名非法字符
_ILLEGAL_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# 英文停用词 + 论文结构词（结构词过滤可提升检索信噪比）
_EN_STOPWORDS = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "also", "an", "and", "any",
        "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
        "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "et",
        "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",
        "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
        "may", "me", "might", "more", "most", "must", "my", "no", "nor", "not", "now", "of", "off",
        "on", "once", "only", "or", "other", "ought", "our", "ours", "out", "over", "own", "same",
        "shall", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
        "them", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
        "who", "whom", "why", "will", "with", "would", "you", "your", "al", "eg", "ie", "fig",
        "figure", "table", "section", "et al", "abstract", "introduction", "conclusion",
        "references", "keywords", "index terms", "doi", "http", "https", "www", "preprint",
    }
)

# 中文停用词（仅对独立单字生效；二元组由上下文决定，过度过滤会损伤召回）
_CJK_STOPWORDS = frozenset(
    {"的", "了", "在", "是", "和", "与", "及", "或", "对", "于", "等", "中", "上", "下", "而", "也"}
)

# 判断标题候选行时需要剔除的页眉/版权/分页噪声
_HEADER_HINTS = (
    "arxiv:",
    "doi:",
    "copyright",
    "\u00a9",
    "all rights reserved",
    "preprint",
    "under review",
    "proceedings of",
    "issn",
    "isbn",
    "http://",
    "https://",
    "personal use is permitted",
    "authorized licensed use",
    "downloaded from",
)

# 机构/作者附加信息关键词，用于剔除标题与作者候选中的噪声行
_AFFILIATION_HINTS = (
    "university",
    "department",
    "school of",
    "institute",
    "laboratory",
    "college",
    "centre",
    "center",
    "faculty",
    "academy",
    "member",
    "fellow",
    "student",
    "ieee",
    "acm",
    "inc.",
    "ltd",
    "corporation",
)

# 期刊/会议线索词，用于 venue 猜测
_VENUE_HINTS = (
    "transactions",
    "journal",
    "conference",
    "proceedings",
    "symposium",
    "workshop",
    "magazine",
    "letters",
    "arxiv",
    "ieee",
    "acm",
    "eusipco",
    "globecom",
    "wcl",
)

# venue 行需要剔除的版权噪声短语
_VENUE_SKIP_PHRASES = (
    "personal use",
    "permitted",
    "authorized licensed",
    "downloaded",
    "reserved",
    "all rights",
    "member",
    "fellow",
)

_ORDINAL_RE = re.compile(r"^\s*\d+\s*(?:st|nd|rd|th)\s+", re.IGNORECASE)
_MEMBER_SUFFIX_RE = re.compile(
    r"(?i)\s*,?\s*(?:senior\s+|student\s+|life\s+|graduate\s+)?member\b.*$"
)
_NAME_TOKEN = r"(?:[A-Z]\.|[A-Z](?:[A-Za-z'\u2019\-]*[a-z])?)"
_PERSON_NAME_RE = re.compile(rf"^{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{1,2}}$")
# 人名片段两端需要剔除的脚注/序号符号
_NAME_NOISE_CHARS = " \u2217\u2020\u2021\u00b9\u00b2\u00b3*.,;:0123456789"

# 明显不属于人名的常见名词（用于区分"标题续行"与"单个作者名"）
_NON_NAME_WORDS = frozenset(
    {
        "adaptive", "aggregation", "allocation", "analysis", "approach", "based", "communication",
        "communications", "computing", "conference", "constellation", "constellations", "data",
        "decentralized", "deep", "distributed", "earth", "edge", "efficient", "federated",
        "framework", "in", "intelligence", "introduction", "journal", "learning", "leo", "low",
        "method", "methods", "model", "models", "network", "networks", "neural", "of", "offloading",
        "on", "onboard", "optimization", "orbit", "orbital", "performance", "personalized",
        "proceedings", "resource", "review", "satellite", "satellites", "scheduling", "space",
        "survey", "system", "systems", "the", "transactions", "university",
    }
)

# PDF 连字（ligature）还原表，避免 "Oﬄoading" 搜不到 "offloading"
_LIGATURES = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
    }
)
_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
_YEAR_HINT_RE = re.compile(
    r"(?i)(?:\u00a9|copyright|published|accepted|received|preprint)[^\d\n]{0,40}"
    r"(19[89]\d|20[0-3]\d)"
    r"|arxiv:[^\n]{0,40}?(19[89]\d|20[0-3]\d)"
)
_ARXIV_LINE_RE = re.compile(r"(?i)^arxiv:\s*([^\s\[]+)\s*(\[[^\]]+\])?")

_ABSTRACT_START_RE = re.compile(
    r"(?im)^[ \t]*(?:abstract|\u6458\s*\u8981)[ \t]*[\u2014\u2013\-:\uff1a.]?[ \t]*"
)
_ABSTRACT_ANY_RE = re.compile(r"(?i)\babstract\b[ \t]*[\u2014\u2013\-:\uff1a.]?")
_ABSTRACT_END_RE = re.compile(
    r"(?im)^[ \t]*(?:index\s+terms|key\s*words?|ccs\s+concepts|"
    r"(?:i|1)\s*[.)]?\s*introduction|introduction|"
    r"\u5173\u952e\u8bcd|\u5173\u952e\u5b57)"
)
_KEYWORDS_RE = re.compile(
    r"(?i)(?:index\s+terms|key\s*words?|\u5173\u952e\u8bcd|\u5173\u952e\u5b57)[\s*_]*"
    r"[\u2014\u2013\-:\uff1a]?"
)

# 关键词片段两端需要剔除的噪声字符（含中文标点与 arXiv 脚注符号；括号保留以维持配对）
_KEYWORD_TRIM = " .,;:'\"\u201c\u201d\u2018\u2019\u3002\uff1a:\uff1b\u2217\u2020\u2021"

# 关键词段落遇到这些标记即视为结束（避免把后面的 arXiv / DOI 行吞进关键词）
_KEYWORDS_STOP_HINTS = (
    "arxiv:",
    "doi",
    "http",
    "@",
    "\u00a9",
    "introduction",
    "ntroduction",
    "references",
    "index terms",
    "keywords",
)

_MD_TITLE_RE = re.compile(r"^#\s+(.+)$")
_MD_AUTHOR_RE = re.compile(r"(?im)^\s*>?\s*(?:authors?|\u4f5c\u8005)\s*[:\uff1a]\s*(.+)$")
_MD_VENUE_RE = re.compile(
    r"(?im)^\s*>?\s*(?:venue|\u6765\u6e90|\u53d1\u8868|\u671f\u520a|\u4f1a\u8bae)"
    r"\s*[:\uff1a]\s*(.+)$"
)
_MD_YEAR_RE = re.compile(
    r"(?m)^\s*>?\s*(?:year|\u5e74\u4efd|\u53d1\u8868\u65f6\u95f4|published)\s*[:\uff1a]\s*(\d{4})"
)


def _collapse_ws(text: str) -> str:
    """把任意空白压缩为单个空格并去除首尾。"""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_block(text: str) -> str:
    """规范化整块文本：统一换行、还原连字与断词、压缩空格与空行。"""
    normalized = str(text or "").translate(_LIGATURES)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    # 还原排版断词（行尾连字符 + 下一行小写字母），提升检索与关键词质量
    normalized = re.sub(r"([A-Za-z])-[ ]*\n[ ]*([a-z])", r"\1\2", normalized)
    normalized = re.sub(r"[ ]{2,}", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _safe_stem(stem: str, *, limit: int = 100) -> str:
    """把文件名主干清理成安全的缓存文件名。"""
    cleaned = _ILLEGAL_NAME_RE.sub("_", str(stem or "")).strip(" ._")
    if not cleaned:
        return "document"
    return cleaned[:limit].rstrip(" ._") or "document"


def _iso_timestamp(seconds: float) -> str:
    """时间戳（秒）→ 带时区的 ISO8601 字符串。"""
    try:
        moment = datetime.fromtimestamp(float(seconds), tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return ""
    return moment.isoformat(timespec="seconds")


def _now_iso() -> str:
    """当前 UTC 时间 → ISO8601 字符串。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_int(value: Any) -> int | None:
    """尽力把任意值转为整数，失败返回 None。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str_list(value: Any) -> list[str]:
    """把任意值安全转换为字符串列表。"""
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item).strip()]


def make_doc_id(logical_path: str) -> str:
    """逻辑路径 → 稳定文档 id（``lib_`` + sha1 前 10 位十六进制）。"""
    digest = hashlib.sha1(str(logical_path).encode("utf-8")).hexdigest()[:10]
    return f"lib_{digest}"


def document_type(path: Path | str) -> str:
    """按扩展名判断文献类型：``pdf`` / ``md`` / ``txt``。"""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".md", ".markdown"}:
        return "md"
    if suffix == ".txt":
        return "txt"
    return suffix.lstrip(".") or "unknown"


def tokenize(text: str) -> list[str]:
    """把中英混排文本切成检索词。

    英文/数字按 ``\\w+`` 小写化并过滤停用词；中文按连续汉字切二元组
    （单字区间保留单字），同样过滤常见虚词。
    """
    if not text:
        return []
    lowered = str(text).lower()
    tokens: list[str] = []
    for match in _EN_TOKEN_RE.finditer(lowered):
        word = match.group(0)
        if word in _EN_STOPWORDS:
            continue
        if len(word) < 2 and not word.isdigit():
            continue
        tokens.append(word)
    for match in _CJK_RUN_RE.finditer(str(text)):
        run = match.group(0)
        if len(run) == 1:
            if run not in _CJK_STOPWORDS:
                tokens.append(run)
            continue
        for start in range(len(run) - 1):
            bigram = run[start : start + 2]
            if bigram in _CJK_STOPWORDS:
                continue
            tokens.append(bigram)
    return tokens


class BM25:
    """Okapi BM25 打分器（中文按二元组词元处理）。"""

    def __init__(self, docs: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        """用词元化后的文档集合初始化统计量。"""
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_count = len(docs)
        self._freqs: list[Counter[str]] = [Counter(doc) for doc in docs]
        self._lengths = [len(doc) for doc in docs]
        total = sum(self._lengths)
        self._avg_len = (total / self.doc_count) if self.doc_count else 1.0
        if self._avg_len <= 0:
            self._avg_len = 1.0
        document_freq: Counter[str] = Counter()
        for freqs in self._freqs:
            document_freq.update(freqs.keys())
        self._idf = {
            term: math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }

    def scores(self, query: list[str]) -> list[float]:
        """计算查询词元对每个文档的 BM25 得分。"""
        results = [0.0] * self.doc_count
        if not self.doc_count or not query:
            return results
        for term in query:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index, freqs in enumerate(self._freqs):
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                norm = self.k1 * (
                    1.0 - self.b + self.b * self._lengths[index] / self._avg_len
                )
                results[index] += idf * (tf * (self.k1 + 1.0)) / (tf + norm)
        return results


def chunk_pages(
    pages: list[dict[str, Any]], *, chunk_chars: int = 1400, overlap: int = 200
) -> list[dict[str, Any]]:
    """把逐页文本切成带重叠的检索块（块不跨页，便于引用页码）。"""
    size = max(120, int(chunk_chars))
    lap = max(0, int(overlap))
    if lap >= size:
        lap = max(0, size // 5)
    chunks: list[dict[str, Any]] = []
    for position, page in enumerate(pages):
        page_no = _as_int(page.get("page")) or (position + 1)
        text = _normalize_block(str(page.get("text") or ""))
        if not text:
            continue
        start = 0
        length = len(text)
        while start < length:
            end = min(length, start + size)
            if end < length:
                end = _cut_point(text, start, end)
            piece = text[start:end].strip()
            if piece:
                chunks.append({"page": page_no, "text": piece, "chars": len(piece)})
            if end >= length:
                break
            start = max(start + 1, end - lap)
    return chunks


def _cut_point(text: str, start: int, end: int) -> int:
    """在 ``[start, end)`` 内尽量找一个语义边界作为切分点。"""
    window_start = max(start + 1, end - 160)
    window = text[window_start:end]
    for separator in ("\n\n", "\n", ". ", "\u3002", "! ", "; ", " "):
        found = window.rfind(separator)
        if found > 0:
            cut = window_start + found + len(separator)
            if start < cut <= end:
                return cut
    return end


def extract_pdf_text(path: Path) -> tuple[list[str], str]:
    """用 pypdf 逐页抽取 PDF 文本，返回 ``(每页文本, 标记)``。

    标记取值：``pypdf``（成功）或 ``missing``（未安装 pypdf / 解析失败）。
    本函数不抛异常，损坏文件同样返回空列表，便于上层记录错误后继续索引。
    """
    try:
        import pypdf
    except ImportError:
        return [], "missing"
    reader_factory = getattr(pypdf, "PdfReader", None)
    if reader_factory is None:
        return [], "missing"
    try:
        reader = reader_factory(str(path), strict=False)
        pages: list[str] = []
        for page in reader.pages:
            try:
                content = page.extract_text() or ""
            except Exception:
                content = ""
            pages.append(str(content))
    except Exception:
        return [], "missing"
    return pages, "pypdf"


def _cache_file(path: Path, extract_dir: Path) -> Path:
    """PDF 抽取文本缓存路径：``extract_dir/<stem>.txt``。"""
    return Path(extract_dir) / f"{_safe_stem(Path(path).stem)}.txt"


def _read_extract_cache(cache_path: Path, source: Path) -> tuple[list[dict[str, Any]], str] | None:
    """读取抽取文本缓存；缺失、过期或内容不可解析时返回 None。"""
    try:
        if not cache_path.is_file():
            return None
        if cache_path.stat().st_mtime + 1.0 < source.stat().st_mtime:
            return None
        raw = cache_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    pages = _split_extract_cache(raw)
    if not pages:
        return None
    return pages, "cache"


def _split_extract_cache(raw: str) -> list[dict[str, Any]]:
    """按 ``===== PAGE n =====`` 分节解析抽取文本缓存。"""
    pages: list[dict[str, Any]] = []
    matches = list(PAGE_MARKER_RE.finditer(raw))
    if not matches:
        return pages
    for index, match in enumerate(matches):
        page_no = _as_int(match.group(1)) or (index + 1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        pages.append({"page": page_no, "text": _normalize_block(raw[start:end])})
    return pages


def _write_extract_cache(cache_path: Path, pages: Sequence[dict[str, Any]]) -> None:
    """把逐页文本写入抽取缓存（失败静默，不影响主流程）。"""
    blocks: list[str] = []
    for page in pages:
        page_no = int(page.get("page") or 1)
        text = str(page.get("text") or "")
        blocks.append(f"{PAGE_MARKER_TEMPLATE.format(page=page_no)}\n{text}\n")
    payload = "\n".join(blocks)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_name(f"{cache_path.name}.tmp")
        temp_path.write_text(payload, encoding="utf-8")
        os.replace(temp_path, cache_path)
    except OSError:
        with contextlib.suppress(OSError):
            cache_path.with_name(f"{cache_path.name}.tmp").unlink()


def _read_text_document(path: Path) -> str | None:
    """读取 Markdown/TXT 文本；UTF-8 失败时回退 GBK。"""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pass
    except OSError:
        return None
    try:
        return path.read_text(encoding="gbk", errors="replace")
    except OSError:
        return None


def _truncate_pages(pages: Sequence[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    """累计文本裁剪到 ``max_chars``，保留页码信息。"""
    budget = max(0, int(max_chars))
    result: list[dict[str, Any]] = []
    for page in pages:
        if budget <= 0:
            break
        text = str(page.get("text") or "")
        if len(text) > budget:
            text = text[:budget]
        budget -= len(text)
        result.append({"page": _as_int(page.get("page")) or (len(result) + 1), "text": text})
    return result


def read_document_text(
    path: Path, extract_dir: Path, *, max_chars: int = 1_200_000
) -> tuple[list[dict[str, Any]], str]:
    """读取文献正文，返回 ``(pages, note)``。

    - ``.md`` / ``.txt``：按空行分段，全部内容归入 ``page=1``，标记 ``markdown``；
    - ``.pdf``：优先复用 ``extract_dir/<stem>.txt`` 缓存（标记 ``cache``），
      否则用 pypdf 抽取并写缓存（标记 ``pypdf``）；
    - 无 pypdf 且无缓存时返回 ``([], "missing")``。
    """
    target = Path(path)
    kind = document_type(target)
    if kind == "pdf":
        cache_path = _cache_file(target, extract_dir)
        cached = _read_extract_cache(cache_path, target)
        if cached is not None:
            pages, note = cached
            return _truncate_pages(pages, max_chars), note
        texts, note = extract_pdf_text(target)
        if not texts:
            return [], note
        pages = [
            {"page": index, "text": _normalize_block(text)}
            for index, text in enumerate(texts, start=1)
        ]
        _write_extract_cache(cache_path, pages)
        return _truncate_pages(pages, max_chars), note
    if kind in {"md", "txt"}:
        text = _read_text_document(target)
        if text is None:
            return [], "missing"
        normalized = _normalize_paragraphs(text)[: max(0, int(max_chars))].strip()
        if not normalized:
            return [], "missing"
        return [{"page": 1, "text": normalized}], "markdown"
    return [], "missing"


def _normalize_paragraphs(text: str) -> str:
    """按空行切段后重新拼接，压缩冗余空白。"""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [
        _collapse_ws(block) for block in re.split(r"\n\s*\n", normalized) if _collapse_ws(block)
    ]
    return "\n\n".join(paragraphs)


def _fallback_title(path: Path) -> str:
    """文件名兜底标题：去掉 ``[11]_`` 前缀，下划线转空格。"""
    stem = re.sub(r"^\[\d+\][_\-\s]*", "", Path(path).stem)
    cleaned = _collapse_ws(stem.replace("_", " ").replace("-", " "))
    return cleaned or Path(path).stem


def _title_candidates(lines: Sequence[str]) -> list[tuple[int, str]]:
    """从首页前 15 行中挑出可能的标题行（行号, 文本）。

    标题一定出现在摘要 / Index Terms 之前，遇到摘要标记即停止，
    避免把摘要正文的长行误判为标题。
    """
    candidates: list[tuple[int, str]] = []
    for position, raw in enumerate(list(lines)[:15]):
        line = _collapse_ws(raw)
        lowered = line.lower()
        if lowered.startswith(("abstract", "index terms", "keywords", "summary", "\u6458\u8981")):
            break
        if len(line) < 12 or len(line) > 220:
            continue
        if any(hint in lowered for hint in _HEADER_HINTS):
            continue
        if any(hint in lowered for hint in _AFFILIATION_HINTS):
            continue
        if "@" in line:
            continue
        if _looks_like_author_line(line):
            continue
        if sum(character.isdigit() for character in line) > len(line) * 0.2:
            continue
        if sum(character.isalpha() for character in line) < len(line) * 0.5:
            continue
        candidates.append((position, line))
    return candidates


def _guess_pdf_title(lines: Sequence[str]) -> tuple[str, int]:
    """猜 PDF 标题：摘要之前最长的非页眉行，并拼接续行。

    返回 ``(标题, 标题结束后第一行的行号)``，行号用于定位作者行。
    """
    candidates = _title_candidates(lines)
    if not candidates:
        candidates = _title_candidates(list(lines)[:30])
    if not candidates:
        return "", 0
    position, title = max(candidates, key=lambda item: len(item[1]))
    merged = title
    end = position + 1
    for offset, raw in enumerate(list(lines)[position + 1 : position + 4], start=1):
        piece = _collapse_ws(raw)
        if not piece or len(piece) > 100 or "@" in piece:
            break
        lowered = piece.lower()
        if lowered.startswith(("abstract", "index terms", "keywords")):
            break
        if any(hint in lowered for hint in _AFFILIATION_HINTS):
            break
        if any(character.isdigit() for character in piece):
            break
        if _looks_like_author_line(piece) or _looks_like_person_name(piece):
            break
        if merged.endswith((".", "?", "!", ":", ";")):
            break
        if len(merged) + len(piece) > 200:
            break
        merged = f"{merged} {piece}"
        end = position + 1 + offset
    return merged, end


def _looks_like_person_name(name: str) -> bool:
    """判断片段是否像人名（2-3 个首字母大写的词，且不含名词性词汇）。"""
    cleaned = name.strip(" ,;:")
    if not cleaned or len(cleaned) > 60:
        return False
    if not _PERSON_NAME_RE.match(cleaned):
        return False
    words = [word.strip(".,;:").lower() for word in cleaned.split()]
    return not any(word in _NON_NAME_WORDS for word in words)


def _looks_like_author_line(line: str) -> bool:
    """判断整行是否像作者行（≥2 个逗号/and 分隔的人名片段）。"""
    text = _collapse_ws(line)
    if len(text) < 8:
        return False
    pieces = [piece for piece in re.split(r",|\band\b|&", text) if piece.strip()]
    if len(pieces) < 2:
        return False
    names = 0
    for piece in pieces:
        cleaned = _ORDINAL_RE.sub("", piece)
        cleaned = _MEMBER_SUFFIX_RE.sub("", cleaned)
        cleaned = cleaned.strip(_NAME_NOISE_CHARS)
        if _looks_like_person_name(cleaned):
            names += 1
    return names >= 2 and names >= len(pieces) - 1


def _guess_pdf_authors(lines: Sequence[str], start: int) -> list[str]:
    """从标题（含续行）之后若干行中猜作者列表。"""
    author_lines = list(lines)
    authors: list[str] = []
    seen: set[str] = set()
    for raw in author_lines[max(0, start) : max(0, start) + 12]:
        line = _collapse_ws(raw)
        if not line:
            continue
        if line.lower().startswith(("abstract", "index terms", "keywords")):
            break
        cleaned = _MEMBER_SUFFIX_RE.sub("", line)
        for piece in re.split(r",|\band\b|&", cleaned):
            unnumbered = _ORDINAL_RE.sub("", piece)
            if any(character.isdigit() for character in unnumbered):
                continue
            name = unnumbered.strip(_NAME_NOISE_CHARS)
            lowered = name.lower()
            if not name or "@" in name or any(hint in lowered for hint in _AFFILIATION_HINTS):
                continue
            if not _looks_like_person_name(name):
                continue
            if name.casefold() in seen:
                continue
            seen.add(name.casefold())
            authors.append(name)
            if len(authors) >= 8:
                return authors
    return authors


def _guess_venue(lines: Sequence[str], title: str) -> str:
    """从页眉/首页信息中猜期刊或会议名。"""
    title_key = _collapse_ws(title).casefold()
    for raw in list(lines)[:80]:
        line = _collapse_ws(raw)
        if len(line) < 6 or len(line) > 160:
            continue
        lowered = line.casefold()
        if title_key and title_key in lowered:
            continue
        arxiv_match = _ARXIV_LINE_RE.match(line)
        if arxiv_match:
            category = (arxiv_match.group(2) or "").strip()
            return _collapse_ws(f"arXiv {category}")
        if not any(hint in lowered for hint in _VENUE_HINTS):
            continue
        if any(phrase in lowered for phrase in _VENUE_SKIP_PHRASES):
            continue
        if _looks_like_author_line(line):
            continue
        cleaned = re.sub(
            r"(?i)^(?:published\s+in|appeared\s+in|in)\s+", "", line
        ).strip(" .,;:\u00a9")
        cleaned = re.sub(r"(?i)\s*(?:vol\.|volume)\s*\d+.*$", "", cleaned).strip(" .,;:")
        if len(cleaned) < 6:
            continue
        if re.match(r"(?i)^(?:\u00a9|copyright)?\s*\d{4}", cleaned):
            continue
        if sum(character.isalpha() for character in cleaned) < len(cleaned) * 0.4:
            continue
        return cleaned[:120]
    return ""


def _guess_abstract(text: str) -> str:
    """从正文中截取摘要（Abstract 之后到 Index Terms / Introduction 之前）。"""
    if not text:
        return ""
    match = _ABSTRACT_START_RE.search(text)
    if match is None:
        match = _ABSTRACT_ANY_RE.search(text)
    if match is None:
        return ""
    tail = text[match.end() :]
    end = _ABSTRACT_END_RE.search(tail)
    if end is not None and end.start() > 40:
        tail = tail[: end.start()]
    cleaned = _collapse_ws(tail)
    if len(cleaned) < 40:
        return ""
    return cleaned[:1600]


def _guess_keywords(text: str) -> list[str]:
    """从 Index Terms / Keywords 段落中提取关键词列表。"""
    match = _KEYWORDS_RE.search(text or "")
    if match is None:
        return []
    tail = text[match.end() :]
    collected: list[str] = []
    for raw in tail.split("\n")[:4]:
        line = _collapse_ws(raw)
        if not line:
            break
        if any(hint in line.lower() for hint in _KEYWORDS_STOP_HINTS):
            break
        collected.append(line)
        if len(collected) >= 2:
            break
    tail = " ".join(collected)[:300].strip(" \u2014\u2013-:\uff1a.")
    if not tail:
        return []
    keywords: list[str] = []
    seen: set[str] = set()
    for piece in re.split(r"[;,\uff0c\u3001\u00b7]|\band\b|\uff1b", tail):
        word = _collapse_ws(piece.strip(_KEYWORD_TRIM))
        if not word or len(word) > 60:
            continue
        key = word.casefold()
        if key in seen:
            continue
        seen.add(key)
        keywords.append(word)
        if len(keywords) >= 12:
            break
    return keywords


def _guess_markdown_title(lines: Sequence[str], path: Path, kind: str) -> str:
    """Markdown 取首个一级标题；TXT 取首个非空行；都失败则用文件名。"""
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("# "):
            return _collapse_ws(stripped[2:])
    if kind == "txt":
        for raw in lines:
            stripped = raw.strip()
            if stripped and len(stripped) <= 160:
                return _collapse_ws(stripped)
    return _fallback_title(path)


def _guess_markdown_abstract(lines: Sequence[str], title: str) -> str:
    """Markdown 取标题之后的首个正文段落作为摘要。"""
    paragraph: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            if paragraph:
                break
            continue
        if line.startswith(("#", ">", "|", "```", "- ", "* ", "+ ", "!", "---", "[")):
            if paragraph:
                break
            continue
        if title and _collapse_ws(line).strip("# ") == _collapse_ws(title):
            continue
        paragraph.append(line)
    text = _collapse_ws(" ".join(paragraph))
    if len(text) < 30:
        return ""
    return text[:1600]


def _guess_markdown_authors(head: str) -> list[str]:
    """从 ``作者：`` 等标记行中提取作者。"""
    match = _MD_AUTHOR_RE.search(head)
    if match is None:
        return []
    authors: list[str] = []
    for piece in re.split(r"[,;\uff0c\u3001\uff1b]|\band\b|\u548c", match.group(1)):
        name = _collapse_ws(piece.strip(" .,;:\uff0c\u3001"))
        if not name or len(name) > 60:
            continue
        authors.append(name)
    return authors[:8]


def _guess_markdown_year(head: str) -> int | None:
    """从 ``年份：`` 标记行中提取四位年份。"""
    match = _MD_YEAR_RE.search(head)
    if match is None:
        return None
    return _as_int(match.group(1))


def _guess_year(text: str) -> int | None:
    """猜发表年份：优先版权/arXiv 提示行，其次取出现最多的四位年份。"""
    if not text:
        return None
    hinted = _YEAR_HINT_RE.search(text)
    if hinted is not None:
        for group in hinted.groups():
            if group:
                return _as_int(group)
    matches = _YEAR_RE.findall(text)
    if not matches:
        return None
    counter = Counter(_as_int(item) for item in matches if _as_int(item) is not None)
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def guess_metadata(
    pages: list[dict[str, Any]], path: Path, doc_type: str
) -> dict[str, Any]:
    """从抽取文本猜测文献元数据。

    返回 ``{"title","authors","year","venue","abstract","keywords"}``；
    PDF 走首页版式启发式，Markdown/TXT 走标题与标记行，全部失败时用文件名兜底。
    """
    lines: list[str] = []
    for page in list(pages)[:2]:
        lines.extend(str(page.get("text") or "").splitlines())
    head_text = "\n".join(lines)
    full_text = "\n".join(str(page.get("text") or "") for page in pages)
    title = ""
    authors: list[str] = []
    venue = ""
    abstract = ""
    keywords: list[str] = []
    if doc_type == "pdf":
        title, title_end = _guess_pdf_title(lines)
        if title:
            authors = _guess_pdf_authors(lines, title_end)
        venue = _guess_venue(lines, title)
        abstract = _guess_abstract(full_text or head_text)
        keywords = _guess_keywords(full_text[:8000] if full_text else head_text)
        year = _guess_year(head_text) or _guess_year(full_text[:4000])
    else:
        title = _guess_markdown_title(lines, Path(path), doc_type)
        abstract = _guess_markdown_abstract(lines, title)
        authors = _guess_markdown_authors(head_text)
        venue_match = _MD_VENUE_RE.search(head_text)
        venue = _collapse_ws(venue_match.group(1)) if venue_match else ""
        keywords = _guess_keywords(head_text)
        year = _guess_markdown_year(head_text) or _guess_year("\n".join(lines[:6]))
    return {
        "title": _collapse_ws(title) or _fallback_title(Path(path)),
        "authors": authors,
        "year": year,
        "venue": venue,
        "abstract": abstract,
        "keywords": keywords,
    }


def _derive_tags(title: str, keywords: Sequence[str]) -> list[str]:
    """由标题与关键词派生轻量标签。

    优先保留 3 字符以上的英文词；英文不足时用中文二元组补齐，保证中文文献也有标签。
    """
    latin: list[str] = []
    cjk: list[str] = []
    seen: set[str] = set()
    for source in (title, " ".join(keywords)):
        for token in tokenize(source):
            if token in seen:
                continue
            if token.isascii():
                if len(token) < 3 or token.isdigit():
                    continue
                seen.add(token)
                latin.append(token)
            elif len(token) == 2:
                seen.add(token)
                cjk.append(token)
    return (latin + cjk)[:8]


@dataclass(frozen=True)
class LiteratureDoc:
    """一条已索引的文献记录。"""

    id: str
    path: str
    abs_path: str
    title: str
    authors: list[str]
    year: int | None
    venue: str
    doc_type: str
    size: int
    pages: int
    chunks: int
    abstract: str
    keywords: list[str]
    extracted: bool
    extract_note: str
    modified_at: str
    source: str

    def to_public(self) -> dict[str, Any]:
        """导出前端契约字段（剔除 ``abs_path`` 等服务端内部信息）。"""
        return {
            "id": self.id,
            "path": self.path,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "type": self.doc_type,
            "source": self.source,
            "size": self.size,
            "pages": self.pages,
            "chunks": self.chunks,
            "abstract": self.abstract,
            "keywords": list(self.keywords),
            "tags": _derive_tags(self.title, self.keywords),
            "extracted": self.extracted,
            "extract_note": self.extract_note,
            "modified_at": self.modified_at,
        }


def _doc_from_cache(entry: Any) -> LiteratureDoc | None:
    """从缓存字典还原文献记录；关键字段缺失时返回 None。"""
    if not isinstance(entry, dict):
        return None
    doc_id = str(entry.get("id") or "")
    logical = str(entry.get("path") or "")
    if not doc_id or not logical:
        return None
    return LiteratureDoc(
        id=doc_id,
        path=logical,
        abs_path=str(entry.get("abs_path") or ""),
        title=str(entry.get("title") or Path(logical).name),
        authors=_as_str_list(entry.get("authors")),
        year=_as_int(entry.get("year")),
        venue=str(entry.get("venue") or ""),
        doc_type=str(entry.get("doc_type") or "txt"),
        size=_as_int(entry.get("size")) or 0,
        pages=_as_int(entry.get("pages")) or 0,
        chunks=_as_int(entry.get("chunks")) or 0,
        abstract=str(entry.get("abstract") or ""),
        keywords=_as_str_list(entry.get("keywords")),
        extracted=bool(entry.get("extracted")),
        extract_note=str(entry.get("extract_note") or "missing"),
        modified_at=str(entry.get("modified_at") or ""),
        source=str(entry.get("source") or ""),
    )


def _doc_entry(doc: LiteratureDoc, *, mtime: float, extract: str, chars: int) -> dict[str, Any]:
    """把文献记录打包成缓存条目（附带增量重建所需的内部字段）。"""
    return {
        "id": doc.id,
        "path": doc.path,
        "abs_path": doc.abs_path,
        "title": doc.title,
        "authors": list(doc.authors),
        "year": doc.year,
        "venue": doc.venue,
        "doc_type": doc.doc_type,
        "size": doc.size,
        "pages": doc.pages,
        "chunks": doc.chunks,
        "abstract": doc.abstract,
        "keywords": list(doc.keywords),
        "extracted": doc.extracted,
        "extract_note": doc.extract_note,
        "modified_at": doc.modified_at,
        "source": doc.source,
        "mtime": float(mtime),
        "chars": int(chars),
        "extract": extract,
    }


class LiteratureIndex:
    """文献库索引：扫描目录、抽取文本、分块、BM25 检索与 RAG 上下文。"""

    def __init__(
        self,
        roots: list[tuple[str, Path]],
        cache_path: Path,
        extract_dir: Path,
        *,
        max_file_size: int = 30 * 1024 * 1024,
        max_chars: int = 1_200_000,
        chunk_chars: int = 1400,
        chunk_overlap: int = 200,
    ) -> None:
        """初始化索引；不扫描目录、不读大文件，可安全地在导入后立即构造。"""
        self.roots: list[tuple[str, Path]] = [
            (str(label), Path(path)) for label, path in roots
        ]
        self.cache_path = Path(cache_path)
        self.extract_dir = Path(extract_dir)
        self.max_file_size = max(1024, int(max_file_size))
        self.max_chars = max(1000, int(max_chars))
        self.chunk_chars = max(200, int(chunk_chars))
        self.chunk_overlap = max(0, int(chunk_overlap))
        if self.chunk_overlap >= self.chunk_chars:
            self.chunk_overlap = self.chunk_chars // 4
        self._lock = threading.RLock()
        self._docs: list[LiteratureDoc] = []
        self._by_id: dict[str, LiteratureDoc] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._pages: dict[str, list[dict[str, Any]]] = {}
        self._chunks: dict[str, list[dict[str, Any]]] = {}
        self._chunk_index: list[tuple[str, dict[str, Any]]] = []
        self._bm25: BM25 | None = None
        self._corpus_ready = False
        self.last_errors: list[dict[str, str]] = []

    # ------------------------------------------------------------------ 缓存

    def load(self) -> None:
        """读取索引缓存；文件缺失、损坏或版本不符时保持空索引。"""
        with self._lock:
            try:
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            if not isinstance(payload, dict):
                return
            if _as_int(payload.get("version")) != INDEX_VERSION:
                return
            entries = payload.get("docs")
            if not isinstance(entries, list):
                return
            docs: list[LiteratureDoc] = []
            meta: dict[str, dict[str, Any]] = {}
            for entry in entries:
                doc = _doc_from_cache(entry)
                if doc is None or doc.id in meta:
                    continue
                docs.append(doc)
                meta[doc.id] = dict(entry)
            self._set_docs(docs, meta)

    def save(self) -> None:
        """把索引元数据写入缓存 JSON（chunk 文本不入缓存，避免文件过大）。"""
        with self._lock:
            payload = {
                "version": INDEX_VERSION,
                "updated_at": _now_iso(),
                "params": {
                    "max_file_size": self.max_file_size,
                    "max_chars": self.max_chars,
                    "chunk_chars": self.chunk_chars,
                    "chunk_overlap": self.chunk_overlap,
                },
                "roots": [[label, str(path)] for label, path in self.roots],
                "docs": [self._meta[doc.id] for doc in self._docs if doc.id in self._meta],
            }
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = self.cache_path.with_name(f"{self.cache_path.name}.tmp")
                temp_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
                )
                os.replace(temp_path, self.cache_path)
            except OSError:
                with contextlib.suppress(OSError):
                    self.cache_path.with_name(f"{self.cache_path.name}.tmp").unlink()

    # ------------------------------------------------------------------ 构建

    def rebuild(self, *, force: bool = False) -> dict[str, Any]:
        """增量重建索引。

        文件 ``(mtime, size)`` 未变且已有抽取缓存时复用旧记录；损坏文件记录到
        ``errors`` 后继续，绝不整体失败。返回
        ``{"indexed","parsed","reused","skipped","errors"}``。
        """
        with self._lock:
            previous = dict(self._meta)
            docs: list[LiteratureDoc] = []
            meta: dict[str, dict[str, Any]] = {}
            errors: list[dict[str, str]] = []
            parsed = 0
            reused = 0
            skipped = 0
            for root_index, (label, root) in enumerate(self.roots):
                if not root.is_dir():
                    continue
                for path in self._iter_files(root):
                    logical = f"{root_index}:{self._relative(path, root)}"
                    try:
                        stat = path.stat()
                    except OSError as exc:
                        skipped += 1
                        errors.append({"path": logical, "error": f"无法读取: {exc}"})
                        continue
                    if stat.st_size > self.max_file_size:
                        skipped += 1
                        continue
                    kind = document_type(path)
                    doc_id = make_doc_id(logical)
                    old = previous.get(doc_id)
                    if _is_reusable(old, stat, kind, force):
                        doc = _doc_from_cache(old)
                        if doc is not None:
                            docs.append(doc)
                            meta[doc_id] = old
                            reused += 1
                            continue
                    try:
                        doc, chars = self._build_doc(path, logical, label, stat, kind, doc_id)
                    except Exception as exc:
                        skipped += 1
                        errors.append({"path": logical, "error": f"{type(exc).__name__}: {exc}"})
                        continue
                    parsed += 1
                    docs.append(doc)
                    meta[doc_id] = _doc_entry(
                        doc,
                        mtime=stat.st_mtime,
                        extract=self._extract_path_for(path),
                        chars=chars,
                    )
                    if kind == "pdf" and not doc.extracted:
                        errors.append(
                            {
                                "path": logical,
                                "error": "未抽取到文本：缺少 pypdf 或 PDF 损坏",
                            }
                        )
            self._set_docs(docs, meta)
            self.last_errors = errors
            self.save()
            return {
                "indexed": len(docs),
                "parsed": parsed,
                "reused": reused,
                "skipped": skipped,
                "errors": errors,
            }

    def _build_doc(
        self,
        path: Path,
        logical: str,
        label: str,
        stat: Any,
        kind: str,
        doc_id: str,
    ) -> tuple[LiteratureDoc, int]:
        """解析单个文件，返回文献记录与正文字符数。"""
        pages, note = read_document_text(path, self.extract_dir, max_chars=self.max_chars)
        metadata = guess_metadata(pages, path, kind)
        chunks = chunk_pages(
            pages, chunk_chars=self.chunk_chars, overlap=self.chunk_overlap
        )
        chars = sum(len(str(page.get("text") or "")) for page in pages)
        doc = LiteratureDoc(
            id=doc_id,
            path=logical,
            abs_path=str(path.resolve()),
            title=metadata["title"],
            authors=list(metadata["authors"]),
            year=metadata["year"],
            venue=metadata["venue"],
            doc_type=kind,
            size=int(stat.st_size),
            pages=len(pages),
            chunks=len(chunks),
            abstract=metadata["abstract"],
            keywords=list(metadata["keywords"]),
            extracted=bool(pages),
            extract_note=note,
            modified_at=_iso_timestamp(stat.st_mtime),
            source=label,
        )
        return doc, chars

    def _iter_files(self, root: Path) -> list[Path]:
        """列出根目录下可索引的文献文件（跳过缓存目录与隐藏文件）。"""
        files: list[Path] = []
        extract_root = self.extract_dir
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if path.name.startswith("."):
                continue
            relative_parts = path.relative_to(root).parts[:-1]
            if any(part.lower() in SKIP_DIR_NAMES for part in relative_parts):
                continue
            try:
                path.relative_to(extract_root)
            except ValueError:
                files.append(path)
        return files

    def _relative(self, path: Path, root: Path) -> str:
        """相对路径 → POSIX 字符串（逻辑路径的 ``xxx/yyy`` 部分）。"""
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.name

    def _extract_path_for(self, path: Path) -> str:
        """该文件的抽取文本缓存绝对路径（非 PDF 返回空串）。"""
        if document_type(path) != "pdf":
            return ""
        return str(_cache_file(path, self.extract_dir))

    def _set_docs(self, docs: list[LiteratureDoc], meta: dict[str, dict[str, Any]]) -> None:
        """替换索引状态并清空惰性文本缓存（调用方需持锁）。"""
        self._docs = docs
        self._by_id = {doc.id: doc for doc in docs}
        self._meta = meta
        self._pages.clear()
        self._chunks.clear()
        self._chunk_index = []
        self._bm25 = None
        self._corpus_ready = False

    # ------------------------------------------------------------------ 读取

    def items(self) -> list[LiteratureDoc]:
        """返回全部已索引文献（按根目录与路径顺序）。"""
        with self._lock:
            return list(self._docs)

    def get(self, doc_id: str) -> LiteratureDoc | None:
        """按 id 取单篇文献，未命中返回 None。"""
        with self._lock:
            return self._by_id.get(str(doc_id))

    def pages(self, doc_id: str) -> list[dict[str, Any]]:
        """返回单篇文献的逐页文本 ``[{"page":1,"text":...}]``。"""
        with self._lock:
            doc = self._by_id.get(str(doc_id))
            if doc is None:
                return []
            if doc.id in self._pages:
                return list(self._pages[doc.id])
            pages = self._load_pages(doc)
            self._pages[doc.id] = pages
            return list(pages)

    def _load_pages(self, doc: LiteratureDoc) -> list[dict[str, Any]]:
        """按需读取正文（PDF 走抽取缓存，Markdown/TXT 直接读文件）。"""
        if not doc.abs_path:
            return []
        path = Path(doc.abs_path)
        if not path.is_file():
            return []
        try:
            pages, _note = read_document_text(
                path, self.extract_dir, max_chars=self.max_chars
            )
        except Exception:
            return []
        return pages

    # ------------------------------------------------------------------ 检索

    def _ensure_corpus(self) -> None:
        """惰性构建 BM25 语料（调用方需持锁）。"""
        if self._corpus_ready:
            return
        chunk_index: list[tuple[str, dict[str, Any]]] = []
        token_lists: list[list[str]] = []
        for doc in self._docs:
            pages = self._pages.get(doc.id)
            if pages is None:
                pages = self._load_pages(doc)
                self._pages[doc.id] = pages
            chunks = self._chunks.get(doc.id)
            if chunks is None:
                chunks = chunk_pages(
                    pages, chunk_chars=self.chunk_chars, overlap=self.chunk_overlap
                )
                self._chunks[doc.id] = chunks
            for chunk in chunks:
                chunk_index.append((doc.id, chunk))
                token_lists.append(tokenize(str(chunk.get("text") or "")))
        self._chunk_index = chunk_index
        self._bm25 = BM25(token_lists) if token_lists else None
        self._corpus_ready = True

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """全文检索：BM25 命中 chunk 后按文档聚合，标题命中的文档加权。

        每条结果为 ``doc.to_public()`` 加上 ``score`` 与 1-2 条 ``snippets``
        （``{"page": int, "text": str}``，截取查询词附近 ±150 字）。
        """
        tokens = tokenize(query)
        text = str(query or "").strip()
        if not text:
            return []
        limit = max(1, min(_as_int(limit) or 8, 50))
        with self._lock:
            self._ensure_corpus()
            hits_by_doc = self._collect_hits(tokens)
            if not hits_by_doc and tokens:
                hits_by_doc = self._substring_hits(text)
            if not hits_by_doc:
                return []
            query_terms = set(tokens)
            results: list[dict[str, Any]] = []
            for doc_id, hits in hits_by_doc.items():
                doc = self._by_id.get(doc_id)
                if doc is None:
                    continue
                hits.sort(key=lambda pair: pair[0], reverse=True)
                score = hits[0][0] + 0.3 * sum(pair[0] for pair in hits[1:3])
                title_tokens = set(tokenize(f"{doc.title} {' '.join(doc.keywords)}"))
                matched = len(query_terms & title_tokens)
                if matched and query_terms:
                    score *= 1.0 + 2.0 * (matched / len(query_terms))
                payload = doc.to_public()
                payload["score"] = round(float(score), 4)
                payload["snippets"] = [
                    {
                        "page": _as_int(chunk.get("page")) or 1,
                        "text": _make_snippet(str(chunk.get("text") or ""), text, tokens),
                    }
                    for _score, chunk in hits[:2]
                ]
                results.append(payload)
            results.sort(key=lambda item: item["score"], reverse=True)
            return results[:limit]

    def _collect_hits(
        self, tokens: list[str]
    ) -> dict[str, list[tuple[float, dict[str, Any]]]]:
        """按文档聚合 BM25 chunk 得分（调用方需持锁）。"""
        hits_by_doc: dict[str, list[tuple[float, dict[str, Any]]]] = {}
        if self._bm25 is None or not tokens:
            return hits_by_doc
        scores = self._bm25.scores(tokens)
        for position, score in enumerate(scores):
            if score <= 0 or position >= len(self._chunk_index):
                continue
            doc_id, chunk = self._chunk_index[position]
            hits_by_doc.setdefault(doc_id, []).append((float(score), chunk))
        return hits_by_doc

    def _substring_hits(self, query: str) -> dict[str, list[tuple[float, dict[str, Any]]]]:
        """BM25 全零时退化为大小写不敏感的子串匹配。"""
        needle = query.casefold()
        hits_by_doc: dict[str, list[tuple[float, dict[str, Any]]]] = {}
        if not needle:
            return hits_by_doc
        for doc_id, chunks in self._chunks.items():
            for chunk in chunks:
                haystack = str(chunk.get("text") or "").casefold()
                if needle in haystack:
                    hits_by_doc.setdefault(doc_id, []).append((1.0, chunk))
        return hits_by_doc

    def context(
        self, query: str, max_chars: int = 6000, limit: int = 5
    ) -> tuple[str, list[dict[str, Any]]]:
        """构造 RAG 上下文与引用列表。

        ``context_text`` 每段形如 ``[1] 《标题》 p.3:\\n正文``，总长不超过
        ``max_chars``；``citations`` 与编号一一对应。
        """
        budget = max(200, int(max_chars))
        results = self.search(query, limit=limit)
        parts: list[str] = []
        citations: list[dict[str, Any]] = []
        used = 0
        for result in results:
            for snippet in result.get("snippets") or []:
                text = str(snippet.get("text") or "").strip()
                if not text:
                    continue
                index = len(citations) + 1
                header = f"[{index}] \u300a{result['title']}\u300b p.{snippet['page']}:\n"
                reserved = len(header) + (2 if parts else 0)
                remaining = budget - used - reserved
                if remaining < 80:
                    return "\n\n".join(parts), citations
                body = text if len(text) <= remaining else text[:remaining].rstrip()
                part = f"{header}{body}"
                parts.append(part)
                used += len(part) + (2 if len(parts) > 1 else 0)
                citations.append(
                    {
                        "index": index,
                        "doc_id": result["id"],
                        "title": result["title"],
                        "page": int(snippet["page"]),
                        "quote": body,
                    }
                )
                if used >= budget:
                    return "\n\n".join(parts), citations
        return "\n\n".join(parts), citations

    def stats(self) -> dict[str, Any]:
        """返回索引统计：总数、各类型数量、已解析数、字符数与根目录状态。"""
        with self._lock:
            docs = self._docs
            pdf_count = sum(1 for doc in docs if doc.doc_type == "pdf")
            md_count = sum(1 for doc in docs if doc.doc_type == "md")
            txt_count = sum(1 for doc in docs if doc.doc_type == "txt")
            parsed = sum(1 for doc in docs if doc.extracted)
            chars = sum(_as_int(self._meta.get(doc.id, {}).get("chars")) or 0 for doc in docs)
            roots = [
                {"label": label, "path": str(path), "exists": path.is_dir()}
                for label, path in self.roots
            ]
            return {
                "total": len(docs),
                "pdf": pdf_count,
                "md": md_count,
                "txt": txt_count,
                "parsed": parsed,
                "chars": chars,
                "roots": roots,
            }


def _make_snippet(text: str, query: str, tokens: Sequence[str], *, radius: int = 150) -> str:
    """截取查询词附近 ±radius 字的片段。"""
    compact = _collapse_ws(text)
    if not compact:
        return ""
    lowered = compact.casefold()
    needles: list[str] = []
    for candidate in [query.strip(), *tokens]:
        if candidate and candidate.casefold() not in needles:
            needles.append(candidate.casefold())
    for needle in needles:
        found = lowered.find(needle)
        if found < 0:
            continue
        start = max(0, found - radius)
        end = min(len(compact), found + len(needle) + radius)
        prefix = "\u2026" if start > 0 else ""
        suffix = "\u2026" if end < len(compact) else ""
        return f"{prefix}{compact[start:end].strip()}{suffix}"
    head = compact[: 2 * radius]
    return head + ("\u2026" if len(compact) > len(head) else "")


def _is_reusable(entry: Any, stat: Any, kind: str, force: bool) -> bool:
    """判断旧缓存条目能否直接复用（mtime/size 未变且有抽取缓存）。"""
    if force or not isinstance(entry, dict):
        return False
    if _as_int(entry.get("size")) != int(stat.st_size):
        return False
    stored_mtime = entry.get("mtime")
    try:
        if abs(float(stored_mtime) - float(stat.st_mtime)) > 1e-6:
            return False
    except (TypeError, ValueError):
        return False
    if kind == "pdf":
        extract = str(entry.get("extract") or "")
        if not extract:
            return False
        extract_path = Path(extract)
        if not extract_path.is_file() or extract_path.stat().st_size <= 0:
            return False
    return True
