"""中英双语气象词扩展。

本地文献库里有英文原版 PDF 和中文精读/翻译文档，而用户的研究目标与提问通常是中文。
BM25 只能匹配同语言词元，因此这里提供一个轻量的领域词典：把中文查询扩展出英文
关键词，让「卸载」「星座」「时延」这类词也能命中英文论文。不引入任何模型或外部依赖。
"""

from __future__ import annotations

from collections.abc import Iterable

# 中文关键词 → 英文检索词（可多个，空格分隔）
BILINGUAL_TERMS: dict[str, str] = {
    "太空联邦学习": "space federated learning",
    "联邦学习": "federated learning",
    "卫星星座": "satellite constellation",
    "卫星": "satellite satellites",
    "星座": "constellation",
    "低轨": "low earth orbit LEO",
    "低地球轨道": "LEO low earth orbit",
    "轨道": "orbit orbital",
    "地面站": "ground station",
    "星间链路": "inter-satellite link ISL",
    "星地链路": "space-to-ground link",
    "链路": "link",
    "通信": "communication",
    "调度": "scheduling schedule",
    "数据卸载": "data offloading offload",
    "卸载": "offloading offload",
    "聚合": "aggregation aggregate",
    "去中心化": "decentralized",
    "边缘计算": "edge computing",
    "星上计算": "onboard computing in-orbit computing",
    "时延": "delay latency",
    "延迟": "delay latency",
    "收敛": "convergence converge",
    "准确率": "accuracy",
    "精度": "accuracy precision",
    "数据均衡": "data balance",
    "非独立同分布": "non-IID",
    "非iid": "non-IID",
    "异构": "heterogeneous",
    "鲁棒": "robust robustness",
    "资源分配": "resource allocation",
    "能耗": "energy consumption power",
    "能量": "energy",
    "通信开销": "communication overhead",
    "压缩": "compression",
    "移动性": "mobility",
    "覆盖": "coverage",
    "可见性": "visibility",
    "多层聚合": "hierarchical aggregation",
    "分层聚合": "hierarchical aggregation",
    "多址": "multiple access",
    "功率控制": "power control",
    "强化学习": "reinforcement learning",
    "图神经网络": "graph neural network",
    "遥感": "remote sensing",
    "数据集": "dataset",
    "实验": "experiment",
    "基线": "baseline",
    "对比": "comparison",
}

# 英文关键词 → 中文检索词（反向补充，用于中文问题里夹英文术语的场景）
_REVERSE: dict[str, str] = {
    "offloading": "卸载",
    "federated": "联邦",
    "satellite": "卫星",
    "constellation": "星座",
    "scheduling": "调度",
    "latency": "时延",
    "accuracy": "准确率",
    "hierarchical": "分层",
}


def _contains_term(haystack: str, term: str) -> bool:
    return term.lower() in haystack


def expand_query(text: str) -> str:
    """把中文查询扩展为「原文 + 英文领域词」的检索串。"""
    source = str(text or "")
    lowered = source.lower()
    additions: list[str] = []
    for chinese, english in BILINGUAL_TERMS.items():
        if chinese.lower() in lowered and not _contains_term(lowered, english.split()[0]):
            additions.append(english)
    if not additions:
        return source
    return f"{source} {' '.join(additions)}"


def expand_text(text: str) -> str:
    """把任意中文文本中的关键词扩展成中英混合文本（用于相关度计算）。"""
    return expand_query(text)


def suggested_english_keywords(text: str, limit: int = 8) -> list[str]:
    """返回命中的英文关键词（去重，供情报检索/调试展示）。"""
    lowered = str(text or "").lower()
    keywords: list[str] = []
    for chinese, english in BILINGUAL_TERMS.items():
        if chinese.lower() in lowered:
            for word in english.split():
                if word not in keywords:
                    keywords.append(word)
    return keywords[:limit]


def reverse_terms(english_text: str) -> Iterable[str]:
    """英文文本命中的中文关键词（辅助情报打标）。"""
    lowered = str(english_text or "").lower()
    return [chinese for english, chinese in _REVERSE.items() if english in lowered]


__all__ = [
    "BILINGUAL_TERMS",
    "expand_query",
    "expand_text",
    "reverse_terms",
    "suggested_english_keywords",
]
