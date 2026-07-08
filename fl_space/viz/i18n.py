"""
SpaceFL 可视化国际化 (i18n) — 中英文翻译映射。

用法:
    from fl_space.viz.i18n import t

    label = t("Longitude (deg)", lang="zh")   # → "经度 (°)"
    label = t("Longitude (deg)", lang="en")   # → "Longitude (deg)"
"""

from __future__ import annotations

# ── CJK 字体配置 ─────────────────────────────────────────────────
_CJK_FONT_CONFIGURED = False

# 按优先级排列的 CJK 字体候选列表
_CJK_FONT_CANDIDATES = [
    "Microsoft YaHei",      # 微软雅黑 (Windows)
    "SimHei",               # 黑体 (Windows)
    "Noto Sans SC",         # Google 思源黑体 (跨平台)
    "WenQuanYi Micro Hei",  # 文泉驿微米黑 (Linux)
    "WenQuanYi Zen Hei",    # 文泉驿正黑 (Linux)
    "STHeiti",              # 华文黑体 (macOS)
    "Heiti SC",             # 黑体-简 (macOS)
    "PingFang SC",          # 苹方-简 (macOS)
    "Microsoft JhengHei",   # 微软正黑体 (Windows)
    "SimSun",               # 宋体 (Windows, fallback)
]


def setup_cjk_font() -> None:
    """配置 matplotlib 以支持中文 (CJK) 字体渲染。

    自动检测系统中可用的 CJK 字体并回退到默认字体。
    仅首次调用时生效，后续调用为 no-op。
    """
    global _CJK_FONT_CONFIGURED
    if _CJK_FONT_CONFIGURED:
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
    except ImportError:
        return

    # 查找第一个可用的 CJK 字体
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    selected = None
    for candidate in _CJK_FONT_CANDIDATES:
        if candidate in available_fonts:
            selected = candidate
            break

    if selected is None:
        # 回退：使用 font_manager 查找任意 CJK 字体
        for f in fm.fontManager.ttflist:
            if any(k in f.name.lower() for k in
                   ['cjk', 'hei', 'song', 'ming', 'kai', 'yahei',
                    'jhenghei', 'noto sans', 'wenquan', 'pingfang',
                    'stheit', 'heiti', 'simsun', 'fangsong', 'kaiti']):
                selected = f.name
                break

    if selected is None:
        _CJK_FONT_CONFIGURED = True
        return

    # 更新 rcParams
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [selected, "DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,  # 防止负号显示为方块
    })

    # 清除字体缓存以确保新设置生效
    fm._load_fontmanager(try_read_cache=False)

    _CJK_FONT_CONFIGURED = True


# ── 翻译字典: 英文 → 中文 ────────────────────────────────────────
_TRANSLATIONS: dict[str, str] = {
    # ── 坐标和地图 ──
    "Longitude (deg)": "经度 (°)",
    "Latitude (deg)": "纬度 (°)",
    "Longitude": "经度",
    "Latitude": "纬度",
    "Longitude (°)": "经度 (°)",
    "Latitude (°)": "纬度 (°)",
    "Time": "时间",
    "Satellite": "卫星",
    "Satellite ID": "卫星编号",
    "Satellites": "卫星数",
    "Ground Station": "地面站",
    "Ground Stations": "地面站数",
    "Ground Stations:": "地面站：",

    # ── 接触矩阵 ──
    "Contact Matrix": "接触矩阵",
    "Contact Matrix Heatmap": "接触矩阵热力图",
    "Contact Rate": "接触率",
    "Contact Rate:": "接触率：",
    "Connected Ground Stations": "已连接地面站数",
    "Connected GS": "已连接GS数",
    "Avg Contacts/slot": "平均每时隙接触数",
    "Avg Contacts/slot:": "平均每时隙接触：",
    "Contact Window Count": "接触窗口次数",
    "Total Contact Windows": "总接触窗口数",
    "Total Contacts per Ground Station": "每地面站总接触数",

    # ── 训练/准确率 ──
    "Round": "轮次",
    "Accuracy": "准确率",
    "Timeslot": "时隙",
    "Accuracy vs Rounds": "准确率 vs 轮次",
    "Accuracy vs Virtual Time": "准确率 vs 虚拟时间",
    "max=": "最大=",
    "90%": "90%",
    "90% threshold": "90% 阈值",
    "Completed Rounds": "已完成轮次",
    "Max Accuracy (FedAvg)": "最高准确率 (FedAvg)",

    # ── 时间分解 ──
    "Per-Round Time Breakdown": "每轮时间分解",
    "Wait Dist": "等待下发",
    "Download": "下载",
    "Train": "训练",
    "Wait Return": "等待回传",
    "Upload": "上传",

    # ── 轨道 ──
    "Altitude:": "轨道高度：",
    "Duration:": "模拟时长：",
    "Snapshot:": "快照时刻：",
    "Clusters:": "星簇：",
    "Orbit:": "轨道：",
    "X (km)": "X (公里)",
    "Y (km)": "Y (公里)",

    # ── 卫星连接 ──
    "Total Contact Time (minutes)": "总接触时长 (分钟)",
    "Contact Rate (%)": "接触率 (%)",
    "Per-Satellite Contact Duration": "每卫星接触时长",
    "Per-Satellite Contact Rate": "每卫星接触率",
    "Satellite Connectivity": "卫星连接性",

    # ── 星座 ──
    "Constellation Map": "星座地图",
    "SpaceFL Constellation": "SpaceFL 星座",
    "Ground Track": "星下点轨迹",
    "Ground Station Distribution": "地面站分布图",

    # ── 图表通用 ──
    "Frequency": "频次",
    "Accuracy Distribution": "准确率分布",
    "mean=": "均值=",
    "median=": "中位数=",

    # ── GS-SAT 分析 ──
    "GS-Satellite Contact Counts": "地面站-卫星接触次数",
    "GS-Satellite Contact Analysis": "地面站-卫星接触分析",
    "Per-Satellite Contact Duration": "每卫星接触时长",
    "Per-Satellite Contact Rate": "每卫星接触率",

    # ── 轨道剖面 ──
    "Orbit Cross-Section": "轨道剖面图",

    # ── 图例 ──
    "SpaceFL": "SpaceFL",
    "Standard FL": "标准联邦学习",

    # ── 地面站相关 ──
    "Ground Station Positions": "地面站位置",

    # ── 网格 ──
    "SpaceFL Grid Search Summary": "SpaceFL 网格搜索总览",

    # ── 星簇相关 ──
    "sats": "颗",
    "satellites evenly spaced": "颗卫星均匀分布",
    "alt": "高度",
    "incl": "倾角",

    # ── 通用 ──
    "final acc": "最终准确率",
    "max acc": "最高准确率",
}

# ── 模板翻译 (带占位符的格式化字符串) ──────────────────────────
_TEMPLATES: dict[str, str] = {
    # 接触率
    "Contact Rate: {rate:.1f}%": "接触率：{rate:.1f}%",
    "Contact: {rate:.1%}": "接触率：{rate:.1%}",

    # 准确率
    "Accuracy: SpaceFL vs Standard FL": "准确率：SpaceFL vs 标准联邦学习",
    "Accuracy: {name}": "准确率：{name}",
    "Time Breakdown: {name}": "时间分解：{name}",
    "Contact Heatmap (GS={gs}, {sats} Sats)": "接触热力图 (GS={gs}, {sats} 颗卫星)",
    "Ground Stations (GS={gs})": "地面站 (GS={gs})",

    # 轨道相关
    "Orbit: {alt}km alt, {incl}° incl\n{sats} satellites evenly spaced":
        "轨道：{alt}公里 高度, {incl}° 倾角\n{sats} 颗卫星均匀分布",

    # 联系热力图标题
    "Contact Matrix — GS={gs}, SAT={sat}, Rate={rate:.1%}":
        "接触矩阵 — GS={gs}, SAT={sat}, 接触率={rate:.1%}",

    # 卫星连接性
    "Satellite Connectivity — GS={gs}, {sats} Sats @ {alt}km":
        "卫星连接性 — GS={gs}, {sats} 颗卫星 @ {alt}公里",

    # GS-SAT
    "GS-Satellite Contact Counts (GS={gs}, SAT={sat})":
        "地面站-卫星接触次数 (GS={gs}, SAT={sat})",
    "GS-Satellite Contact Analysis — GS={gs}, SAT={sat}":
        "地面站-卫星接触分析 — GS={gs}, SAT={sat}",

    # 准确率
    "Accuracy vs Rounds (GS={gs}, SAT={sat})":
        "准确率 vs 轮次 (GS={gs}, SAT={sat})",
    "Accuracy vs Rounds (μ={mu}, GS={gs})":
        "准确率 vs 轮次 (μ={mu}, GS={gs})",

    # 轨道剖面
    "Orbit Cross-Section — GS={gs}, SAT={sat} @ {alt}km":
        "轨道剖面图 — GS={gs}, SAT={sat} @ {alt}公里",

    # 地面站位置
    "Ground Station Positions (GS={gs}) — Paper Table 3":
        "地面站位置 (GS={gs}) — 论文表3",

    # 网格汇总
    "SpaceFL Grid Search Summary — FedAvg, GS×SAT":
        "SpaceFL 网格搜索总览 — FedAvg, GS×SAT",

    # Dashboard 标题
    "SpaceFL — 星座与地面站": "SpaceFL — 星座与地面站",
    "SpaceFL — 接触矩阵热力图": "SpaceFL — 接触矩阵热力图",
    "SpaceFL — 星座概览": "SpaceFL — 星座概览",
    "SpaceFL — Quick View": "SpaceFL — 快速视图",

    # FedAvg 标题
    "SpaceFL FedAvg — GS={gs}, SAT={sat}, Final Acc={acc:.3f}":
        "SpaceFL FedAvg — GS={gs}, SAT={sat}, 最终准确率={acc:.3f}",

    # 时间轴
    "Time ({hours:.0f}h total, {slot_min}min/slot)":
        "时间 (总计{hours:.0f}小时, {slot_min}分钟/时隙)",
    "Time ({hours:.0f}h, {slot_min}min/slot)":
        "时间 ({hours:.0f}小时, {slot_min}分钟/时隙)",

    # GS图例
    "Ground Stations:\n{gs_text}": "地面站：\n{gs_text}",

    # FedProx 标题
    "SpaceFL FedProx — {gs}GS × {sats} Sats ({classes} classes/sat)":
        "SpaceFL FedProx — {gs}GS × {sats} 颗卫星 ({classes} 类/卫星)",

    # Demo 标题
    "Demo 1: Dual-Shell Constellation (Polar + LEO)":
        "演示1：双壳层星座 (极轨 + LEO)",
    "Demo 2: Custom Research Constellation (via Registry)":
        "演示2：自定义研究星座 (注册表)",
    "Demo 3: Mixed — Walker Shell + Custom Satellites (SSO + Equatorial)":
        "演示3：混合 — Walker壳层 + 自定义卫星 (SSO + 赤道)",
    "Demo 4: Constellation Comparison":
        "演示4：星座对比",
    "Demo 5: Ground Tracks with Contact Highlights":
        "演示5：星下点轨迹与接触高亮",

    # 卫星标签
    "SAT-{i} ({incl:.0f}deg incl.)": "SAT-{i} (倾角{incl:.0f}°)",
}


def t(text: str, lang: str = "en") -> str:
    """翻译文本。

    Parameters
    ----------
    text : str
        英文原文。
    lang : str
        目标语言，'en' 或 'zh'。

    Returns
    -------
    str
        翻译后的文本。如果找不到翻译则返回原文。
    """
    if lang == "en":
        return text
    if lang == "zh":
        return _TRANSLATIONS.get(text, text)
    return text


def tf(template: str, lang: str = "en", **kwargs) -> str:
    """翻译并格式化模板字符串。

    先查找模板的翻译，再用 kwargs 进行格式化。

    Parameters
    ----------
    template : str
        英文模板字符串。
    lang : str
        目标语言，'en' 或 'zh'。
    **kwargs
        格式化参数。

    Returns
    -------
    str
        翻译并格式化后的文本。
    """
    if lang == "en":
        fmt = template
    elif lang == "zh":
        fmt = _TEMPLATES.get(template, template)
    else:
        fmt = template
    return fmt.format(**kwargs)


def _format_or_return(template: str, lang: str, **kwargs) -> str:
    """内部辅助：先翻译模板，再格式化。如果 kwargs 为空则只翻译。"""
    if lang == "en":
        fmt = template
    elif lang == "zh":
        fmt = _TEMPLATES.get(template, _TRANSLATIONS.get(template, template))
    else:
        fmt = template

    if kwargs:
        return fmt.format(**kwargs)
    return fmt
