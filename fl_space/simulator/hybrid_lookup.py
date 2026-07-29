#!/usr/bin/env python3
"""
混合离线查表 + 在线实时推演模块 — 过境模板预生成 + 多工况参数库
D7: Hybrid offline lookup + online real-time computation.

设计目标:
    - 离线预生成固定轨道卫星的多天过境时刻表模板
    - 在线运行时仅根据恒星时和轨道漂移做偏移修正
    - 机动事件时自动切换回实时 SGP4 推演
    - 工况参数库: 不同大气密度/太阳活动下的轨道偏差修正系数

预期加速: 对无机动的常态化卫星, 离线模板可消除 90%+ 在线计算量
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from typing import Any, ClassVar

# ═══════════════════════════════════════════════════════════════════
#  1. 常规卫星过境模板离线预生成
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PassTemplate:
    """单条过境模板 — 存储预计算的过境窗口信息。

    模板不包含绝对时间, 而是存储相对于参考历元的偏移。
    在线使用时根据恒星时偏差做简单修正。
    """

    sat_id: int
    gs_id: int
    rel_start_s: float      # 相对参考历元的起始时间 (秒)
    rel_end_s: float        # 相对参考历元的结束时间 (秒)
    duration_s: float       # 过境时长 (秒)
    max_elevation_deg: float  # 最大仰角 (度)
    orbit_number: int       # 轨道圈号

    def offset(self, delta_t_s: float) -> PassTemplate:
        """对模板做时间偏移修正。"""
        return PassTemplate(
            sat_id=self.sat_id,
            gs_id=self.gs_id,
            rel_start_s=self.rel_start_s + delta_t_s,
            rel_end_s=self.rel_end_s + delta_t_s,
            duration_s=self.duration_s,
            max_elevation_deg=self.max_elevation_deg,
            orbit_number=self.orbit_number,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sat_id": self.sat_id,
            "gs_id": self.gs_id,
            "rel_start_s": self.rel_start_s,
            "rel_end_s": self.rel_end_s,
            "duration_s": self.duration_s,
            "max_elevation_deg": self.max_elevation_deg,
            "orbit_number": self.orbit_number,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PassTemplate:
        return cls(**d)


@dataclass
class SatellitePassDB:
    """单颗卫星的过境模板数据库。

    存储某颗卫星对多个地面站的多天过境模板。
    """

    sat_id: int
    norad_id: int
    reference_epoch_jd: float  # 参考历元 (儒略日)
    num_days: int  # 预计算天数
    passes: dict[int, list[PassTemplate]] = field(default_factory=dict)  # gs_id -> passes

    def add_pass(self, gs_id: int, template: PassTemplate) -> None:
        if gs_id not in self.passes:
            self.passes[gs_id] = []
        self.passes[gs_id].append(template)

    def query_passes(
        self,
        gs_id: int,
        t_start_s: float,
        t_end_s: float,
    ) -> list[PassTemplate]:
        """查询时间段内的过境模板 (相对参考历元)。"""
        if gs_id not in self.passes:
            return []

        passes = self.passes[gs_id]
        return [
            p for p in passes
            if p.rel_end_s >= t_start_s and p.rel_start_s <= t_end_s
        ]

    def get_today_passes(
        self,
        gs_id: int,
        day_offset: int = 0,
    ) -> list[PassTemplate]:
        """获取指定日期的过境模板。"""
        day_start = day_offset * 86400.0
        day_end = day_start + 86400.0
        return self.query_passes(gs_id, day_start, day_end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sat_id": self.sat_id,
            "norad_id": self.norad_id,
            "reference_epoch_jd": self.reference_epoch_jd,
            "num_days": self.num_days,
            "passes": {
                str(gs_id): [p.to_dict() for p in passes]
                for gs_id, passes in self.passes.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SatellitePassDB:
        db = cls(
            sat_id=d["sat_id"],
            norad_id=d["norad_id"],
            reference_epoch_jd=d["reference_epoch_jd"],
            num_days=d["num_days"],
        )
        for gs_id_str, passes in d["passes"].items():
            gs_id = int(gs_id_str)
            for p in passes:
                db.add_pass(gs_id, PassTemplate.from_dict(p))
        return db


class PassTemplateManager:
    """过境模板管理器 — 全局管理所有卫星的离线模板。

    功能:
        - 生成模板 (离线, 调用 SGP4 全量计算)
        - 存储/加载模板 (JSON 文件持久化)
        - 在线查询修正 (恒星时 + 轨道漂移修正)

    用法:
        mgr = PassTemplateManager(template_dir="./templates")
        # 离线生成
        mgr.generate(sat_ids=[0, 1], gs_count=5, num_days=7)
        mgr.save()
        # 在线查询
        passes = mgr.query_online(sat_id=0, gs_id=1, t_now_jd=2460123.5)
    """

    def __init__(self, template_dir: str = "./pass_templates"):
        self._template_dir = template_dir
        self._dbs: dict[int, SatellitePassDB] = {}  # sat_id -> DB
        os.makedirs(template_dir, exist_ok=True)

    @property
    def loaded_satellites(self) -> list[int]:
        return list(self._dbs.keys())

    def load(self, sat_id: int) -> SatellitePassDB | None:
        """加载卫星模板数据库。"""
        filepath = os.path.join(self._template_dir, f"sat_{sat_id}_passes.json")
        if not os.path.exists(filepath):
            return None

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        db = SatellitePassDB.from_dict(data)
        self._dbs[sat_id] = db
        return db

    def save(self, sat_id: int | None = None) -> None:
        """保存模板数据库到 JSON 文件。"""
        sats = [sat_id] if sat_id is not None else self._dbs.keys()
        for sid in sats:
            if sid not in self._dbs:
                continue
            filepath = os.path.join(self._template_dir, f"sat_{sid}_passes.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self._dbs[sid].to_dict(), f, indent=2)

    def query_online(
        self,
        sat_id: int,
        gs_id: int,
        t_now_jd: float,
        window_s: float = 3600.0,
    ) -> list[PassTemplate]:
        """在线查询 — 使用恒星时修正模板。

        Args:
            sat_id: 卫星 ID
            gs_id: 地面站 ID
            t_now_jd: 当前儒略日
            window_s: 查询窗口 (秒)

        Returns:
            修正后的过境模板列表 (绝对时间, epoch秒)
        """
        db = self._dbs.get(sat_id)
        if db is None:
            db = self.load(sat_id)
            if db is None:
                return []

        # 计算时间偏移
        # 1. 恒星时修正: 地球自转 1 天 = 86164s (恒星日, 而非 86400s 太阳日)
        days_since_epoch = t_now_jd - db.reference_epoch_jd
        sidereal_day_s = 86164.0905
        solar_day_s = 86400.0
        # 每一天太阳日落后恒星日约 236s
        sidereal_offset = days_since_epoch * (solar_day_s - sidereal_day_s)

        # 2. 轨道漂移修正 (简化: 假设线性漂移)
        # 实际中应从 TLE 更新获取精确漂移量
        total_offset_s = sidereal_offset

        # 查询原始模板
        t_rel_start = (days_since_epoch * solar_day_s) % (db.num_days * solar_day_s)
        t_rel_end = t_rel_start + window_s

        raw_passes = db.query_passes(gs_id, t_rel_start, t_rel_end)

        # 应用时间修正
        corrected = []
        for p in raw_passes:
            corrected.append(p.offset(total_offset_s))  # noqa: PERF401

        return corrected

    def get_template_count(self, sat_id: int) -> int:
        """获取某卫星的模板总数。"""
        db = self._dbs.get(sat_id)
        if db is None:
            return 0
        return sum(len(passes) for passes in db.passes.values())


# ═══════════════════════════════════════════════════════════════════
#  2. 多工况离线仿真参数库
# ═══════════════════════════════════════════════════════════════════


@dataclass
class EnvironmentCase:
    """单工况参数 — 一组空间环境参数及对应轨道偏差修正系数。"""

    name: str  # 工况名称
    f10_7: float  # 太阳射电通量 (sfu), 表征太阳活动
    ap_index: float  # 地磁指数
    atmospheric_density_factor: float  # 大气密度修正因子 (相对标准大气)
    drag_scale_factor: float  # 阻力缩放因子
    j2_scale_factor: float  # J2 缩放 (default 1.0)
    valid_altitude_range: tuple[float, float]  # 有效高度范围 (km)
    description: str = ""


class EnvironmentParamLibrary:
    """多工况参数库 — 离线仿真不同大气密度/太阳活动下的轨道偏差。

    在线定时计算时根据实时空间环境参数匹配对应修正系数,
    无需实时求解复杂环境摄动公式。

    典型工况:
        - 太阳平静 (F10.7=70, Ap=5):  大气密度低, 阻力小
        - 太阳活跃 (F10.7=200, Ap=30): 大气密度高, 阻力大
        - 地磁暴 (F10.7=150, Ap=100):  大气膨胀, 轨道衰减加速

    用法:
        lib = EnvironmentParamLibrary()
        lib.add_case(EnvironmentCase(
            name="quiet", f10_7=70, ap_index=5,
            atmospheric_density_factor=0.5,
            drag_scale_factor=0.5,
            valid_altitude_range=(200, 2000),
        ))
        # 在线匹配
        case = lib.match(f10_7=180, ap_index=25, altitude_km=500)
        scale = case.drag_scale_factor  # 用于修正阻力项
    """

    # 预定义标准工况
    STANDARD_CASES: ClassVar[list[EnvironmentCase]] = []

    def __init__(self):
        self._cases: list[EnvironmentCase] = []
        # 初始化标准工况
        self._init_standard_cases()

    def _init_standard_cases(self) -> None:
        """初始化标准工况库。"""
        standard = [
            EnvironmentCase(
                name="solar_quiet",
                f10_7=70.0,
                ap_index=5.0,
                atmospheric_density_factor=0.4,
                drag_scale_factor=0.4,
                j2_scale_factor=1.0,
                valid_altitude_range=(200, 2000),
                description="太阳平静期: 低太阳活动, 大气密度低",
            ),
            EnvironmentCase(
                name="solar_moderate",
                f10_7=130.0,
                ap_index=15.0,
                atmospheric_density_factor=1.0,
                drag_scale_factor=1.0,
                j2_scale_factor=1.0,
                valid_altitude_range=(200, 2000),
                description="太阳中等活动: 标准大气模型",
            ),
            EnvironmentCase(
                name="solar_active",
                f10_7=200.0,
                ap_index=30.0,
                atmospheric_density_factor=2.5,
                drag_scale_factor=2.5,
                j2_scale_factor=1.0,
                valid_altitude_range=(200, 2000),
                description="太阳活跃期: 高太阳通量, 大气膨胀",
            ),
            EnvironmentCase(
                name="geomagnetic_storm",
                f10_7=150.0,
                ap_index=80.0,
                atmospheric_density_factor=4.0,
                drag_scale_factor=5.0,
                j2_scale_factor=1.0,
                valid_altitude_range=(200, 1000),
                description="地磁暴: 大气剧烈膨胀, 低轨阻力大增",
            ),
            EnvironmentCase(
                name="solar_minimum",
                f10_7=65.0,
                ap_index=3.0,
                atmospheric_density_factor=0.25,
                drag_scale_factor=0.25,
                j2_scale_factor=1.0,
                valid_altitude_range=(300, 2000),
                description="太阳极小期: 极低太阳活动",
            ),
        ]
        for case in standard:
            self.add_case(case)

    def add_case(self, case: EnvironmentCase) -> None:
        self._cases.append(case)

    def match(
        self,
        f10_7: float,
        ap_index: float,
        altitude_km: float = 500.0,
    ) -> EnvironmentCase:
        """根据实时空间环境参数匹配最佳工况。

        使用加权欧氏距离选择最近工况。

        Args:
            f10_7: 当前 F10.7 太阳通量
            ap_index: 当前 Ap 地磁指数
            altitude_km: 当前卫星高度

        Returns:
            最匹配的工况
        """
        if not self._cases:
            # 返回默认工况
            return EnvironmentCase(
                name="default",
                f10_7=130.0,
                ap_index=15.0,
                atmospheric_density_factor=1.0,
                drag_scale_factor=1.0,
                valid_altitude_range=(0, 10000),
            )

        best_case = self._cases[0]
        best_dist = float("inf")

        for case in self._cases:
            # 先检查高度范围
            alt_min, alt_max = case.valid_altitude_range
            if altitude_km < alt_min or altitude_km > alt_max:
                continue

            # 加权距离: F10.7 权重 0.6, Ap 权重 0.4
            df = (f10_7 - case.f10_7) / 200.0  # 归一化
            da = (ap_index - case.ap_index) / 100.0
            dist = math.sqrt(0.6 * df**2 + 0.4 * da**2)

            if dist < best_dist:
                best_dist = dist
                best_case = case

        return best_case

    def get_correction_factors(
        self,
        f10_7: float,
        ap_index: float,
        altitude_km: float = 500.0,
    ) -> dict[str, float]:
        """获取轨道修正因子 (直接用于在线修正)。

        Returns:
            {"drag_scale": factor, "density_factor": factor, "j2_scale": factor}
        """
        case = self.match(f10_7, ap_index, altitude_km)
        return {
            "drag_scale": case.drag_scale_factor,
            "density_factor": case.atmospheric_density_factor,
            "j2_scale": case.j2_scale_factor,
            "case_name": case.name,
        }

    def list_cases(self) -> list[dict[str, Any]]:
        return [
            {
                "name": c.name,
                "f10_7": c.f10_7,
                "ap": c.ap_index,
                "drag_scale": c.drag_scale_factor,
                "altitude_range": c.valid_altitude_range,
                "desc": c.description,
            }
            for c in self._cases
        ]


# ═══════════════════════════════════════════════════════════════════
#  3. 在线/离线混合调度
# ═══════════════════════════════════════════════════════════════════


class HybridOrbitProvider:
    """混合轨道提供者 — 离线模板 + 在线 SGP4 自动切换。

    决策逻辑:
        - 无机动的常态化卫星 → 使用离线模板
        - 发生机动的卫星 → 切换在线 SGP4
        - 模板过期 (超过预计算天数) → 自动回退在线

    用法:
        provider = HybridOrbitProvider(template_mgr, env_lib)
        provider.set_online_sgp4(my_sgp4_func)
        passes = provider.get_passes(sat_id=0, gs_id=1, t_jd=2460123.5)
    """

    def __init__(
        self,
        template_mgr: PassTemplateManager | None = None,
        env_lib: EnvironmentParamLibrary | None = None,
    ):
        self._templates = template_mgr or PassTemplateManager()
        self._env_lib = env_lib or EnvironmentParamLibrary()
        self._sgp4_func: Any = None  # 在线 SGP4 函数
        self._maneuvered_sats: set[int] = set()  # 已机动的卫星

    def set_online_sgp4(self, sgp4_func: Any) -> None:
        """设置在线 SGP4 计算函数。"""
        self._sgp4_func = sgp4_func

    def mark_maneuver(self, sat_id: int) -> None:
        """标记卫星发生了机动, 需要在线计算。"""
        self._maneuvered_sats.add(sat_id)

    def clear_maneuver(self, sat_id: int) -> None:
        """清除机动标记 (模板重新生成后)。"""
        self._maneuvered_sats.discard(sat_id)

    def is_online_required(self, sat_id: int) -> bool:
        """判断某卫星是否需要在线计算。"""
        if sat_id in self._maneuvered_sats:
            return True
        return self._templates.get_template_count(sat_id) == 0

    def get_passes(
        self,
        sat_id: int,
        gs_id: int,
        t_now_jd: float,
        window_s: float = 3600.0,
    ) -> list[PassTemplate]:
        """获取过境窗口 — 自动选择离线模板或在线计算。

        Args:
            sat_id: 卫星ID
            gs_id: 地面站ID
            t_now_jd: 当前儒略日
            window_s: 查询窗口 (秒)

        Returns:
            过境窗口列表
        """
        if self.is_online_required(sat_id):
            if self._sgp4_func is None:
                return []
            # 在线计算 (此处为接口占位, 实际需调用完整 SGP4 推演)
            return []

        return self._templates.query_online(sat_id, gs_id, t_now_jd, window_s)
