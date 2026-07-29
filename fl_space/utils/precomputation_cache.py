"""
离线预计算 + 分层缓存优化模块（论文维度五）
==========================================

实现：
1. 静态参数离线预计算（地面站 ECEF/ENU 旋转矩阵/遮挡边界）
2. 窗口分层冷热缓存（热缓存 0-6h / 冷缓存 24-48h）
3. 增量更新（只刷新热缓存，冷缓存增量追加）

设计原则：
    - 地面站 ECEF、站心旋转矩阵、遮挡边界一次性离线算出后缓存
    - 每次定时演算直接读取缓存常量，无需重复坐标转换
    - 窗口分为热/冷两层，定时刷新只重算热缓存区间

Notes
-----
内存占用：48h 典型场景下约 50-200 MB（取决于站数/星数/采样率）。
"""

from __future__ import annotationsfrom dataclasses import dataclass, fieldimport jsonimport mathimport time# ============================================================
# 静态预计算类型
# ============================================================

@dataclass
class StaticCachedParams:
    """地面站静态预计算参数（一次性算出后不可变）。

    Fields
    ------
    ecef : tuple[float, float, float]
        WGS84 椭球下 ECEF 坐标 (km)。
    enu_matrix : tuple[tuple[float, ...], ...]
        站心 3x3 旋转矩阵 M（用于 ENU 转换），展平存为 9 元素 tuple。
    occlusion_mask : dict
        地形/建筑物遮挡查表 {azimuth_bin: min_elevation_deg}，默认空。
    """
    ecef: tuple[float, float, float] = (0.0, 0.0, 0.0)
    enu_matrix: tuple[float, ...] = field(default_factory=lambda: tuple(range(9)))
    occlusion_mask: dict[int, float] = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "ecef": list(self.ecef),
            "occlusion_mask": dict(self.occlusion_mask),
        }


# ============================================================
# 分层窗口缓存
# ============================================================

@dataclass
class CachedWindow:
    """缓存的过境窗口精简结构（不含完整仰角序列）。"""
    sat_id: int
    gs_id: int
    ts_start: int
    ts_end: int
    duration_slots: int
    avg_elevation_deg: float = 0.0
    max_elevation_deg: float = 0.0
    est_downlink_mb: float = 0.0
    conflict: bool = False
    conflict_level: int = 2

    @staticmethod
    def from_pass_record(r) -> CachedWindow:

        return CachedWindow(
            sat_id=r.sat_id,
            gs_id=r.gs_id,
            ts_start=r.ts_start,
            ts_end=r.ts_end,
            duration_slots=r.duration_slots,
            avg_elevation_deg=r.avg_elevation_deg,
            max_elevation_deg=r.max_elevation_deg,
            est_downlink_mb=r.est_downlink_mb,
            conflict=r.conflict,
            conflict_level=r.conflict_level,
        )


class HotColdWindowCache:
    """窗口分层冷热缓存管理器。

    - **热缓存**（0~6h 待执行窗口）：完整存储仰角序列、预估下行量、冲突标记
    - **冷缓存**（24~48h 远期窗口）：仅保存起止时间、总时长，简化存储
    - 定时刷新只重算热缓存区间，冷缓存增量更新

    Parameters
    ----------
    hot_horizon_slots : int
        热缓存时间范围（timeslot 数，约对应 6h）。
    cold_horizon_slots : int
        冷缓存时间范围（timeslot 数，约对应 24~48h）。
    timeslot_duration_min : float
        每个 timeslot 的分钟数。
    """

    def __init__(
        self,
        hot_horizon_slots: int = 360,
        cold_horizon_slots: int = 2880,
        timeslot_duration_min: float = 1.0,
    ):
        self._hot_horizon = hot_horizon_slots
        self._cold_horizon = cold_horizon_slots
        self._slot_dur = timeslot_duration_min

        # 热缓存：完整窗口 + 仰角序列
        self._hot_windows: dict[int, list[CachedWindow]] = {}  # {gs_id: [...]}
        self._hot_elevation_curves: dict[tuple[int, int, int], list[float]] = {}
        # key: (sat_id, gs_id, ts_start)

        # 冷缓存：精简窗口（仅起止时间+时长）
        self._cold_windows: dict[int, list[CachedWindow]] = {}

        # 时间戳
        self._last_hot_refresh = 0.0
        self._last_cold_refresh = 0.0

        # 统计
        self._hot_hits = 0
        self._cold_hits = 0
        self._misses = 0

    # ---- 缓存查询 ----

    def get_hot_window(
        self, gs_id: int, ts: int
    ) -> CachedWindow | None:
        """在热缓存中查找某时刻某地面站的窗口。"""
        for w in self._hot_windows.get(gs_id, []):
            if w.ts_start <= ts <= w.ts_end:
                self._hot_hits += 1
                return w
        self._misses += 1
        return None

    def get_cold_window(
        self, gs_id: int, ts: int
    ) -> CachedWindow | None:
        """在冷缓存中查找。"""
        for w in self._cold_windows.get(gs_id, []):
            if w.ts_start <= ts <= w.ts_end:
                self._cold_hits += 1
                return w
        self._misses += 1
        return None

    def get_elevation_curve(
        self, sat_id: int, gs_id: int, ts_start: int
    ) -> list[float] | None:
        return self._hot_elevation_curves.get((sat_id, gs_id, ts_start))

    # ---- 缓存写入 ----

    def set_hot_windows(
        self, gs_id: int, windows: list[CachedWindow],
        elevation_curves: dict[tuple[int, int, int], list[float]] | None = None,
    ) -> None:
        self._hot_windows[gs_id] = windows
        if elevation_curves:
            self._hot_elevation_curves.update(elevation_curves)
        self._last_hot_refresh = time.time()

    def set_cold_windows(
        self, gs_id: int, windows: list[CachedWindow],
    ) -> None:
        self._cold_windows[gs_id] = windows
        self._last_cold_refresh = time.time()

    # ---- 增量刷新 ----

    def shift_windows(self, advance_slots: int) -> None:
        """时间推进：热缓存在前移，冷缓存中的窗口降级到热缓存。

        当时间推进 advance_slots 后：
        1. 热缓存中 ts_start < advance_slots 的过期窗口移出
        2. 冷缓存中 ts_start < hot_horizon + advance_slots 的窗口升级到热缓存
        """
        # 清理热缓存中已过期窗口
        for gs_id in list(self._hot_windows.keys()):
            self._hot_windows[gs_id] = [
                w for w in self._hot_windows[gs_id]
                if w.ts_end >= advance_slots
            ]
            # 清理对应的仰角曲线
            for key in list(self._hot_elevation_curves.keys()):
                _sat_id, gs_id_k, ts_start = key
                if ts_start < advance_slots and gs_id_k == gs_id:
                    del self._hot_elevation_curves[key]

        # 冷缓存中的窗口升级到热缓存
        hot_boundary = self._hot_horizon + advance_slots
        for gs_id in list(self._cold_windows.keys()):
            upgraded = []
            remaining = []
            for w in self._cold_windows[gs_id]:
                if w.ts_start <= hot_boundary:
                    upgraded.append(w)
                else:
                    remaining.append(w)
            if upgraded:
                self._hot_windows.setdefault(gs_id, []).extend(upgraded)
            self._cold_windows[gs_id] = remaining

    def invalidate_sat(self, sat_id: int) -> None:
        """使某卫星的所有缓存失效（变轨时调用）。"""
        for gs_id in list(self._hot_windows.keys()):
            self._hot_windows[gs_id] = [
                w for w in self._hot_windows[gs_id] if w.sat_id != sat_id
            ]
        for gs_id in list(self._cold_windows.keys()):
            self._cold_windows[gs_id] = [
                w for w in self._cold_windows[gs_id] if w.sat_id != sat_id
            ]
        for key in list(self._hot_elevation_curves.keys()):
            if key[0] == sat_id:
                del self._hot_elevation_curves[key]

    # ---- 统计 ----

    @property
    def stats(self) -> dict:
        total = self._hot_hits + self._cold_hits + self._misses
        return {
            "hot_windows_count": sum(len(w) for w in self._hot_windows.values()),
            "cold_windows_count": sum(len(w) for w in self._cold_windows.values()),
            "hot_hits": self._hot_hits,
            "cold_hits": self._cold_hits,
            "misses": self._misses,
            "hit_ratio": (
                round((self._hot_hits + self._cold_hits) / max(total, 1), 4)
            ),
            "last_hot_refresh": round(self._last_hot_refresh, 1),
            "last_cold_refresh": round(self._last_cold_refresh, 1),
        }

    def clear(self) -> None:
        self._hot_windows.clear()
        self._cold_windows.clear()
        self._hot_elevation_curves.clear()
        self._hot_hits = self._cold_hits = self._misses = 0


# ============================================================
# 静态参数离线预计算工场
# ============================================================

def precompute_gs_static_params(
    lat_deg: float,
    lon_deg: float,
    alt_km: float = 0.0,
) -> StaticCachedParams:
    """离线预计算单个地面站的所有静态参数。

    一次性算出 ECEF 坐标 + ENU 旋转矩阵 + 遮挡边界查表，
    存入 StaticCachedParams 供后续直接读取。

    Returns
    -------
    StaticCachedParams
    """
    from fl_space.utils.coordinate_utils import geodetic_to_ecef

    ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_km)

    # ENU 旋转矩阵（展平为 9 元 tuple）
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    slat, clat = math.sin(lat), math.cos(lat)
    slon, clon = math.sin(lon), math.cos(lon)
    enu_flat = (
        -slon, clon, 0.0,
        -slat * clon, -slat * slon, clat,
        clat * clon, clat * slon, slat,
    )

    # 遮挡 mask（默认空，可由外部加载 DEM 数据填入）
    occlusion: dict[int, float] = {}

    return StaticCachedParams(
        ecef=ecef,
        enu_matrix=enu_flat,
        occlusion_mask=occlusion,
    )


def precompute_all_gs_params(
    gs_coords: list[tuple[float, float, float]],
) -> list[StaticCachedParams]:
    """批量预计算所有地面站静态参数。

    Parameters
    ----------
    gs_coords : list[tuple[float, float, float]]
        各站 [(lat_deg, lon_deg, alt_km), ...]。

    Returns
    -------
    list[StaticCachedParams]
    """
    return [precompute_gs_static_params(lat, lon, alt) for lat, lon, alt in gs_coords]


def save_cache_to_json(cache: HotColdWindowCache, filepath: str) -> None:
    """将冷热缓存序列化保存（用于持久化）。"""
    data = {
        "hot": {
            str(gs_id): [w.__dict__ for w in windows]
            for gs_id, windows in cache._hot_windows.items()
        },
        "cold": {
            str(gs_id): [w.__dict__ for w in windows]
            for gs_id, windows in cache._cold_windows.items()
        },
        "stats": cache.stats,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_cache_from_json(filepath: str) -> HotColdWindowCache | None:
    """从 JSON 文件恢复冷热缓存。"""
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    cache = HotColdWindowCache()
    for gs_str, windows in data.get("hot", {}).items():
        cache._hot_windows[int(gs_str)] = [
            CachedWindow(**w) for w in windows
        ]
    for gs_str, windows in data.get("cold", {}).items():
        cache._cold_windows[int(gs_str)] = [
            CachedWindow(**w) for w in windows
        ]
    return cache
