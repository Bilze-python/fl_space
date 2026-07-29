"""
缓存体系多级复用优化模块 (论文维度六)
======================================

实现:
1. 轨道基础参数持久化缓存 (Redis / 本地文件)
2. 过境窗口增量缓存更新 (仅替换近域热窗口)
3. 星站匹配结果缓存 (复用历史打分权重)

与前文 precomputation_cache 互补:
    precomputation_cache: 静态参数预计算 + 冷热分层窗口缓存
    multi_tier_cache:     Redis 持久化 + 增量更新 + 匹配复用
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# ============================================================
# 1. 轨道基础参数持久化缓存 (文件/内存)
# ============================================================

@dataclass
class PersistentCacheEntry:
    """持久化缓存项。"""

    key: str
    data: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    ttl: float = 0.0  # 0 = 永不过期


class MultiplexCacheStore:
    """多级缓存存储 (内存 + JSON 文件持久化)。

    替代 Redis: 内存字典 + JSON 文件, 重启后从文件恢复。
    实际部署可替换为 Redis 直连。
    """

    def __init__(self, cache_dir: str = "./cache", use_redis: bool = False):
        self._dir = cache_dir
        self._use_redis = use_redis
        self._mem: dict[str, PersistentCacheEntry] = {}
        os.makedirs(cache_dir, exist_ok=True)
        self._load_from_disk()

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self._mem.get(key)
        if entry is None:
            return None
        # TTL 检查
        if entry.ttl > 0:
            import time
            if time.time() > entry.ttl:
                del self._mem[key]
                return None
        return entry.data

    def set(self, key: str, data: dict[str, Any], ttl_seconds: float = 0.0) -> None:
        import time
        ttl = time.time() + ttl_seconds if ttl_seconds > 0 else 0.0
        self._mem[key] = PersistentCacheEntry(key=key, data=data, ttl=ttl)
        self._save_to_disk()

    def delete(self, key: str) -> None:
        self._mem.pop(key, None)

    def _save_to_disk(self) -> None:
        if self._use_redis:
            return  # Redis 持久化由 Redis 自身管理
        payload = {
            key: {"data": e.data, "version": e.version, "ttl": e.ttl}
            for key, e in self._mem.items()
        }
        filepath = os.path.join(self._dir, "persistent_cache.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _load_from_disk(self) -> None:
        filepath = os.path.join(self._dir, "persistent_cache.json")
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, encoding="utf-8") as f:
                payload = json.load(f)
            for key, val in payload.items():
                self._mem[key] = PersistentCacheEntry(
                    key=key,
                    data=val.get("data", {}),
                    version=val.get("version", 1),
                    ttl=val.get("ttl", 0.0),
                )
        except (json.JSONDecodeError, OSError):
            pass


def cache_tle_data(
    store: MultiplexCacheStore,
    sat_id: int,
    tle_line1: str,
    tle_line2: str,
    ttl_hours: float = 24.0,
) -> None:
    """缓存 TLE 数据到持久化存储。"""
    store.set(
        f"tle:sat:{sat_id}",
        {"line1": tle_line1, "line2": tle_line2},
        ttl_seconds=ttl_hours * 3600.0,
    )


def cache_gs_params(
    store: MultiplexCacheStore,
    gs_id: int,
    lat_deg: float,
    lon_deg: float,
    alt_km: float,
    min_el_deg: float,
    max_concurrent: int,
) -> None:
    """缓存地面站静态参数。"""
    store.set(
        f"gs:params:{gs_id}",
        {
            "lat_deg": lat_deg,
            "lon_deg": lon_deg,
            "alt_km": alt_km,
            "min_el_deg": min_el_deg,
            "max_concurrent": max_concurrent,
        },
        ttl_seconds=720 * 3600.0,  # 30 天
    )


# ============================================================
# 2. 过境窗口增量缓存更新
# ============================================================

@dataclass
class IncrementalWindowCache:
    """过境窗口增量缓存。

    策略:
        - 未到当前时刻的历史窗口直接删除
        - 未变轨卫星的远期窗口仅增量修正, 不全部重算
        - 仅替换 0-6h 近域高精度窗口
    """

    # 时间基准: "current_timeslot" 推进
    current_ts: int = 0
    # {sat_id: {gs_id: [(ts_start, ts_end, avg_el, max_el, est_mb), ...]}}
    hot_windows: dict[int, dict[int, list[tuple]]] = field(default_factory=dict)
    cold_windows: dict[int, dict[int, list[tuple]]] = field(default_factory=dict)
    # 变轨标记
    maneuvered_sats: set[int] = field(default_factory=set)
    # 热窗口 horizon (timeslots)
    hot_horizon_slots: int = 360   # 6h @ 1min/slot
    cold_horizon_slots: int = 2880  # 48h

    def advance_time(self, new_current_ts: int) -> None:
        """推进时间窗口, 删除过期历史窗口, cold -> hot 降级。"""
        # 删除 t < new_current_ts 的历史窗口
        for sat_wins in self.hot_windows.values():
            for gs_id in list(sat_wins.keys()):
                sat_wins[gs_id] = [
                    w for w in sat_wins[gs_id] if w[1] >= new_current_ts
                ]
                if not sat_wins[gs_id]:
                    del sat_wins[gs_id]

        for sat_wins in self.cold_windows.values():
            for gs_id in list(sat_wins.keys()):
                sat_wins[gs_id] = [
                    w for w in sat_wins[gs_id] if w[1] >= new_current_ts
                ]
                if not sat_wins[gs_id]:
                    del sat_wins[gs_id]

        # cold -> hot 降级: cold 窗口中 ts_start < hot_horizon 的移到 hot
        hot_bound = new_current_ts + self.hot_horizon_slots
        for sat_id, sat_wins in list(self.cold_windows.items()):
            for gs_id, wins in list(sat_wins.items()):
                hot_ready = [w for w in wins if w[0] <= hot_bound]
                remaining = [w for w in wins if w[0] > hot_bound]
                if hot_ready:
                    self.hot_windows.setdefault(sat_id, {}).setdefault(gs_id, []).extend(hot_ready)
                if remaining:
                    sat_wins[gs_id] = remaining
                else:
                    del sat_wins[gs_id]
            if not sat_wins:
                del self.cold_windows[sat_id]

        self.current_ts = new_current_ts

    def invalidate_sat(self, sat_id: int) -> None:
        """变轨后失效该卫星的全部缓存。"""
        self.maneuvered_sats.add(sat_id)
        self.hot_windows.pop(sat_id, None)
        self.cold_windows.pop(sat_id, None)

    def update_hot_window(
        self,
        sat_id: int,
        gs_id: int,
        ts_start: int,
        ts_end: int,
        avg_el: float,
        max_el: float,
        est_mb: float,
    ) -> None:
        """更新热窗口 (0-6h 近域高精度)。"""
        if sat_id in self.maneuvered_sats:
            self.maneuvered_sats.discard(sat_id)
        entry = (ts_start, ts_end, avg_el, max_el, est_mb)
        sat_wins = self.hot_windows.setdefault(sat_id, {})
        gs_wins = sat_wins.setdefault(gs_id, [])
        # 去重替换
        gs_wins[:] = [w for w in gs_wins if w[0] != ts_start]
        gs_wins.append(entry)
        gs_wins.sort(key=lambda w: w[0])

    def is_cached_valid(self, sat_id: int) -> bool:
        return sat_id not in self.maneuvered_sats


# ============================================================
# 3. 星站匹配结果缓存
# ============================================================

@dataclass
class MatchScoreCache:
    """星站匹配打分缓存。

    同一卫星+同一地面站在多天内过境模式高度相似,
    缓存历史匹配打分结果, 新推演时直接复用。
    """

    # {(sat_id, gs_id): (avg_score, count, last_update_ts)}
    _scores: dict[tuple[int, int], tuple[float, int, int]] = field(default_factory=dict)
    _decay_factor: float = 0.95  # 衰减因子

    def get_score(self, sat_id: int, gs_id: int) -> float | None:
        """获取缓存的匹配得分 (指数衰减)。"""
        key = (sat_id, gs_id)
        entry = self._scores.get(key)
        if entry is None:
            return None
        avg_score, count, _last_ts = entry
        # 衰减: 越久远的得分越低
        decayed = avg_score * (self._decay_factor ** max(0, count - 1))
        return round(decayed, 4)

    def update_score(
        self,
        sat_id: int,
        gs_id: int,
        new_score: float,
        current_ts: int,
    ) -> None:
        """更新匹配得分 (增量 EMA 平滑)。"""
        key = (sat_id, gs_id)
        if key in self._scores:
            avg_score, count, _last = self._scores[key]
            # EMA: new_avg = alpha*new + (1-alpha)*avg
            alpha = 0.3
            updated = alpha * new_score + (1 - alpha) * avg_score
            self._scores[key] = (updated, count + 1, current_ts)
        else:
            self._scores[key] = (new_score, 1, current_ts)

    def clear_expired(self, max_age_slots: int, current_ts: int) -> None:
        """清除超过最大时效的缓存。"""
        expired = [
            k for k, (_s, _c, ts) in self._scores.items()
            if current_ts - ts > max_age_slots
        ]
        for k in expired:
            del self._scores[k]
