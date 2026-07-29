"""预判式预计算缓存 — 前瞻多步定时指标（不依赖分层缓存）。



提供：

    1. 前瞻多步定时预生成 — 未来 1/3/6 小时的 T_wait, E_max

    2. 窗口预警预标记 — 距接轨 < 5min 的卫星打上就绪标记

    3. 重访间隔聚类预判 — 按 30min/3h/9h 分组批量计算打分

    4. 调度打分预聚合 — 提前算完所有候选卫星分数，实时只排序

"""



from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass

class ForwardTimingEntry:

    """前瞻定时条目。



    Attributes

    ----------

    sat_id : int

    forecast_timeslot : int

    T_wait : int

    E_max : int

    in_window : bool

    window_remaining : int

    next_contact_slot : int

    """

    sat_id: int

    forecast_timeslot: int

    T_wait: int = -1

    E_max: int = 0

    in_window: bool = False

    window_remaining: int = 0

    next_contact_slot: int = -1





@dataclass

class WindowWarning:

    """窗口预警标记。"""

    sat_id: int

    timeslot: int

    remaining_min: float

    is_critical: bool = False

    ready_to_transmit: bool = True





@dataclass

class RevisitGroup:

    """重访间隔聚类组。"""

    group_name: str

    base_interval_min: float

    client_ids: list[int] = field(default_factory=list)

    shared_E_max: int = 5  # noqa: N815

    shared_mu: float = 0.01

    shared_tau: int = 5





class PredictiveTimingCache:

    """预判式预计算缓存 — 前瞻生成未来定时指标。



    三阶前瞻：短期 1h/细粒度, 中期 3h/中粒度, 长期 6h/粗粒度。

    """



    SHORT_FORECAST_SLOTS = 60

    MEDIUM_FORECAST_SLOTS = 180

    LONG_FORECAST_SLOTS = 360

    SHORT_GRANULARITY = 10

    MEDIUM_GRANULARITY = 30

    LONG_GRANULARITY = 60



    def __init__(

        self, scheduler: Any,

        timeslot_duration_min: float = 1.0,

        compute_time_per_epoch_min: float = 2.0,

        warning_threshold_min: float = 5.0,

    ):

        self._scheduler = scheduler

        self.timeslot_duration_min = timeslot_duration_min

        self.compute_time_per_epoch_min = compute_time_per_epoch_min

        self.warning_threshold_min = warning_threshold_min

        self.t_epoch_slots = max(1, int(compute_time_per_epoch_min / timeslot_duration_min))

        self.warning_slots = max(1, int(warning_threshold_min / timeslot_duration_min))

        self._forward_cache: dict[int, dict[int, ForwardTimingEntry]] = {}

        self._warning_cache: dict[int, list[WindowWarning]] = {}

        self._revisit_groups: dict[str, RevisitGroup] = {}

        self._score_cache: dict[int, list[tuple[int, float]]] = {}

        self._expire_before: int = -1

        self._num_sats = scheduler._sim.num_satellites



    def generate_forecast(self, current_ts: int) -> None:

        """生成从当前时刻起 1h/3h/6h 的前瞻定时指标。"""

        for offset in range(0, self.SHORT_FORECAST_SLOTS, self.SHORT_GRANULARITY):

            self._build_forecast_snapshot(current_ts + offset)

        for offset in range(self.SHORT_FORECAST_SLOTS, self.MEDIUM_FORECAST_SLOTS, self.MEDIUM_GRANULARITY):

            self._build_forecast_snapshot(current_ts + offset)

        for offset in range(self.MEDIUM_FORECAST_SLOTS, self.LONG_FORECAST_SLOTS, self.LONG_GRANULARITY):

            self._build_forecast_snapshot(current_ts + offset)

        self._expire_before = current_ts



    def _build_forecast_snapshot(self, forecast_ts: int) -> None:

        snapshot: dict[int, ForwardTimingEntry] = {}

        for sat_id in range(self._num_sats):

            T_wait = self._scheduler.get_timing_wait(sat_id, forecast_ts)  # noqa: N806

            T_window = self._scheduler.get_timing_window(sat_id, forecast_ts)  # noqa: N806



            if T_window > 0:

                usable = max(0, T_window - self.warning_slots)

                emax = usable // self.t_epoch_slots if self.t_epoch_slots > 0 else 1

                entry = ForwardTimingEntry(

                    sat_id=sat_id, forecast_timeslot=forecast_ts,

                    T_wait=-1, E_max=max(1, emax),

                    in_window=True, window_remaining=T_window,

                )

            elif T_wait > 0:

                emax = T_wait // self.t_epoch_slots if self.t_epoch_slots > 0 else 1

                entry = ForwardTimingEntry(

                    sat_id=sat_id, forecast_timeslot=forecast_ts,

                    T_wait=T_wait, E_max=max(1, emax),

                    next_contact_slot=forecast_ts + T_wait,

                )

            else:

                entry = ForwardTimingEntry(

                    sat_id=sat_id, forecast_timeslot=forecast_ts,

                    T_wait=999999, E_max=0,

                )

            snapshot[sat_id] = entry

        self._forward_cache[forecast_ts] = snapshot



    def query_forward(self, sat_id: int, target_ts: int) -> ForwardTimingEntry | None:

        """查询前瞻定时指标 O(1)。"""

        snapshot = self._forward_cache.get(target_ts)

        return snapshot.get(sat_id) if snapshot else None



    def query_nearest_forecast(self, sat_id: int, target_ts: int) -> ForwardTimingEntry | None:

        """查找最近的前瞻条目。"""

        keys = sorted(self._forward_cache.keys())

        if not keys:

            return None

        idx = next((i for i, k in enumerate(keys) if k >= target_ts), len(keys) - 1)

        return self._forward_cache[keys[idx]].get(sat_id)



    def generate_warnings(self, current_ts: int) -> list[WindowWarning]:

        """生成当前时刻的窗口预警列表。"""

        warnings: list[WindowWarning] = []

        for sat_id in range(self._num_sats):

            T_window = self._scheduler.get_timing_window(sat_id, current_ts)  # noqa: N806

            T_wait = self._scheduler.get_timing_wait(sat_id, current_ts)  # noqa: N806

            if T_window > 0:

                remaining_min = T_window * self.timeslot_duration_min

                if remaining_min <= self.warning_threshold_min:

                    warnings.append(WindowWarning(

                        sat_id=sat_id, timeslot=current_ts,

                        remaining_min=remaining_min,

                        is_critical=remaining_min <= 2.0,

                        ready_to_transmit=True,

                    ))

            elif T_wait > 0 and T_wait <= self.warning_slots:

                warnings.append(WindowWarning(

                    sat_id=sat_id, timeslot=current_ts,

                    remaining_min=T_wait * self.timeslot_duration_min,

                    ready_to_transmit=False,

                ))

        self._warning_cache[current_ts] = warnings

        return warnings



    def get_warnings(self, timeslot: int) -> list[WindowWarning]:

        """获取窗口预警（缓存优先）。"""

        return self._warning_cache.get(timeslot) or self.generate_warnings(timeslot)



    def classify_revisit_groups(self, current_ts: int) -> dict[str, RevisitGroup]:

        """按重访间隔聚类卫星: short(<1h) / medium(1-3h) / long(>3h)。"""

        groups = {

            "short": RevisitGroup("short", 30.0),

            "medium": RevisitGroup("medium", 120.0),

            "long": RevisitGroup("long", 360.0),

        }

        for sat_id in range(self._num_sats):

            entry = self.query_forward(sat_id, current_ts)

            if entry is None:

                T_wait_abs = 60  # noqa: N806

            else:

                T_wait_abs = max(entry.T_wait, 0) + max(entry.window_remaining, 0)  # noqa: N806

            wait_min = T_wait_abs * self.timeslot_duration_min

            if wait_min <= 60.0:

                groups["short"].client_ids.append(sat_id)

            elif wait_min <= 180.0:

                groups["medium"].client_ids.append(sat_id)

            else:

                groups["long"].client_ids.append(sat_id)



        for group in groups.values():

            if not group.client_ids:

                continue

            interval = group.base_interval_min

            group.shared_E_max = max(1, int(interval / self.compute_time_per_epoch_min))

            group.shared_mu = 0.01 * (30.0 / max(interval, 1.0))

            if interval > 180.0:

                group.shared_tau = 8

            elif interval > 60.0:

                group.shared_tau = 5

            else:

                group.shared_tau = 3



        self._revisit_groups = groups

        return groups



    def get_group_params(self, sat_id: int) -> dict[str, Any]:

        """获取卫星的重访分组参数。"""

        for name, group in self._revisit_groups.items():

            if sat_id in group.client_ids:

                return {"group": name, "E_max": group.shared_E_max,

                        "mu": group.shared_mu, "tau": group.shared_tau}

        return {"group": "unknown", "E_max": 5, "mu": 0.01, "tau": 5}



    def precompute_scores(

        self, current_ts: int, staleness_map: dict[int, int],

        alpha: float = 1.0, beta: float = 0.5,

    ) -> list[tuple[int, float]]:

        """预计算所有候选卫星调度得分: Score = alpha*T_wait + beta*Staleness。"""

        scores: list[tuple[int, float]] = []

        for sat_id in range(self._num_sats):

            entry = self.query_forward(sat_id, current_ts)

            if entry is None:

                T_wait = max(0, self._scheduler.get_timing_wait(sat_id, current_ts))  # noqa: N806

            else:

                T_wait = max(0, entry.T_wait)  # noqa: N806

            staleness = staleness_map.get(sat_id, 0)

            scores.append((sat_id, alpha * T_wait + beta * staleness))

        scores.sort(key=lambda x: x[1])

        self._score_cache[current_ts] = scores

        return scores



    def get_cached_scores(self, timeslot: int, top_k: int = -1) -> list[tuple[int, float]]:

        """获取缓存的调度得分。"""

        scores = self._score_cache.get(timeslot, [])

        return scores[:top_k] if top_k > 0 and len(scores) > top_k else scores



    def expire_old(self, current_ts: int) -> None:

        """淘汰过期数据。"""

        self._expire_before = current_ts

        for cache in [self._forward_cache, self._warning_cache, self._score_cache]:

            for k in [k for k in cache if k < current_ts]:

                del cache[k]



    def get_stats(self) -> dict[str, Any]:

        """获取预判式缓存统计。"""

        return {

            "forward_cache_entries": sum(len(v) for v in self._forward_cache.values()),

            "forward_timesnap_count": len(self._forward_cache),

            "warning_cache_size": len(self._warning_cache),

            "score_cache_size": len(self._score_cache),

            "revisit_groups": list(self._revisit_groups.keys()),

        }



    def clear(self) -> None:

        """清空所有缓存。"""

        self._forward_cache.clear()

        self._warning_cache.clear()

        self._revisit_groups.clear()

        self._score_cache.clear()

        self._expire_before = -1

