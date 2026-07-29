"""

定时缓存池 — 仿真实时滚动定时的核心缓存模块。



提供：二分查找 O(logN)、T_window/T_wait/E_max 预缓存、增量滚动更新、三层定时框架。

"""



from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass

class TimingMetrics:

    """单卫星当前时刻的定时指标缓存。



    Attributes

    ----------

    satellite_id : int

    current_timeslot : int

    T_window : int

        当前窗口剩余通信时长（timeslot数），-1 表示不在窗口中。

    T_wait : int

        本次窗口结束到下一次接轨的间隔（timeslot数），窗口中时为 -1。

    E_max : int

        基于 T_wait 与单 epoch 耗时预计算的最大可训练轮数。

    next_contact_slot : int

        下一次接轨窗口的起始 timeslot，-1 表示无后续接轨。

    next_contact_duration : int

        下一次接轨窗口的持续 timeslot 数。

    isl_relay_available : bool

    isl_relay_sat : int

    gs_weights : dict[int, float]

    """

    satellite_id: int = -1

    current_timeslot: int = 0

    T_window: int = -1

    T_wait: int = -1

    E_max: int = 0

    T_train_total: int = -1

    next_contact_slot: int = -1

    next_contact_duration: int = 0

    isl_relay_available: bool = False

    isl_relay_sat: int = -1

    gs_weights: dict[int, float] = field(default_factory=dict)

    cache_timestamp: float = 0.0



    def is_valid(self, current_ts: int) -> bool:

        return self.current_timeslot == current_ts



    def to_dict(self) -> dict[str, Any]:

        return {

            "satellite_id": self.satellite_id,

            "current_timeslot": self.current_timeslot,

            "T_window": self.T_window,

            "T_wait": self.T_wait,

            "E_max": self.E_max,

            "T_train_total": self.T_train_total,

            "next_contact_slot": self.next_contact_slot,

            "next_contact_duration": self.next_contact_duration,

            "isl_relay_available": self.isl_relay_available,

            "isl_relay_sat": self.isl_relay_sat,

        }





@dataclass

class ContactWindowSorted:

    """排序后的通信窗口（按起始 timeslot 有序）。"""

    sat_id: int

    starts: np.ndarray

    ends: np.ndarray

    gs_ids_list: list[list[int]]



    def __len__(self) -> int:

        return len(self.starts)





@dataclass

class LayeredTimingState:

    """多尺度分层定时框架状态。



    L1 天级：轨道周期、星簇ISL预计算

    L2 小时级：T_wait/E_max 缓存刷新

    L3 分钟级：窗口刷新、陈旧差值、任务优先级

    """

    layer1_interval_slots: int = 1440

    layer2_interval_slots: int = 60

    last_layer1_refresh: int = -1

    last_layer2_refresh: int = -1





class TimingCachePool:

    """定时缓存池 — 优化实时查询性能的核心组件。



    提供：二分查找定位当前通信窗口 O(logN)、T_window/T_wait/E_max 预缓存、

          增量滚动更新机制、多尺度分层定时框架接口。

    """



    def __init__(

        self, scheduler: Any,

        timeslot_duration_min: float = 1.0,

        compute_time_per_epoch_min: float = 2.0,

    ):

        self._scheduler = scheduler

        self.timeslot_duration_min = timeslot_duration_min

        self.compute_time_per_epoch_min = compute_time_per_epoch_min

        self.t_epoch_slots = max(1, int(compute_time_per_epoch_min / timeslot_duration_min))

        self._sorted_windows: dict[int, ContactWindowSorted] = {}

        self._cache: dict[int, TimingMetrics] = {}

        self._last_updated_ts: int = -1

        self._layered_state = LayeredTimingState()

        self._build_sorted_windows()



    def _build_sorted_windows(self) -> None:

        for sat_id in range(self._scheduler._sim.num_satellites):

            windows = self._scheduler.get_windows(sat_id)

            if not windows:

                self._sorted_windows[sat_id] = ContactWindowSorted(

                    sat_id=sat_id, starts=np.array([], dtype=int),

                    ends=np.array([], dtype=int), gs_ids_list=[],

                )

                continue

            n = len(windows)

            starts = np.zeros(n, dtype=int)

            ends = np.zeros(n, dtype=int)

            gs_ids_list: list[list[int]] = []

            for i, w in enumerate(windows):

                starts[i] = w.timeslot_start

                ends[i] = w.timeslot_end

                gs_ids_list.append(list(w.gs_ids))

            self._sorted_windows[sat_id] = ContactWindowSorted(

                sat_id=sat_id, starts=starts, ends=ends, gs_ids_list=gs_ids_list,

            )



    def _binary_search_window(self, sat_id: int, timeslot: int) -> tuple[int, int, list[int]]:

        """二分查找定位卫星当前所在窗口或下一次窗口 O(logN)。"""

        sw = self._sorted_windows.get(sat_id)

        if sw is None or len(sw.starts) == 0:

            return (-1, -1, [])



        idx = bisect.bisect_right(sw.starts, timeslot)

        for check_idx in [idx - 1, idx - 2]:

            if 0 <= check_idx < len(sw.starts):  # noqa: SIM102

                if sw.starts[check_idx] <= timeslot <= sw.ends[check_idx]:

                    next_idx = check_idx + 1 if check_idx + 1 < len(sw.starts) else -1

                    return (check_idx, next_idx, sw.gs_ids_list[check_idx])



        next_idx = idx if idx < len(sw.starts) else -1

        return (-1, next_idx, [])



    def _compute_metrics(self, sat_id: int, timeslot: int) -> TimingMetrics:

        metrics = TimingMetrics(satellite_id=sat_id, current_timeslot=timeslot)

        sw = self._sorted_windows.get(sat_id)

        if sw is None or len(sw.starts) == 0:

            metrics.T_wait = 999999

            metrics.E_max = 0

            return metrics



        win_idx, next_idx, visible_gs = self._binary_search_window(sat_id, timeslot)



        if win_idx >= 0:

            metrics.T_window = int(sw.ends[win_idx] - timeslot + 1)

            metrics.T_wait = -1

            if next_idx >= 0:

                gap = int(sw.starts[next_idx] - sw.ends[win_idx])

                metrics.next_contact_slot = int(sw.starts[next_idx])

                metrics.next_contact_duration = int(sw.ends[next_idx] - sw.starts[next_idx] + 1)

            else:

                gap = 999999

            metrics.E_max = self._calc_emax(metrics.T_window, gap)

        else:

            metrics.T_window = -1

            if next_idx >= 0:

                metrics.T_wait = int(sw.starts[next_idx] - timeslot)

                metrics.next_contact_slot = int(sw.starts[next_idx])

                metrics.next_contact_duration = int(sw.ends[next_idx] - sw.starts[next_idx] + 1)

                next_gap = (

                    int(sw.starts[next_idx + 1] - sw.ends[next_idx])

                    if next_idx + 1 < len(sw.starts) else 999999

                )

                metrics.E_max = self._calc_emax(metrics.next_contact_duration, next_gap)

            else:

                metrics.T_wait = 999999

                metrics.E_max = 0



        if win_idx >= 0 and visible_gs:

            metrics.gs_weights = self._compute_gs_weights(visible_gs)

        if metrics.T_wait > 0 and metrics.E_max > 0:

            metrics.T_train_total = metrics.T_wait

        elif metrics.T_window > 0:

            metrics.T_train_total = 0



        return metrics



    def _calc_emax(self, window_slots: int, next_gap_slots: int) -> int:

        available = min(window_slots, next_gap_slots)

        if available <= 0:

            return 0

        return max(1, available // self.t_epoch_slots)



    def _compute_gs_weights(self, gs_ids: list[int]) -> dict[int, float]:

        if not gs_ids:

            return {}

        sim = self._scheduler._sim

        weights: dict[int, float] = {}

        for gid in gs_ids:

            if gid < sim.num_ground_stations:

                gs = sim.ground_network.stations[gid]

                lat_abs = abs(gs.lat_deg)

                if lat_abs > 60.0:

                    base_w = 2.0

                elif lat_abs > 30.0:

                    base_w = 1.5

                else:

                    base_w = 1.0

                weights[gid] = base_w

        total = sum(weights.values())

        return {k: v / total for k, v in weights.items()} if total > 0 else {g: 1.0 / len(gs_ids) for g in gs_ids}



    def query(self, sat_id: int, timeslot: int, force_refresh: bool = False) -> TimingMetrics:

        cached = self._cache.get(sat_id)

        if not force_refresh and cached is not None and cached.is_valid(timeslot):

            return cached

        metrics = self._compute_metrics(sat_id, timeslot)

        self._cache[sat_id] = metrics

        return metrics



    def query_all(self, timeslot: int) -> dict[int, TimingMetrics]:

        return {sat_id: self.query(sat_id, timeslot)

                for sat_id in range(self._scheduler._sim.num_satellites)}



    def incremental_update(self, timeslot: int) -> None:

        layered = self._layered_state

        if layered.last_layer2_refresh < 0 or timeslot - layered.last_layer2_refresh >= layered.layer2_interval_slots:

            self._refresh_layer2(timeslot)

            layered.last_layer2_refresh = timeslot

        if layered.last_layer1_refresh < 0 or timeslot - layered.last_layer1_refresh >= layered.layer1_interval_slots:

            self._refresh_layer1(timeslot)

            layered.last_layer1_refresh = timeslot

        self._last_updated_ts = timeslot



    def _refresh_layer2(self, timeslot: int) -> None:

        for sat_id in range(self._scheduler._sim.num_satellites):

            self.query(sat_id, timeslot, force_refresh=True)



    def _refresh_layer1(self, timeslot: int) -> None:

        self._build_sorted_windows()

        self._cache.clear()



    def get_dynamic_epochs(

        self, sat_id: int, timeslot: int,

        max_epochs: int = 10, warning_threshold_min: float = 5.0,

    ) -> int:

        """动态 Epoch 计算：根据 T_wait 自适应调整本地训练轮数。"""

        metrics = self.query(sat_id, timeslot)

        warning_slots = max(1, int(warning_threshold_min / self.timeslot_duration_min))

        if metrics.T_window > 0:

            remaining = metrics.T_window

            if remaining <= warning_slots:

                return 1

            emax = (remaining - warning_slots) // self.t_epoch_slots

        elif metrics.T_wait > 0:

            emax = metrics.T_wait // self.t_epoch_slots

        else:

            return 1

        return max(1, min(emax, max_epochs))



    def get_dynamic_mu(

        self, sat_id: int, timeslot: int,

        base_mu: float = 0.01, base_interval_min: float = 30.0,

    ) -> float:

        """自适应 μ: μ_dyn = base_mu × (T_base / T_wait)。"""

        metrics = self.query(sat_id, timeslot)

        if metrics.T_wait <= 0:

            return base_mu

        t_wait_min = metrics.T_wait * self.timeslot_duration_min

        ratio = base_interval_min / max(t_wait_min, 1.0)

        mu_dyn = base_mu * ratio

        return min(max(mu_dyn, base_mu * 0.1), base_mu * 10.0)



    def get_staleness_threshold(

        self, sat_id: int, timeslot: int, base_tau: int = 5,

    ) -> int:

        """动态陈旧阈值 τ: 长间隔卫星放宽，短间隔收紧。"""

        metrics = self.query(sat_id, timeslot)

        if metrics.T_wait <= 0:

            return base_tau

        t_wait_hours = (metrics.T_wait * self.timeslot_duration_min) / 60.0

        if t_wait_hours > 3.0:

            return min(base_tau + 5, 12)

        elif t_wait_hours > 1.0:

            return base_tau + 2

        else:

            return max(3, base_tau - 2)



    def get_isl_relay_info(self, sat_id: int, timeslot: int) -> tuple[bool, int]:

        """查询卫星是否可通过 ISL 中继通信。"""

        sim = self._scheduler._sim

        if not sim.isl_config.enabled:

            return (False, -1)

        sat_name = f"SAT-{sat_id:02d}"

        peers = sim.isl_peers_at(sat_name, timeslot)

        if not peers:

            return (False, -1)

        for peer_name in peers:

            try:

                peer_id = int(peer_name.replace("SAT-", ""))

            except (ValueError, AttributeError):

                continue

            if len(self._scheduler.get_connected_gss(peer_id, timeslot)) > 0:

                return (True, peer_id)

        return (False, -1)



    def compute_multi_dim_score(

        self, sat_id: int, staleness: int, timeslot: int,

        alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.3,

    ) -> float:

        """多维度复合加权打分: Score = alpha*(T_wait+T_window) + beta*Staleness + gamma*ISL_priority。"""

        metrics = self.query(sat_id, timeslot)

        t_total = max(metrics.T_wait, 0) + max(metrics.T_window, 0)

        isl_relay, _ = self.get_isl_relay_info(sat_id, timeslot)

        return alpha * t_total + beta * staleness + gamma * (0.0 if isl_relay else 1.0)



    def get_client_scores(

        self, client_ids: list[int], staleness_map: dict[int, int], timeslot: int,

    ) -> list[tuple[int, float]]:

        scores = [(cid, self.compute_multi_dim_score(cid, staleness_map.get(cid, 0), timeslot))

                  for cid in client_ids]

        scores.sort(key=lambda x: x[1])

        return scores



    def check_window_warning(

        self, sat_id: int, timeslot: int, warning_threshold_min: float = 5.0,

    ) -> bool:

        """检查是否临近窗口结束，需强制停止训练。"""

        metrics = self.query(sat_id, timeslot)

        if metrics.T_window <= 0:

            return False

        return metrics.T_window * self.timeslot_duration_min <= warning_threshold_min



    def get_stats(self) -> dict[str, Any]:

        return {

            "cached_satellites": len(self._cache),

            "sorted_windows_count": sum(len(sw) for sw in self._sorted_windows.values()),

            "last_updated_timeslot": self._last_updated_ts,

            "layered_state": {

                "L1_interval": self._layered_state.layer1_interval_slots,

                "L2_interval": self._layered_state.layer2_interval_slots,

                "last_L1_refresh": self._layered_state.last_layer1_refresh,

                "last_L2_refresh": self._layered_state.last_layer2_refresh,

            },

        }



    def clear(self) -> None:

        self._cache.clear()

        self._sorted_windows.clear()

        self._layered_state = LayeredTimingState()

        self._last_updated_ts = -1

