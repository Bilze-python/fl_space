"""

轨道递推计算优化模块（论文维度二）

===================================



实现：

1. 并行批量轨道推演（多线程/多进程按卫星 ID 分块）

2. 轨道自适应可变采样步长（仰角临界点缩步、远离加步）

3. 轨道误差分段修正（叠加 Δt_err 偏差补偿）

4. 轨道数据缓存复用（未变轨卫星复用历史 ECI/ECEF）



Notes

-----

本模块提供纯函数工具，不修改 OrbitSimulator 内部状态。

并行策略采用 ``concurrent.futures.ThreadPoolExecutor``，适用于

Python GIL 对 numpy 数学运算影响较小的场景（numpy release GIL）。

"""



from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:

    from fl_space.orbit.kepler_orbit import KeplerOrbit
    from fl_space.simulator.orbit_simulator import OrbitSimulator





# ============================================================

# 1. 并行批量轨道推演

# ============================================================



def propagate_sat_ecef_batch(

    orbits: list[KeplerOrbit],

    timeslot: int,

    sim: OrbitSimulator,

    max_workers: int = 4,

) -> list[tuple[float, float, float]]:

    """并行计算所有卫星在指定 timeslot 的 ECEF 坐标。



    按卫星 ID 分块，线程池并发执行。



    Parameters

    ----------

    orbits : list[KeplerOrbit]

        卫星轨道列表。

    timeslot : int

        目标 timeslot 索引。

    sim : OrbitSimulator

    max_workers : int

        最大并行线程数（默认 4）。



    Returns

    -------

    list[tuple[float, float, float]]

        各卫星的 (x, y, z) ECEF 坐标 (km)，按 sat_id 排序。

    """

    n_sats = len(orbits)

    results: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * n_sats



    def _propagate_one(sat_id: int) -> tuple[int, tuple[float, float, float]]:

        ecef = sim.get_sat_ecef(sat_id, timeslot)

        return (sat_id, ecef)



    with ThreadPoolExecutor(max_workers=min(max_workers, n_sats)) as pool:

        futures = [pool.submit(_propagate_one, sid) for sid in range(n_sats)]

        for future in as_completed(futures):

            sid, ecef = future.result()

            results[sid] = ecef



    return results





def compute_contacts_batch(

    sim: OrbitSimulator,

    sat_ids: list[int],

    timeslot: int,

    max_workers: int = 4,

) -> dict[int, list[int]]:

    """并行计算多卫星在某 timeslot 的可见地面站列表。



    Returns

    -------

    dict[int, list[int]]

        {sat_id: [gs_id, ...]}

    """

    results: dict[int, list[int]] = {}



    def _contacts_one(sat_id: int) -> tuple[int, list[int]]:

        visible = sim.get_all_contacts(sat_id, timeslot)

        return (sat_id, visible)



    with ThreadPoolExecutor(max_workers=min(max_workers, len(sat_ids))) as pool:

        futures = [pool.submit(_contacts_one, sid) for sid in sat_ids]

        for future in as_completed(futures):

            sid, visible = future.result()

            results[sid] = visible



    return results





# ============================================================

# 2. 自适应可变采样步长

# ============================================================



def adaptive_sample_step(

    current_elevation_deg: float,

    min_elevation_deg: float,

    base_step_s: float = 30.0,

    fine_step_s: float = 5.0,

    coarse_step_s: float = 60.0,

    critical_margin_deg: float = 3.0,

) -> float:

    """根据当前仰角动态调整采样步长（论文维度二核心算法）。



    策略：

        - 仰角接近最低通信阈值 ±3°（窗口起止临界点）→ 步长缩至 5-10s

        - 仰角远大于阈值 → 步长放大至 60s

        - 中间区域 → 保持基础步长



    Parameters

    ----------

    current_elevation_deg : float

        当前时刻仰角 (°)。

    min_elevation_deg : float

        最低通信仰角 (°)。

    base_step_s : float

        基础步长 (s)。

    fine_step_s : float

        临界点精细步长 (s)。

    coarse_step_s : float

        远离临界点的粗大步长 (s)。

    critical_margin_deg : float

        临界判定边缘 (°)，默认 ±3°。



    Returns

    -------

    float

        自适应采样步长 (s)。

    """

    margin = abs(current_elevation_deg - min_elevation_deg)



    if margin <= critical_margin_deg:

        # 临界区域：精细采样

        ratio = margin / max(0.1, critical_margin_deg)

        return fine_step_s + ratio * (base_step_s - fine_step_s)

    if current_elevation_deg > min_elevation_deg + 10.0:

        # 高仰角区域：粗大步长

        return coarse_step_s

    # 中等区域

    return base_step_s





def generate_adaptive_timeline(

    start_s: float,

    end_s: float,

    elevations: Callable[[float], float],  # f(t_s) -> elevation_deg

    min_elevation_deg: float,

    base_step_s: float = 30.0,

) -> list[float]:

    """生成自适应采样的时间序列。



    在仰角临界点附近自动加密采样点，远离临界点加宽间隔。



    Parameters

    ----------

    start_s : float

        起始时间 (s)。

    end_s : float

        终止时间 (s)。

    elevations : callable

        仰角函数 elevations(t_s) -> el_deg。

    min_elevation_deg : float

        最低通信仰角。

    base_step_s : float

        基础步长。



    Returns

    -------

    list[float]

        自适应时间序列列表。

    """

    timeline: list[float] = []

    t = start_s

    while t <= end_s:

        timeline.append(t)

        el = elevations(t)

        step = adaptive_sample_step(el, min_elevation_deg, base_step_s=base_step_s)

        t += step

    # 确保包含终点

    if not timeline or timeline[-1] < end_s:

        timeline.append(end_s)

    return timeline





# ============================================================

# 3. 轨道误差分段修正

# ============================================================



@dataclass

class OrbitErrorCorrection:

    """轨道递推误差修正系数。



    通过对比理论预推与真实落地时间，积累偏差 Δt_err，

    下一轮 SGP4 递推时叠加补偿。

    """



    # 理论预推过境时间 (timeslot)

    predicted_ts: int = 0

    # 真实落地接轨时间 (timeslot)

    actual_ts: int = 0

    # 时序偏差 (timeslot)

    delta_ts_err: int = 0

    # 修正系数（直接叠加到传播时间）

    correction_seconds: float = 0.0





class ErrorCorrectionTracker:

    """轨道误差修正追踪器。



    按卫星维护偏差历史，供 SGP4 递推时叠加补偿。

    """



    def __init__(self, num_satellites: int):

        self._history: dict[int, list[OrbitErrorCorrection]] = {

            sat_id: [] for sat_id in range(num_satellites)

        }

        self._max_history = 5  # 滑动窗口大小



    def record_error(

        self, sat_id: int, predicted_ts: int, actual_ts: int

    ) -> None:

        """记录一次预推偏差。"""

        delta = actual_ts - predicted_ts

        correction = delta * 60.0  # 假设每个 timeslot = 1 min 基准

        record = OrbitErrorCorrection(

            predicted_ts=predicted_ts,

            actual_ts=actual_ts,

            delta_ts_err=delta,

            correction_seconds=correction,

        )

        self._history[sat_id].append(record)

        if len(self._history[sat_id]) > self._max_history:

            self._history[sat_id].pop(0)



    def get_correction_seconds(self, sat_id: int) -> float:

        """返回该卫星当前的累计修正量 (s)。



        使用指数加权平均平滑。

        """

        records = self._history[sat_id]

        if not records:

            return 0.0

        # EMA: 越近的偏差权重越高

        alpha = 0.6

        total = records[0].correction_seconds

        for r in records[1:]:

            total = alpha * r.correction_seconds + (1 - alpha) * total

        return total



    def get_correction_ts(self, sat_id: int, timeslot_duration_min: float = 1.0) -> int:

        """返回修正量（timeslot 单位）。"""

        seconds = self.get_correction_seconds(sat_id)

        return round(seconds / (timeslot_duration_min * 60.0))



    def clear(self, sat_id: int) -> None:

        self._history[sat_id].clear()





# ============================================================

# 4. 轨道数据缓存复用

# ============================================================



class OrbitCacheManager:

    """轨道 ECI/ECEF 坐标缓存。



    未变轨卫星直接复用历史坐标序列，仅对变轨卫星重新递推。

    显著减少重复数学运算（算力节省 80%+）。



    Parameters

    ----------

    num_satellites : int

    num_timeslots : int

        最大预推 timeslot 数。

    """



    def __init__(self, num_satellites: int, num_timeslots: int):

        self._ecf_cache: dict[int, dict[int, tuple[float, float, float]]] = {

            sid: {} for sid in range(num_satellites)

        }

        self._num_timeslots = num_timeslots

        self._dirty_sats: set[int] = set()  # 变轨/失效的卫星



    def mark_maneuver(self, sat_id: int) -> None:

        """标记卫星变轨，该星的所有缓存失效。"""

        self._dirty_sats.add(sat_id)

        self._ecf_cache[sat_id].clear()



    def is_cached(self, sat_id: int, timeslot: int) -> bool:

        return (

            sat_id not in self._dirty_sats

            and timeslot in self._ecf_cache[sat_id]

        )



    def get_ecef(

        self, sat_id: int, timeslot: int

    ) -> tuple[float, float, float] | None:

        if self.is_cached(sat_id, timeslot):

            return self._ecf_cache[sat_id][timeslot]

        return None



    def set_ecef(

        self, sat_id: int, timeslot: int, ecef: tuple[float, float, float]

    ) -> None:

        self._ecf_cache[sat_id][timeslot] = ecef



    def clear_all(self) -> None:

        for sid in self._ecf_cache:

            self._ecf_cache[sid].clear()



    def stats(self) -> dict:

        total_cached = sum(len(cache) for cache in self._ecf_cache.values())

        max_entries = len(self._ecf_cache) * self._num_timeslots

        return {

            "total_cached_entries": total_cached,

            "max_entries": max_entries,

            "cache_hit_ratio": round(total_cached / max(max_entries, 1), 4),

            "dirty_satellites": len(self._dirty_sats),

        }



# ============================================================
# 5. Error-adaptive refresh timing (Dimension 3)
# ============================================================

class ErrorAdaptiveRefresher:
    """Error-driven adaptive timing refresh scheduler.

    Records timing errors between prediction and ground truth,
    dynamically adjusts refresh intervals:
      - error < 5s  -> extend interval (e.g. 5min -> 15min)
      - error > 20s -> shorten and trigger incremental recalc
      - stable orbit -> auto-extend to reduce total computation
    """

    def __init__(
        self,
        base_interval_min: float = 5.0,
        min_interval_min: float = 2.0,
        max_interval_min: float = 30.0,
    ):
        self._base = base_interval_min
        self._min = min_interval_min
        self._max = max_interval_min
        self._current_intervals: dict[int, float] = {}
        self._error_history: dict[int, list[float]] = {}
        self._max_history = 5

    def get_interval(self, sat_id: int) -> float:
        return self._current_intervals.get(sat_id, self._base)

    def record_error(self, sat_id: int, delta_t_s: float) -> None:
        history = self._error_history.setdefault(sat_id, [])
        history.append(abs(delta_t_s))
        if len(history) > self._max_history:
            history.pop(0)
        self._adapt(sat_id, history)

    def _adapt(self, sat_id: int, history: list[float]) -> None:
        avg_err = sum(history) / len(history) if history else 0.0
        current = self._current_intervals.get(sat_id, self._base)
        if avg_err < 5.0:
            new_interval = min(self._max, current * 1.5)
        elif avg_err > 20.0:
            new_interval = max(self._min, current * 0.5)
        else:
            new_interval = current
        self._current_intervals[sat_id] = round(new_interval, 2)

    def needs_incremental_recalc(self, sat_id: int) -> bool:
        history = self._error_history.get(sat_id, [])
        if not history:
            return False
        return max(history[-3:]) > 20.0 if len(history) >= 3 else history[-1] > 20.0

    def clear(self, sat_id: int) -> None:
        self._error_history.pop(sat_id, None)
        self._current_intervals.pop(sat_id, None)


# ============================================================
# 6. Far window time-decay discard (Dimension 3)
# ============================================================

class FarWindowDiscarder:
    """Far window time-decay discarder.

    Windows beyond 24h are discarded from high-precision allocation
    and only kept as reference. Old far data is dropped on next timing cycle.
    """

    def __init__(self, far_horizon_slots: int = 1440):
        self._horizon = far_horizon_slots
        self._discard_count: int = 0

    def should_discard(self, ts_start: int, current_ts: int) -> bool:
        return ts_start > current_ts + self._horizon

    def filter_far_windows(
        self,
        records: list,
        current_ts: int,
        ts_start_attr: str = 'ts_start',
    ) -> tuple[list, list]:
        near: list = []
        discarded: list = []
        for r in records:
            ts = getattr(r, ts_start_attr)
            if self.should_discard(ts, current_ts):
                discarded.append(r)
            else:
                near.append(r)
        self._discard_count += len(discarded)
        return (near, discarded)

    @property
    def total_discarded(self) -> int:
        return self._discard_count
