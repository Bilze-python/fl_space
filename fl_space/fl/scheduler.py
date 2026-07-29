"""
通信调度器 v2 — 组网优化增强版

职责：
    - 读取 OrbitSimulator 的接触矩阵
    - 为 FL 训练提供通信可行性判断
    - 负载均衡：打散多 GS 争抢同一卫星窗口的扎堆现象
    - 窗口聚合：多 GS 稀疏卫星场景归集过境任务至少数 GS
    - 碎片窗口挖掘：多天线分时复用，提升饱和状态接触率天花板
    - GS 休眠调度：闲置地面站休眠关停，拉高单站利用率
    - 窗口预合并：重叠冲突窗口提前分组，降低优化复杂度
    - 时序平滑滤波：剔除瞬时异常窗口，减少训练输入抖动

设计原则：
    - 接口清晰：输入 = 模拟器引用，输出 = 通信状态查询
    - 低耦合：不依赖任何 FL 算法代码
    - 完全独立于 FL 算法逻辑，可单独测试和复用
"""

from __future__ import annotations

import bisect as _bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fl_space.simulator.orbit_simulator import OrbitSimulator


@dataclass
class CommunicationWindow:
    """
    单个通信窗口描述。

    Attributes
    ----------
    sat_id : int
        卫星 ID。
    gs_ids : list[int]
        可见地面站 ID 列表。
    timeslot_start : int
        窗口起始 timeslot。
    timeslot_end : int
        窗口结束 timeslot。
    duration_slots : int
        窗口持续时间（timeslot 数）。
    gs_weights : dict[int, float]
        多地面站权重映射 {gs_id: weight}。
        极地站长覆盖权重更高。
    """
    sat_id: int
    gs_ids: list[int]
    timeslot_start: int
    timeslot_end: int
    duration_slots: int
    gs_weights: dict[int, float] | None = None


class CommunicationScheduler:
    """
    FL 通信调度器。

    从模拟器读取接触矩阵，提供面向 FL 训练的通信查询接口。

    核心功能：
        - 查询某时刻哪些客户端可通信
        - 提取通信窗口列表（连续可通信的时间段）
        - 统计通信可用性
        - 二分查找定位当前窗口 O(logN)
        - T_wait / T_window 定时指标查询

    Parameters
    ----------
    simulator : OrbitSimulator
        已运行完毕的轨道模拟器实例。

    使用示例::

        from fl_space.simulator import OrbitSimulator
        from fl_space.fl.scheduler import CommunicationScheduler

        sim = OrbitSimulator(num_satellites=10, num_ground_stations=5)
        scheduler = CommunicationScheduler(sim)

        # 查询 timeslot 60 时哪些卫星可通信
        connected = scheduler.get_connected_sats(60)
        print(f"可通信卫星: {connected}")  # [0, 2, 5, 7]

        # 二分查找当前窗口
        window = scheduler.binary_search_window(0, 60)
    """

    def __init__(self, simulator: OrbitSimulator):
        self._sim = simulator
        self._contact_matrix = simulator.contact_matrix.simple_matrix

        # 预计算通信窗口缓存
        self._windows: dict[int, list[CommunicationWindow]] = {}
        self._build_windows()

        # ── 排序窗口数组（二分查找优化）──
        self._sorted_starts: dict[int, list[int]] = {}
        self._sorted_ends: dict[int, list[int]] = {}
        self._build_sorted_index()

        # ── 二分查找结果 LRU 缓存 ──
        self._bs_cache: dict[tuple[int, int], CommunicationWindow | None] = {}
        self._bs_cache_max = 4096

        # ── Round 12: 组网优化状态 ──
        self._gs_load: dict[int, int] = {}  # GS 负载计数
        self._gs_active: set[int] = set()   # 活跃 GS 集合
        self._gs_sleeping: set[int] = set() # 休眠 GS 集合
        self._load_history: list[float] = []  # 负载方差历史
        self._fragment_windows: dict[int, list[CommunicationWindow]] = {}  # 碎片窗口
        self._pre_merged_groups: list[list[CommunicationWindow]] = []  # 预合并组
        # GS 上限（超出部分不参与常规分配计算）
        self._gs_cap: int = 10

    def _build_windows(self) -> None:
        """
        从接触矩阵提取通信窗口。

        将每个卫星的连续可见区间合并为通信窗口。
        """
        n_sats = self._contact_matrix.shape[0]
        n_slots = self._contact_matrix.shape[1]

        for sat_id in range(n_sats):
            windows: list[CommunicationWindow] = []

            in_window = False
            window_start = 0
            window_gs: set[int] = set()

            for ts in range(n_slots):
                gs_list = self._sim.get_all_contacts(sat_id, ts)

                if gs_list and not in_window:
                    # 窗口开始
                    in_window = True
                    window_start = ts
                    window_gs = set(gs_list)
                elif gs_list and in_window:
                    # 窗口延续，合并地面站
                    window_gs.update(gs_list)
                elif not gs_list and in_window:
                    # 窗口结束
                    in_window = False
                    windows.append(CommunicationWindow(
                        sat_id=sat_id,
                        gs_ids=sorted(window_gs),
                        timeslot_start=window_start,
                        timeslot_end=ts - 1,
                        duration_slots=ts - window_start,
                    ))
                # 最后一个 timeslot 仍在窗口中
                if in_window and ts == n_slots - 1:
                    windows.append(CommunicationWindow(
                        sat_id=sat_id,
                        gs_ids=sorted(window_gs),
                        timeslot_start=window_start,
                        timeslot_end=ts,
                        duration_slots=ts - window_start + 1,
                    ))

            self._windows[sat_id] = windows

    def _build_sorted_index(self) -> None:
        """构建排序窗口数组，用于二分查找加速。

        对每颗卫星的 windows 按 timeslot_start 排序，
        构建 starts/ends 数组，供 binary_search_window 使用。
        """
        for sat_id, windows in self._windows.items():
            if not windows:
                self._sorted_starts[sat_id] = []
                self._sorted_ends[sat_id] = []
                continue
            self._sorted_starts[sat_id] = [w.timeslot_start for w in windows]
            self._sorted_ends[sat_id] = [w.timeslot_end for w in windows]

    # ── 二分查找优化（O(logN) 替代 O(N)）────────────────────
    def binary_search_window(
        self, sat_id: int, timeslot: int
    ) -> CommunicationWindow | None:
        """
        二分查找定位卫星当前所在通信窗口（含 LRU 缓存加速）。

        高频调用 (sat_id, timeslot) 组合从缓存返回，
        绕过 O(logN) 二分 + 回溯检查。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        timeslot : int
            查询时刻。

        Returns
        -------
        CommunicationWindow | None
            当前所在窗口，不在窗口中时返回 None。
        """
        cache_key = (sat_id, timeslot)
        cached = self._bs_cache.get(cache_key)
        if cached is not None or cached is False:
            return cached if cached is not False else None

        starts = self._sorted_starts.get(sat_id, [])
        ends = self._sorted_ends.get(sat_id, [])
        windows = self._windows.get(sat_id, [])

        if not starts:
            self._bs_cache[cache_key] = False
            self._evict_bs_cache()
            return None

        idx = _bisect.bisect_right(starts, timeslot)

        result: CommunicationWindow | None = None
        # 回溯检查：最多回溯 2 个窗口
        for check_idx in [idx - 1, idx - 2]:
            if (0 <= check_idx < len(starts)
                    and starts[check_idx] <= timeslot <= ends[check_idx]):
                w = windows[check_idx]
                # 附加多GS权重
                if w.gs_weights is None and w.gs_ids:
                    w.gs_weights = self._compute_gs_weights(w.gs_ids)
                result = w
                break

        self._bs_cache[cache_key] = result if result is not None else False
        self._evict_bs_cache()
        return result

    def _evict_bs_cache(self) -> None:
        """BS 缓存驱逐：容量超限时删除最旧的 1/4 条目。"""
        if len(self._bs_cache) > self._bs_cache_max:
            remove_count = self._bs_cache_max // 4
            for key in list(self._bs_cache.keys())[:remove_count]:
                del self._bs_cache[key]

    def find_next_window(
        self, sat_id: int, after_timeslot: int
    ) -> CommunicationWindow | None:
        """
        二分查找下一次通信窗口。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        after_timeslot : int
            此后时刻的时隙号。

        Returns
        -------
        CommunicationWindow | None
            下一个窗口，无后续窗口时返回 None。
        """
        starts = self._sorted_starts.get(sat_id, [])
        windows = self._windows.get(sat_id, [])

        if not starts:
            return None

        idx = _bisect.bisect_right(starts, after_timeslot)
        if 0 <= idx < len(windows):
            return windows[idx]
        return None

    # ── 定时指标查询（T_wait / T_window）───────────────────
    def get_timing_window(
        self, sat_id: int, timeslot: int
    ) -> int:
        """
        获取当前窗口剩余通信时长 T_window。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        timeslot : int
            当前时刻。

        Returns
        -------
        int
            剩余窗口 timeslot 数，不在窗口中时返回 0。
        """
        w = self.binary_search_window(sat_id, timeslot)
        if w is None:
            return 0
        return max(0, w.timeslot_end - timeslot + 1)

    def get_timing_wait(
        self, sat_id: int, timeslot: int
    ) -> int:
        """
        获取当前窗口结束后到下一次接轨的等待时间 T_wait。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        timeslot : int
            当前时刻。

        Returns
        -------
        int
            等待 timeslot 数，在当前窗口中时返回 -1。
        """
        w = self.binary_search_window(sat_id, timeslot)
        if w is not None:
            return -1  # 当前在窗口内

        next_w = self.find_next_window(sat_id, timeslot)
        if next_w is None:
            return 999999  # 无后续窗口
        return max(0, next_w.timeslot_start - timeslot)

    def _compute_gs_weights(self, gs_ids: list[int]) -> dict[int, float]:
        """计算多地面站权重：极地站权重高，赤道站权重低。

        Parameters
        ----------
        gs_ids : list[int]
            可见地面站 ID 列表。

        Returns
        -------
        dict[int, float]
            归一化权重。
        """
        if not gs_ids:
            return {}
        sim = self._sim
        weights: dict[int, float] = {}
        for gid in gs_ids:
            if 0 <= gid < sim.num_ground_stations:
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
        return {k: v / total for k, v in weights.items()} if total > 0 else {}

    def get_connected_sats(self, timeslot: int) -> list[int]:
        """
        获取指定时刻可通信的卫星列表。

        Parameters
        ----------
        timeslot : int
            时间槽编号。

        Returns
        -------
        list[int]
            可通信的卫星 ID 列表。
        """
        return self._sim.get_satellites_in_contact(timeslot)

    def get_connected_gss(self, sat_id: int, timeslot: int) -> list[int]:
        """
        获取某卫星在指定时刻可见的地面站列表。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        timeslot : int
            时间槽编号。

        Returns
        -------
        list[int]
            可见地面站 ID 列表。
        """
        return self._sim.get_all_contacts(sat_id, timeslot)

    def get_windows(self, sat_id: int) -> list[CommunicationWindow]:
        """
        获取某卫星的所有通信窗口。

        Parameters
        ----------
        sat_id : int
            卫星 ID。

        Returns
        -------
        list[CommunicationWindow]
            通信窗口列表，按时间排序。
        """
        return self._windows.get(sat_id, [])

    def is_connected(self, sat_id: int, timeslot: int) -> bool:
        """
        判断某卫星在指定时刻是否可通信。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        timeslot : int
            时间槽编号。

        Returns
        -------
        bool
            True 表示可通信。
        """
        return len(self._sim.get_all_contacts(sat_id, timeslot)) > 0

    def get_connectivity_stats(self) -> dict[str, Any]:
        """
        获取通信连通性统计。

        Returns
        -------
        dict
            包含各卫星的窗口数、总可用时间等统计信息。
        """
        stats: dict[str, Any] = {
            "num_sats": self._sim.num_satellites,
            "num_timeslots": self._sim.num_timeslots,
            "per_sat": {},
        }

        for sat_id in range(self._sim.num_satellites):
            windows = self._windows.get(sat_id, [])
            total_slots = sum(w.duration_slots for w in windows)
            stats["per_sat"][sat_id] = {
                "num_windows": len(windows),
                "total_contact_slots": total_slots,
                "contact_rate": (
                    total_slots / self._sim.num_timeslots
                    if self._sim.num_timeslots > 0
                    else 0
                ),
            }

        return stats

    # ── Round 12: 组网优化增强方法 ─────────────────────────

    def set_gs_cap(self, cap: int) -> None:
        """设置 GS 上限，多余 GS 不参与常规分配计算。

        推荐: GS 上限锁定在 8~10，多余地面站改为异地接力接收/备份冗余。
        """
        self._gs_cap = max(1, cap)
        self._gs_load = dict.fromkeys(range(min(self._sim.num_ground_stations, cap)), 0)

    @property
    def gs_cap(self) -> int:
        return self._gs_cap

    @property
    def effective_gs_count(self) -> int:
        """有效参与分配的地面站数量（不超过上限）。"""
        return min(self._sim.num_ground_stations, self._gs_cap)

    @property
    def backup_gs_count(self) -> int:
        """冗余备份地面站数量。"""
        return max(0, self._sim.num_ground_stations - self._gs_cap)

    def compute_load_balance(self, sat_gs_assignments: dict[int, int]) -> float:
        """计算地面站负载均衡方差。

        打散多 GS 争抢同一卫星窗口的扎堆现象，
        把闲置站点利用起来。

        Parameters
        ----------
        sat_gs_assignments : dict[int, int]
            {sat_id: gs_id} 分配映射。

        Returns
        -------
        float
            负载方差（越小越均衡）。
        """
        n_gs = self.effective_gs_count
        if n_gs <= 1:
            return 0.0
        loads = [0.0] * n_gs
        for gs_id in sat_gs_assignments.values():
            if gs_id < n_gs:
                loads[gs_id] += 1
        mean_load = sum(loads) / n_gs
        variance = sum((ld - mean_load) ** 2 for ld in loads) / n_gs
        self._load_history.append(variance)
        if len(self._load_history) > 1000:
            self._load_history.pop(0)
        return variance

    def get_load_variance(self) -> float:
        """获取负载方差。"""
        return self._load_history[-1] if self._load_history else 0.0

    def aggregate_windows_to_core_gs(
        self, windows: list[CommunicationWindow], core_gs_count: int = 3
    ) -> dict[int, list[CommunicationWindow]]:
        """窗口聚合：将分散的卫星过境任务归集至少数地面站集中处理。

        针对多 GS 稀疏卫星场景，避免任务分散到大量闲置 GS。

        Parameters
        ----------
        windows : list[CommunicationWindow]
            原始通信窗口列表。
        core_gs_count : int
            核心地面站数量（默认 3）。

        Returns
        -------
        dict[int, list[CommunicationWindow]]
            {gs_id: 归集后的窗口列表}。
        """
        # 统计各 GS 覆盖的窗口数，选出 top-N 核心站
        gs_window_count: dict[int, int] = {}
        for w in windows:
            for gid in w.gs_ids[: self._gs_cap]:
                gs_window_count[gid] = gs_window_count.get(gid, 0) + 1
        core_gs = sorted(gs_window_count, key=gs_window_count.get, reverse=True)[:core_gs_count]

        aggregated: dict[int, list[CommunicationWindow]] = {g: list[CommunicationWindow]() for g in core_gs}
        for w in windows:
            # 优先分配给覆盖它且属于核心的 GS
            best_gs = next((g for g in w.gs_ids if g in core_gs), None)
            if best_gs is None:
                best_gs = min(w.gs_ids, key=lambda g: gs_window_count.get(g, 0))
                if best_gs not in aggregated:
                    aggregated[best_gs] = []
            aggregated[best_gs].append(w)
        return aggregated

    def schedule_gs_sleep(
        self, active_sat_count: int, threshold_ratio: float = 3.0
    ) -> tuple[list[int], list[int]]:
        """GS 休眠调度：多站少星场景下休眠闲置 GS。

        只保留少量活跃 GS 承接卫星任务，
        拉高单站利用率与整体接触率。

        Parameters
        ----------
        active_sat_count : int
            当前在线卫星数。
        threshold_ratio : float
            GS/SAT 比值阈值，超过则休眠多余 GS。

        Returns
        -------
        tuple[list[int], list[int]]
            (active_gs_ids, sleeping_gs_ids)。
        """
        n_gs = self.effective_gs_count
        if active_sat_count == 0:
            return list(range(n_gs)), []
        gs_sat_ratio = n_gs / active_sat_count
        if gs_sat_ratio <= threshold_ratio:
            # 比例合理，全部激活
            self._gs_active = set(range(n_gs))
            self._gs_sleeping = set()
            return list(range(n_gs)), []
        # 需要休眠：保留 max(3, active_sat_count) 个 GS
        keep_count = max(3, active_sat_count)
        keep_count = min(keep_count, n_gs)
        # 选利用率最高的 keep_count 个 GS
        sorted_gs = sorted(range(n_gs), key=lambda g: self._gs_load.get(g, 0), reverse=True)
        active = sorted_gs[:keep_count]
        sleeping = sorted_gs[keep_count:]
        self._gs_active = set(active)
        self._gs_sleeping = set(sleeping)
        return active, sleeping

    def is_gs_active(self, gs_id: int) -> bool:
        """检查 GS 是否处于活跃状态。"""
        if gs_id >= self._gs_cap:
            return False
        if not self._gs_active:
            return True  # 未初始化时默认全活跃
        return gs_id in self._gs_active

    def mine_fragment_windows(
        self, sat_id: int, min_duration_slots: int = 5
    ) -> list[CommunicationWindow]:
        """碎片窗口挖掘：用多天线分时复用机制挖掘短时碎片窗口。

        将传统被忽略的短时窗口（如 < 10 slots）汇聚利用，
        提升饱和状态下的接触率天花板。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        min_duration_slots : int
            最小碎片窗口持续时长。

        Returns
        -------
        list[CommunicationWindow]
            可用的碎片窗口列表。
        """
        all_windows = self._windows.get(sat_id, [])
        # 找出所有短窗口（但足够容纳最小通信的）
        fragments = [
            w for w in all_windows
            if min_duration_slots <= w.duration_slots <= 20
        ]
        # 合并相邻碎片窗口（间隔 ≤ 3 slots）
        merged: list[CommunicationWindow] = []
        if not fragments:
            return merged
        current = fragments[0]
        for frag in fragments[1:]:
            if frag.timeslot_start - current.timeslot_end <= 3:
                # 合并
                all_gs = sorted(set(current.gs_ids + frag.gs_ids))
                current = CommunicationWindow(
                    sat_id=sat_id,
                    gs_ids=all_gs,
                    timeslot_start=current.timeslot_start,
                    timeslot_end=frag.timeslot_end,
                    duration_slots=frag.timeslot_end - current.timeslot_start + 1,
                )
            else:
                merged.append(current)
                current = frag
        merged.append(current)
        self._fragment_windows[sat_id] = merged
        return merged

    def pre_merge_windows(
        self, windows: list[CommunicationWindow], max_gap_slots: int = 5
    ) -> list[list[CommunicationWindow]]:
        """窗口预合并预处理：将重叠冲突窗口提前分组。

        降低优化模型的求解复杂度，缓解过载震荡。

        Parameters
        ----------
        windows : list[CommunicationWindow]
            待处理的通信窗口。
        max_gap_slots : int
            合并的最大间隔 slots。

        Returns
        -------
        list[list[CommunicationWindow]]
            分组后的窗口组列表。
        """
        if not windows:
            return []
        sorted_win = sorted(windows, key=lambda w: w.timeslot_start)
        groups: list[list[CommunicationWindow]] = [[sorted_win[0]]]
        for w in sorted_win[1:]:
            last_group = groups[-1]
            last_end = max(m.timeslot_end for m in last_group)
            if w.timeslot_start - last_end <= max_gap_slots:
                last_group.append(w)
            else:
                groups.append([w])
        self._pre_merged_groups = groups
        return groups

    def apply_temporal_smooth(
        self, contact_sequence: list[bool], window_size: int = 3
    ) -> list[bool]:
        """时序平滑滤波：剔除瞬时异常窗口，减少训练输入数据抖动。

        使用滑动窗口中值滤波，去除单点异常。

        Parameters
        ----------
        contact_sequence : list[bool]
            原始接触状态序列。
        window_size : int
            平滑窗口大小（奇数）。

        Returns
        -------
        list[bool]
            平滑后的序列。
        """
        n = len(contact_sequence)
        if n < window_size:
            return contact_sequence
        half = window_size // 2
        smoothed = list(contact_sequence)
        for i in range(half, n - half):
            window = contact_sequence[i - half: i + half + 1]
            true_count = sum(window)
            # 多数投票：窗口内 True > half 才保留
            smoothed[i] = true_count > half
        return smoothed

    def get_config_zone(self) -> str:
        """判断当前组网配置属于哪个区间。

        Returns
        -------
        str
            'efficient' (GS≤5, 高效) |
            'stable' (GS≥10, 平稳饱和) |
            'critical' (GS=6~8, 临界震荡) |
            'transition' (过渡)
        """
        gs = self.effective_gs_count
        if gs <= 5:
            return "efficient"
        if gs >= 10:
            return "stable"
        if 6 <= gs <= 8:
            return "critical"
        return "transition"

    def get_scheduling_strategy(self) -> dict[str, Any]:
        """根据 GS 规模返回分层调度策略。

        Returns
        -------
        dict
            包含 strategy 和参数的字典。
        """
        zone = self.get_config_zone()
        strategies = {
            "efficient": {
                "strategy": "fine_timing_optimal",
                "description": "精细化时序最优分配，最大化窗口填充率",
                "use_window_aggregation": False,
                "use_gs_sleep": False,
                "load_balance_strength": 0.0,
                "smoothness_constraint": 0.0,
            },
            "critical": {
                "strategy": "constrained_smooth",
                "description": "增加平滑约束项降复杂度，抑制输出结果震荡",
                "use_window_aggregation": True,
                "use_gs_sleep": False,
                "load_balance_strength": 0.3,
                "smoothness_constraint": 0.2,
            },
            "stable": {
                "strategy": "load_balanced_sleep",
                "description": "负载均衡+站点休眠策略，侧重系统稳定性",
                "use_window_aggregation": False,
                "use_gs_sleep": True,
                "load_balance_strength": 0.15,
                "smoothness_constraint": 0.1,
            },
            "transition": {
                "strategy": "balanced",
                "description": "过渡区间均衡策略",
                "use_window_aggregation": False,
                "use_gs_sleep": False,
                "load_balance_strength": 0.1,
                "smoothness_constraint": 0.05,
            },
        }
        return strategies.get(zone, strategies["transition"])

    def get_recommended_config(self) -> dict[str, Any]:
        """返回组网配置推荐。

        ┌──────────┬────────────────────────────┐
        │ 推荐配置  │ 效果                        │
        ├──────────┼────────────────────────────┤
        │ GS=3~5   │ 高效区间，接触率最高        │
        │ +SAT≥18  │                            │
        │ GS≥10    │ 平稳饱和区间，结果稳定可控  │
        │ +SAT任意 │                            │
        │ GS=6~8   │ 劣势震荡区间，尽量规避使用  │
        │ +SAT=12~ │                            │
        │ 18       │                            │
        └──────────┴────────────────────────────┘
        """
        gs = self.effective_gs_count
        return {
            "gs_count": gs,
            "backup_gs_count": self.backup_gs_count,
            "zone": self.get_config_zone(),
            "strategy": self.get_scheduling_strategy(),
            "recommendations": {
                "efficient": "GS=3~5 + SAT≥18 — 高效区间，接触率最高",
                "stable": "GS≥10 + SAT任意 — 平稳饱和区间",
                "avoid": "GS=6~8 + SAT=12~18 — 劣势震荡区间",
            },
            "sat_recommendation": (
                "卫星集群规模 > 地面站数量，"
                "星座部署优先扩充卫星数量，"
                "地面站采用小规模精干组网"
            ),
        }

    def simulate_time_progression(
        self,
        start_timeslot: int = 0,
        end_timeslot: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        生成时间推进的通信状态序列。

        用于 FL 训练中逐 timeslot 推进，
        每一步返回当前可通信客户端和地面站映射。

        Parameters
        ----------
        start_timeslot : int
            起始时间槽。
        end_timeslot : int | None
            结束时间槽，None 表示到模拟结束。

        Yields
        ------
        dict
            每步的通信状态：
            {"timeslot": int, "connected_sats": [...], "sat_to_gs": {...}}
        """
        if end_timeslot is None:
            end_timeslot = self._sim.num_timeslots

        for ts in range(start_timeslot, end_timeslot):
            connected = self.get_connected_sats(ts)
            sat_to_gs = {
                sid: self.get_connected_gss(sid, ts)
                for sid in connected
            }
            yield {
                "timeslot": ts,
                "connected_sats": connected,
                "sat_to_gs": sat_to_gs,
            }
