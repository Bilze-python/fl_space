"""
通信调度器 — 将模拟器的接触矩阵映射为 FL 通信窗口

职责：
    - 读取 OrbitSimulator 的接触矩阵
    - 为 FL 训练提供通信可行性判断（某时刻某客户端是否可通信）
    - 支持时间步进（逐 timeslot 推进）
    - 完全独立于 FL 算法逻辑，可单独测试和复用

设计原则（遵循导师建议）：
    - 接口清晰：输入 = 模拟器引用，输出 = 通信状态查询
    - 低耦合：不依赖任何 FL 算法代码
    - 细粒度：仅负责"何时可通信"的判断，不涉及 FL 逻辑
"""

from __future__ import annotations

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
    """
    sat_id: int
    gs_ids: list[int]
    timeslot_start: int
    timeslot_end: int
    duration_slots: int


class CommunicationScheduler:
    """
    FL 通信调度器。

    从模拟器读取接触矩阵，提供面向 FL 训练的通信查询接口。

    核心功能：
        - 查询某时刻哪些客户端可通信
        - 提取通信窗口列表（连续可通信的时间段）
        - 统计通信可用性

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
    """

    def __init__(self, simulator: OrbitSimulator):
        self._sim = simulator
        self._contact_matrix = simulator.contact_matrix.simple_matrix

        # 预计算通信窗口缓存
        self._windows: dict[int, list[CommunicationWindow]] = {}
        self._build_windows()

    def _build_windows(self) -> None:
        """
        从接触矩阵提取通信窗口。

        将每个卫星的连续可见区间合并为通信窗口。
        """
        n_sats = self._contact_matrix.shape[0]
        n_slots = self._contact_matrix.shape[1]

        for sat_id in range(n_sats):
            windows: list[CommunicationWindow] = []
            contact_row = self._contact_matrix[sat_id]

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
