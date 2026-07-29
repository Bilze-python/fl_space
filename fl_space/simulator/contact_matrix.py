"""
接触矩阵模块 — 接触数据的存储、查询与导出

提供两种模式:
    Mode 1 (兼容模式): contact[sat][ts] = gs_id 或 -1
        只记录每个 timeslot 第一个可见的地面站。
        与原 orbit_sim_v2.py 兼容。

    Mode 2 (完整模式): contact[sat][ts] = [gs_id, ...]
        记录每个 timeslot 所有可见的地面站。
        满足用户需求 "记录每个卫星与哪些基站传输了信息"。
"""

import json
from typing import Optional

import numpy as np


class ContactMatrix:
    """
    接触矩阵 — 管理卫星-地面站接触数据。

    内部存储:
        _simple: np.ndarray shape (N_sat, N_slots), dtype=int
            -1 表示无接触，>=0 表示第一个可见地面站 ID
        _full: List[List[List[int]]]
            完整可见性数据，与 simple 模式同时维护
    """

    def __init__(
        self,
        num_satellites: int,
        num_timeslots: int,
        mode: str = "full",
    ):
        """
        Parameters
        ----------
        num_satellites : int
            卫星数量。
        num_timeslots : int
            timeslot 总数。
        mode : str
            "simple" — 只存第一个可见站（兼容模式）
            "full" — 存储所有可见站（新模式，默认）
        """
        self.num_satellites = num_satellites
        self.num_timeslots = num_timeslots
        self.mode = mode

        # 兼容矩阵
        self._simple = np.full((num_satellites, num_timeslots), -1, dtype=int)

        # 完整数据（仅 full 模式维护）
        self._full: list[list[list[int]]] = (
            [[[] for _ in range(num_timeslots)] for _ in range(num_satellites)]
            if mode == "full" else []
        )

    def extend(self, new_total_slots: int) -> int:
        """
        扩展接触矩阵到至少 new_total_slots 个时隙。

        保留已有数据，新增列初始化为 -1（无接触）。

        Parameters
        ----------
        new_total_slots : int
            新的总时隙数（必须 >= 当前值）。

        Returns
        -------
        int
            实际扩展后的 num_timeslots。
        """
        if new_total_slots <= self.num_timeslots:
            return self.num_timeslots

        added = new_total_slots - self.num_timeslots
        new_simple = np.full((self.num_satellites, new_total_slots), -1, dtype=int)
        new_simple[:, :self.num_timeslots] = self._simple
        self._simple = new_simple

        if self._full:
            for sat_entries in self._full:
                sat_entries.extend([[] for _ in range(added)])

        self.num_timeslots = new_total_slots
        return self.num_timeslots

    def set_contacts(self, sat_id: int, timeslot: int, gs_ids: list[int]):
        """
        设置指定卫星在指定 timeslot 的可见地面站列表。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        timeslot : int
            timeslot 编号。
        gs_ids : List[int]
            可见地面站 ID 列表（可为空）。
        """
        if not (0 <= sat_id < self.num_satellites):
            return
        if not (0 <= timeslot < self.num_timeslots):
            return

        # 简单模式
        self._simple[sat_id, timeslot] = gs_ids[0] if gs_ids else -1

        # 完整模式
        if self._full:
            self._full[sat_id][timeslot] = list(gs_ids)

    def get_first_contact(self, sat_id: int, timeslot: int) -> int:
        """
        获取简单接触（第一个可见站），兼容模式。

        Returns
        -------
        int
            地面站 ID 或 -1。
        """
        if 0 <= sat_id < self.num_satellites and 0 <= timeslot < self.num_timeslots:
            return int(self._simple[sat_id, timeslot])
        return -1

    def get_all_contacts(self, sat_id: int, timeslot: int) -> list[int]:
        """
        获取所有可见地面站。

        Returns
        -------
        List[int]
        """
        if self._full and 0 <= sat_id < self.num_satellites and 0 <= timeslot < self.num_timeslots:
            # 直接返回内部引用（调用方不应对其修改，调用方如需修改自行拷贝）
            return self._full[sat_id][timeslot]
        # fallback to simple
        gs_id = self.get_first_contact(sat_id, timeslot)
        return [gs_id] if gs_id >= 0 else []

    def get_next_contact(
        self, sat_id: int, after_timeslot: int
    ) -> Optional[tuple[int, int]]:
        """
        获取某卫星在指定 timeslot 之后的下一次接触（numpy 向量化）。

        Returns
        -------
        Optional[Tuple[int, int]]
            (timeslot, gs_id) 或 None
        """
        start = after_timeslot + 1
        if start >= self.num_timeslots:
            return None
        row = self._simple[sat_id, start:]
        in_contact = row >= 0
        if not np.any(in_contact):
            return None
        offset = int(np.argmax(in_contact))
        return (start + offset, int(row[offset]))

    def get_satellites_in_contact(self, timeslot: int) -> list[int]:
        """
        获取指定 timeslot 所有可与地面站通信的卫星 ID（numpy 向量化）。

        Returns
        -------
        List[int]
        """
        col = self._simple[:, timeslot]
        in_contact = col >= 0
        return np.where(in_contact)[0].tolist()

    def get_contact_detail(
        self, sat_id: int, timeslot: int, gs_names: Optional[list[str]] = None
    ) -> dict:
        """
        获取某时刻某卫星的接触详情（便于用户查看）。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        timeslot : int
            timeslot 编号。
        gs_names : List[str], optional
            地面站名称列表，用于可读输出。

        Returns
        -------
        Dict
            {
                'sat_id': int,
                'timeslot': int,
                'contact_count': int,
                'first_gs_id': int or None,
                'all_gs_ids': [int, ...],
                'gs_names': [str, ...]  (if gs_names provided)
            }
        """
        all_gs = self.get_all_contacts(sat_id, timeslot)
        first = self.get_first_contact(sat_id, timeslot)
        detail = {
            'sat_id': sat_id,
            'timeslot': timeslot,
            'contact_count': len(all_gs),
            'first_gs_id': first if first >= 0 else None,
            'all_gs_ids': all_gs,
        }
        if gs_names:
            detail['gs_names'] = [gs_names[gid] for gid in all_gs if 0 <= gid < len(gs_names)]
        return detail

    # ---- 统计 ----

    def compute_statistics(self) -> dict:
        """计算接触统计（numpy 向量化）。"""
        in_contact = self._simple >= 0  # bool 矩阵 (N_sat, N_slots)
        sat_counts = np.count_nonzero(in_contact, axis=1).tolist()
        total_contacts = int(np.sum(sat_counts))

        # 按地面站统计（向量化）
        gs_max_id = max(0, int(self._simple.max()))
        gs_counts = [
            int(np.count_nonzero(self._simple == gs_id))
            for gs_id in range(gs_max_id + 1)
        ]

        total_slots = self.num_satellites * self.num_timeslots
        return {
            'total_contacts': total_contacts,
            'sat_contact_counts': sat_counts,
            'gs_contact_counts': gs_counts,
            'avg_contacts_per_sat': float(np.mean(sat_counts)) if sat_counts else 0.0,
            'contact_rate': total_contacts / total_slots if total_slots > 0 else 0.0,
        }

    # ---- 导出 ----

    def to_dict(
        self,
        gs_names: Optional[list[str]] = None,
    ) -> dict:
        """
        导出为字典。

        Parameters
        ----------
        gs_names : List[str], optional
            地面站名称列表。

        Returns
        -------
        Dict
            含 contact_matrix 和 contact_details（如果 full 模式）。
        """
        data = {
            'mode': self.mode,
            'num_satellites': self.num_satellites,
            'num_timeslots': self.num_timeslots,
            'contact_matrix_simple': self._simple.tolist(),
        }
        if gs_names:
            data['ground_station_names'] = gs_names

        # 完整模式导出详情
        if self._full:
            details = []
            for sat_id in range(self.num_satellites):
                for ts in range(self.num_timeslots):
                    all_gs = self._full[sat_id][ts]
                    if all_gs:
                        detail = {
                            'sat_id': sat_id, 'timeslot': ts,
                            'gs_ids': all_gs,
                        }
                        if gs_names:
                            detail['gs_names'] = [
                                gs_names[gid] for gid in all_gs
                                if 0 <= gid < len(gs_names)
                            ]
                        details.append(detail)
            data['contact_details'] = details

        data['statistics'] = self.compute_statistics()
        return data

    def save_json(self, filepath: str, gs_names: Optional[list[str]] = None):
        """导出为 JSON 文件。"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(gs_names=gs_names), f, indent=2, ensure_ascii=False)

    @property
    def simple_matrix(self) -> np.ndarray:
        """返回兼容模式矩阵 (N_sat × N_slots, -1 表示无接触)。"""
        return self._simple

    def __repr__(self) -> str:
        stats = self.compute_statistics()
        return (
            f"ContactMatrix({self.num_satellites}sats x {self.num_timeslots}slots, "
            f"mode={self.mode}, contact_rate={stats['contact_rate']:.1%})"
        )
