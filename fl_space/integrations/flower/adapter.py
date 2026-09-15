"""
Flower 框架适配器 — 将 SpaceFL 数据结构映射为 Flower 可消费的调度接口。

与师兄项目 autoFly_Stk 的 flower_adapter.py 对齐 API，但后端使用
SpaceFL 的 ContactMatrix + OrbitSimulator 而非 CSV 文件。

时间窗口语义：闭开区间 [start_utc, end_utc)。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fl_space.simulator import OrbitSimulator


@dataclass(frozen=True)
class ContactWindow:
    """单次 (satellite, ground_station) 可见窗口。"""

    satellite_name: str
    station_name: str
    start_utc: datetime
    end_utc: datetime
    duration_s: float
    max_elevation_deg: Optional[float] = None


@dataclass(frozen=True)
class IntraLinkWindow:
    """单次卫星-卫星簇内 LOS 窗口。"""

    satellite_a: str
    satellite_b: str
    cluster_id: Optional[str]
    start_utc: datetime
    end_utc: datetime
    duration_s: float
    min_range_km: Optional[float] = None
    max_range_km: Optional[float] = None


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class FlowerAdapter:
    """Flower 框架适配器 — 消费 SpaceFL OrbitSimulator 提供调度查询。

    构造方式：
        - FlowerAdapter.from_simulator(sim) — 从 OrbitSimulator 构造
        - FlowerAdapter(...) — 直接传入数据

    主要 API 与师兄项目 AccessSchedule 一致：
        - clients_visible_at(t_utc)
        - stations_visible_for(sat_id, t_utc)
        - next_contact(sat_id, after_utc)
        - intra_peers_at(sat_id, t_utc)
        - active_intra_links_at(t_utc)
        - select_clients_for_round(t_utc, max_clients)
    """

    def __init__(
        self,
        satellite_names: list[str],
        station_names: list[str],
        contact_windows: list[ContactWindow],
        intra_links: list[IntraLinkWindow] | None = None,
    ):
        self.satellite_names = tuple(satellite_names)
        self.station_names = tuple(station_names)
        self.contact_windows = tuple(contact_windows)
        self.intra_links = tuple(intra_links or [])

    @classmethod
    def from_simulator(
        cls,
        sim: OrbitSimulator,
        sim_start: Optional[datetime] = None,
    ) -> FlowerAdapter:
        """从 SpaceFL OrbitSimulator 构造适配器。

        Parameters
        ----------
        sim : OrbitSimulator
            SpaceFL 轨道模拟器实例。
        sim_start : datetime, optional
            模拟起始 UTC 时间。默认从 sim.sim_start_date 构造。

        Returns
        -------
        FlowerAdapter
        """
        if sim_start is None:
            yr, mo, dy = sim.sim_start_date
            sim_start = datetime(yr, mo, dy, tzinfo=timezone.utc)

        ts_min = sim.timeslot_duration_min

        # 构造卫星/地面站名称
        sat_names = [f"SAT-{i:02d}" for i in range(sim.num_satellites)]
        gs_names = list(sim.ground_network.names)

        # 从 ContactMatrix 构造 ContactWindow 列表
        contacts: list[ContactWindow] = []
        for sat_id in range(sim.num_satellites):
            for ts in range(sim.num_timeslots):
                gs_ids = sim.get_all_contacts(sat_id, ts)
                for gs_id in gs_ids:
                    start = sim_start + timedelta(minutes=ts * ts_min)
                    end = sim_start + timedelta(minutes=(ts + 1) * ts_min)
                    contacts.append(
                        ContactWindow(
                            satellite_name=sat_names[sat_id],
                            station_name=gs_names[gs_id]
                            if gs_id < len(gs_names)
                            else f"GS-{gs_id}",
                            start_utc=start,
                            end_utc=end,
                            duration_s=ts_min * 60.0,
                        )
                    )

        # 构造 ISL IntraLinkWindow 列表
        intra: list[IntraLinkWindow] = []
        if sim.isl_config.enabled:
            for w in sim.isl_windows:
                intra.append(
                    IntraLinkWindow(
                        satellite_a=w.satellite_a,
                        satellite_b=w.satellite_b,
                        cluster_id=w.cluster_id,
                        start_utc=w.start_utc,
                        end_utc=w.end_utc,
                        duration_s=w.duration_s,
                        min_range_km=w.min_range_km,
                        max_range_km=w.max_range_km,
                    )
                )

        return cls(
            satellite_names=sat_names,
            station_names=gs_names,
            contact_windows=contacts,
            intra_links=intra,
        )

    # ── 调度查询 ──────────────────────────────────────────

    def clients_visible_at(
        self,
        t_utc: str | datetime,
        station_subset: Iterable[str] | None = None,
    ) -> list[str]:
        """返回 t_utc 时刻可见的卫星名列表（去重，字典序）。"""
        t = (
            _ensure_utc(t_utc)
            if isinstance(t_utc, datetime)
            else _ensure_utc(datetime.fromisoformat(str(t_utc).replace("Z", "+00:00")))
        )
        allowed = None if station_subset is None else set(station_subset)
        seen: set[str] = set()
        for w in self.contact_windows:
            if allowed is not None and w.station_name not in allowed:
                continue
            if w.start_utc <= t < w.end_utc:
                seen.add(w.satellite_name)
        return sorted(seen)

    def stations_visible_for(
        self,
        sat_id: str,
        t_utc: str | datetime,
    ) -> list[str]:
        """返回某颗卫星在 t_utc 可见的地面站列表（字典序）。"""
        t = (
            _ensure_utc(t_utc)
            if isinstance(t_utc, datetime)
            else _ensure_utc(datetime.fromisoformat(str(t_utc).replace("Z", "+00:00")))
        )
        seen: set[str] = set()
        for w in self.contact_windows:
            if w.satellite_name != sat_id:
                continue
            if w.start_utc <= t < w.end_utc:
                seen.add(w.station_name)
        return sorted(seen)

    def next_contact(
        self,
        sat_id: str,
        after_utc: str | datetime,
        station_subset: Iterable[str] | None = None,
    ) -> ContactWindow | None:
        """返回 sat_id 在 after_utc 之后的下一次可见窗口。"""
        t = (
            _ensure_utc(after_utc)
            if isinstance(after_utc, datetime)
            else _ensure_utc(datetime.fromisoformat(str(after_utc).replace("Z", "+00:00")))
        )
        allowed = None if station_subset is None else set(station_subset)
        candidates: list[ContactWindow] = []
        for w in self.contact_windows:
            if w.satellite_name != sat_id:
                continue
            if allowed is not None and w.station_name not in allowed:
                continue
            if w.end_utc <= t:
                continue
            candidates.append(w)
        if not candidates:
            return None
        candidates.sort(key=lambda w: (w.start_utc, w.end_utc, w.station_name))
        return candidates[0]

    def intra_peers_at(
        self,
        sat_id: str,
        t_utc: str | datetime,
    ) -> list[str]:
        """返回 t_utc 时刻与 sat_id 有 LOS 的相邻卫星列表（字典序）。"""
        t = (
            _ensure_utc(t_utc)
            if isinstance(t_utc, datetime)
            else _ensure_utc(datetime.fromisoformat(str(t_utc).replace("Z", "+00:00")))
        )
        peers: set[str] = set()
        for w in self.intra_links:
            if not (w.start_utc <= t < w.end_utc):
                continue
            if w.satellite_a == sat_id:
                peers.add(w.satellite_b)
            elif w.satellite_b == sat_id:
                peers.add(w.satellite_a)
        return sorted(peers)

    def active_intra_links_at(
        self,
        t_utc: str | datetime,
    ) -> list[IntraLinkWindow]:
        """返回 t_utc 时刻所有活跃 intra link 窗口。"""
        t = (
            _ensure_utc(t_utc)
            if isinstance(t_utc, datetime)
            else _ensure_utc(datetime.fromisoformat(str(t_utc).replace("Z", "+00:00")))
        )
        return [w for w in self.intra_links if w.start_utc <= t < w.end_utc]

    def select_clients_for_round(
        self,
        t_utc: str | datetime,
        *,
        station_subset: Iterable[str] | None = None,
        max_clients: int | None = None,
    ) -> list[str]:
        """选择一轮 FL 训练的客户端（可见卫星）。

        Parameters
        ----------
        t_utc : str | datetime
            当前轮次时间。
        station_subset : iterable of str, optional
            限定地面站集合。
        max_clients : int, optional
            最大客户端数。

        Returns
        -------
        list[str]
            选择的卫星名列表（字典序）。
        """
        visible = self.clients_visible_at(t_utc, station_subset=station_subset)
        if max_clients is not None and len(visible) > max_clients:
            return visible[:max_clients]
        return visible
