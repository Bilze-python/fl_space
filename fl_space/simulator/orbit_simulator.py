"""
轨道模拟器 — 模块化的太空联邦学习环境模拟

整合了环境层、轨道层、接触矩阵层的所有功能，
提供统一的、参数化的模拟接口。

支持:
    - 自定义行星（地球/火星/月球/自定义）
    - 灵活的星座配置（Walker/星簇/均匀）
    - 两种接触模式（兼容/完整）
    - 可扩展的大气模型
    - 地面站最小仰角约束

与原 orbit_sim_v2.py 兼容，同时提供更丰富的功能。
"""

from datetime import datetime, timedelta, timezone
import time as _time
from typing import Optional

import numpy as np

from fl_space.environment import (
    CelestialBody,
    GroundStationNetwork,
    create_default_network,
    create_extended_network,
)
from fl_space.isl.base import ISLConfig, ISLWindow
from fl_space.orbit import (
    ConstellationConfig,
    KeplerOrbit,
    MultiSatVisibility,
    generate_orbits,
)

from .contact_matrix import ContactMatrix


class OrbitSimulator:
    """
    模块化轨道接触模拟器。

    双后端支持:
        - backend="kepler"  : 轻量开普勒力学（默认，无外部依赖）
        - backend="skyfield": 高精度 SGP4/TLE + JPL 星历（需 pip install skyfield）

    使用示例::

        from fl_space.simulator import OrbitSimulator

        # 默认配置: Kepler后端 + 地球 + 3卫星 + 2地面站
        sim = OrbitSimulator()
        print(sim.summary())

        # Mars + Walker constellation
        from fl_space.environment import CelestialBody
        mars = CelestialBody.mars()
        sim_mars = OrbitSimulator(body=mars, num_satellites=5, num_ground_stations=3)

        # Skyfield高精度后端（地面站可见性 + TLE支持）
        sim_sf = OrbitSimulator(
            backend="skyfield", num_satellites=5, num_ground_stations=3,
            sim_start_date=(2024, 6, 1),
        )

        # 查询接触
        contacts = sim.get_satellites_in_contact(60)
        detail = sim.get_contact_detail(sat_id=0, timeslot=60)

    Parameters
    ----------
    backend : str
        "kepler" (默认) | "skyfield"。Skyfield 提供更精确的 SGP4 传播和
        地面站可见性计算，但需要 pip install skyfield。
    sim_start_date : tuple, optional
        (year, month, day) — 模拟开始日期，仅 skyfield 后端需要。
        默认 (2024, 6, 1)。
    """

    def __init__(
        self,
        # 天体参数
        body: Optional[CelestialBody] = None,
        # 星座参数
        num_satellites: int = 3,
        num_ground_stations: int = 2,
        orbit_altitude_km: float = 500.0,
        orbit_inclination_deg: float = 90.0,
        # 预构建轨道（优先级最高）
        orbits: Optional[list[KeplerOrbit]] = None,
        # 星座分布
        constellation_config: Optional[ConstellationConfig] = None,
        distribution: str = "uniform",  # "walker" | "cluster" | "uniform"
        # 地面站
        ground_station_network: Optional[GroundStationNetwork] = None,
        use_extended_gs: bool = False,
        # 时间参数
        timeslot_duration_min: float = 1.0,
        num_timeslots: int = 1440,
        # 轨道后端选择
        backend: str = "kepler",  # "kepler" | "skyfield"
        sim_start_date: Optional[tuple[int, int, int]] = None,  # (year, month, day) for skyfield
        # 接触矩阵模式
        contact_mode: str = "full",  # "simple" | "full"
        # ISL 星间链路
        isl_config: Optional[ISLConfig] = None,
        # 其他
        random_seed: int = 42,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        body : CelestialBody, optional
            中心天体。默认地球。
        num_satellites : int
            卫星数量。
        num_ground_stations : int
            地面站数量。
        orbit_altitude_km : float
            轨道高度 (km)。
        orbit_inclination_deg : float
            轨道倾角 (°)。
        constellation_config : ConstellationConfig, optional
            星座配置（优先于 num_satellites/orbit_altitude_km）。
        distribution : str
            分布策略: "walker" / "cluster" / "uniform"。
        ground_station_network : GroundStationNetwork, optional
            自定义地面站网络（优先于 num_ground_stations）。
        use_extended_gs : bool
            是否使用扩展地面站（准全球覆盖）。
        timeslot_duration_min : float
            每个 timeslot 时长 (分钟)。
        num_timeslots : int
            timeslot 总数。
        contact_mode : str
            "simple" 兼容模式 / "full" 完整记录。
        random_seed : int
            随机种子。
        verbose : bool
            是否打印进度信息。
        """
        self.rng = np.random.RandomState(random_seed)
        self.verbose = verbose
        self.backend_mode = backend
        self.sim_start_date = sim_start_date or (2024, 6, 1)

        # ---- 天体 ----
        self.body = body if body is not None else CelestialBody.earth()

        # ---- 星座配置 ----(仅当未直接提供 orbits 时使用)----
        if orbits is not None:
            # 使用预构建轨道
            self.orbits = list(orbits)
            self.num_satellites = len(self.orbits)
            self.constellation_config = None
            # 从轨道推断平均参数
            if self.orbits:
                avg_alt = sum(
                    o.elements.semi_major_axis_km - self.body.radius_km for o in self.orbits
                ) / len(self.orbits)
                self.orbit_altitude_km = avg_alt
                self.orbit_inclination_deg = self.orbits[0].elements.inclination_deg
            else:
                self.orbit_altitude_km = orbit_altitude_km
                self.orbit_inclination_deg = orbit_inclination_deg
        elif constellation_config is not None:
            self.constellation_config = constellation_config
            self.num_satellites = constellation_config.num_satellites
            self.orbit_altitude_km = constellation_config.altitude_km
            self.orbit_inclination_deg = constellation_config.inclination_deg
            # ---- 生成轨道 ----
            self.orbits: list[KeplerOrbit] = generate_orbits(self.constellation_config, self.body)
        else:
            self.constellation_config = ConstellationConfig(
                num_satellites=num_satellites,
                num_planes=1,
                inclination_deg=orbit_inclination_deg,
                altitude_km=orbit_altitude_km,
                distribution=distribution,
            )
            self.num_satellites = self.constellation_config.num_satellites
            self.orbit_altitude_km = self.constellation_config.altitude_km
            self.orbit_inclination_deg = self.constellation_config.inclination_deg
            # ---- 生成轨道 ----
            self.orbits: list[KeplerOrbit] = generate_orbits(self.constellation_config, self.body)

        # ---- 地面站 ----
        from fl_space.environment.ground_station import GroundStationNetwork

        if ground_station_network is not None:
            # 统一转换为 GroundStationNetwork（兼容 list[GroundStation] 输入）
            if isinstance(ground_station_network, GroundStationNetwork):
                self.ground_network = ground_station_network
            else:
                self.ground_network = GroundStationNetwork(ground_station_network)
        elif use_extended_gs:
            self.ground_network = create_extended_network(num_ground_stations)
        else:
            self.ground_network = create_default_network(num_ground_stations)

        self.num_ground_stations = self.ground_network.count

        # ---- 时间参数 ----
        self.timeslot_duration_min = timeslot_duration_min
        self.num_timeslots = num_timeslots
        self.num_timeslots_pre = num_timeslots  # 记录初始预计算大小

        # 轨道周期 (取第一颗卫星，因为都使用相同高度)
        self.orbit_period_min = self.orbits[0].period_min if self.orbits else 0.0

        # ---- 轨道后端 ----
        self._sf_backend = None
        self._sf_satellites = []
        if backend == "skyfield":
            self._init_skyfield_backend()

        # ---- 接触矩阵 ----
        self.contact_mode = contact_mode
        self.contact_matrix = ContactMatrix(
            self.num_satellites, self.num_timeslots, mode=contact_mode
        )

        # ---- 保存可见性引擎引用（用于按需扩展）----
        self._multi_vis = None  # 延迟初始化（避免不必要的创建）

        # ---- ECEF 坐标缓存（减少重复轨道推算）----
        from fl_space.orbit.optimizer import OrbitCacheManager

        self._ecef_cache = OrbitCacheManager(self.num_satellites, self.num_timeslots)

        # ---- 地理栅格预筛选器（跳过不可达地面站）----
        from fl_space.simulator.sparse_computer import GeoGridFilter

        self._geo_grid_filter = GeoGridFilter(self)

        # ---- 生成接触数据 ----
        t0 = _time.time()
        self._generate_contacts()
        self._gen_time = _time.time() - t0

        # ---- 统计 ----
        self.stats = self.contact_matrix.compute_statistics()

        # ---- ISL 星间链路 (lazy) ----
        self.isl_config = isl_config if isl_config is not None else ISLConfig(enabled=False)
        self._isl_windows: Optional[list[ISLWindow]] = None
        self._isl_calculator = None

    # ============================================================
    # ISL 星间链路
    # ============================================================

    def compute_isl(self, force: bool = False) -> list[ISLWindow]:
        """计算 ISL 窗口（lazy，首次调用后缓存）。

        Parameters
        ----------
        force : bool
            强制重新计算。

        Returns
        -------
        list[ISLWindow]
            ISL 可见窗口列表。
        """
        if self._isl_windows is not None and not force:
            return self._isl_windows

        if not self.isl_config.enabled:
            self._isl_windows = []
            return self._isl_windows

        from fl_space.isl.intra_cluster import create_isl_calculator

        calculator = create_isl_calculator(self.isl_config)
        self._isl_calculator = calculator

        # 构造 ECEF 时序数据
        n_sats = self.num_satellites
        n_slots = self.num_timeslots
        ecef_positions: dict[str, np.ndarray] = {}
        for sat_id in range(n_sats):
            arr = np.zeros((3, n_slots))
            for ts in range(n_slots):
                time_min = ts * self.timeslot_duration_min
                x, y, z = self.orbits[sat_id].ecef_at_time(time_min)
                arr[0, ts] = x
                arr[1, ts] = y
                arr[2, ts] = z
            name = f"SAT-{sat_id:02d}"
            ecef_positions[name] = arr

        # 星簇分配：按轨道面（plane）分组
        cluster_map: dict[str, Optional[str]] = {}
        if self.isl_config.cluster_mode == "plane" and self.constellation_config:
            for sat_id in range(n_sats):
                name = f"SAT-{sat_id:02d}"
                sats_per_plane = max(n_sats // max(self.constellation_config.num_planes, 1), 1)
                plane_id = sat_id // sats_per_plane
                cluster_map[name] = f"plane-{plane_id}"
        else:
            for sat_id in range(n_sats):
                cluster_map[f"SAT-{sat_id:02d}"] = None

        # 生成时间序列
        base_dt = datetime(*self.sim_start_date, tzinfo=timezone.utc)
        sample_times = [
            base_dt + timedelta(minutes=ts * self.timeslot_duration_min) for ts in range(n_slots)
        ]

        self._isl_windows = calculator.compute(
            ecef_positions=ecef_positions,
            cluster_assignments=cluster_map,
            sample_times=sample_times,
            atmosphere_buffer_km=self.isl_config.atmosphere_buffer_km,
        )

        # 构建 ISL 时隙索引（O(1) 查询替代 O(N) 扫描）
        self._build_isl_index()
        return self._isl_windows

    @property
    def isl_windows(self) -> list[ISLWindow]:
        """ISL 窗口列表（首次访问时自动计算）。"""
        return self.compute_isl()

    @property
    def isl_stats(self) -> dict:
        """ISL 统计摘要。"""
        windows = self.isl_windows
        if not windows:
            return {
                "total_windows": 0,
                "total_duration_s": 0.0,
                "unique_links": 0,
                "avg_duration_s": 0.0,
            }
        total_dur = sum(w.duration_s for w in windows)
        unique_links = len({(w.satellite_a, w.satellite_b) for w in windows})
        return {
            "total_windows": len(windows),
            "total_duration_s": total_dur,
            "unique_links": unique_links,
            "avg_duration_s": total_dur / len(windows),
        }

    def _build_isl_index(self) -> None:
        """构建 ISL 时隙索引 _isl_by_timeslot: dict[timeslot, list[ISLWindow]]。

        将 ISL 窗口的 UTC 时间范围转换为 timeslot 范围并做时隙映射，
        使 isl_active_at() 从 O(N) 扫描降为 O(1) 字典查询。
        对含大量 ISL 窗口的星座（如 megaconstellation）收益显著。
        """
        if not self._isl_windows:
            self._isl_by_timeslot: dict[int, list[ISLWindow]] = {}
            return

        base_dt = datetime(*self.sim_start_date, tzinfo=timezone.utc)
        slot_sec = self.timeslot_duration_min * 60.0
        n_slots = self.num_timeslots
        idx: dict[int, list[ISLWindow]] = {ts: [] for ts in range(n_slots)}

        for w in self._isl_windows:
            start_ts = int((w.start_utc - base_dt).total_seconds() / slot_sec)
            end_ts = int((w.end_utc - base_dt).total_seconds() / slot_sec)
            start_ts = max(0, start_ts)
            end_ts = min(n_slots - 1, end_ts)
            for ts in range(start_ts, end_ts + 1):
                idx[ts].append(w)

        self._isl_by_timeslot = idx

    def isl_active_at(self, timeslot: int) -> list[ISLWindow]:
        """查询指定 timeslot 中活跃的 ISL 窗口（O(1) 索引查询）。

        Parameters
        ----------
        timeslot : int
            时隙编号。

        Returns
        -------
        list[ISLWindow]
            该时隙内活跃的 ISL 窗口。
        """
        # 兼容未构建索引的旧调用（回退到 O(N) 扫描）
        if not hasattr(self, '_isl_by_timeslot') or not self._isl_by_timeslot:
            base_dt = datetime(*self.sim_start_date, tzinfo=timezone.utc)
            ts_start = base_dt + timedelta(minutes=timeslot * self.timeslot_duration_min)
            ts_end = base_dt + timedelta(minutes=(timeslot + 1) * self.timeslot_duration_min)
            return [w for w in self.isl_windows if w.start_utc < ts_end and w.end_utc > ts_start]
        return self._isl_by_timeslot.get(timeslot, [])

    def isl_peers_at(self, sat_name: str, timeslot: int) -> list[str]:
        """查询指定时隙中与某卫星有 ISL 连接的相邻卫星。

        Parameters
        ----------
        sat_name : str
            卫星名称，如 "SAT-00"。
        timeslot : int
            时隙编号。

        Returns
        -------
        list[str]
            对端卫星名称列表（字典序）。
        """
        active = self.isl_active_at(timeslot)
        peers = set()
        for w in active:
            if w.satellite_a == sat_name:
                peers.add(w.satellite_b)
            elif w.satellite_b == sat_name:
                peers.add(w.satellite_a)
        return sorted(peers)

    # ============================================================
    # Skyfield 后端初始化
    # ============================================================

    def _init_skyfield_backend(self):
        """初始化 Skyfield 高精度后端。"""
        from fl_space.orbit.skyfield_backend import SkyfieldOrbitBackend

        self._sf_backend = SkyfieldOrbitBackend()
        self._sf_satellites = []

        # 计算模拟起始时刻的儒略日（TLE epoch 需与模拟时间一致）
        from datetime import datetime

        yr, mo, dy = self.sim_start_date
        datetime(yr, mo, dy)
        # 转为 julian date (简化: datetime to JD)
        a = (14 - mo) // 12
        y = yr + 4800 - a
        m = mo + 12 * a - 3
        epoch_jd = dy + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045 - 0.5

        for i, orbit in enumerate(self.orbits):
            sat = self._sf_backend.create_satellite_from_kepler(
                altitude_km=self.orbit_altitude_km,
                inclination_deg=orbit.elements.inclination_deg,
                raan_deg=orbit.elements.raan_deg,
                true_anomaly_deg=orbit.elements.true_anomaly_deg,
                eccentricity=orbit.elements.eccentricity,
                epoch_jd=epoch_jd,
                name=f"SF-SAT-{i}",
            )
            self._sf_satellites.append(sat)

        if self.verbose:
            print(
                f"    [OrbitSim] Skyfield后端初始化完成 "
                f"({self.num_satellites} satellites, "
                f"epoch {self.sim_start_date}, JPL DE421 ephemeris)"
            )

    # ============================================================
    # 接触矩阵生成
    # ============================================================

    def _generate_contacts(self):
        """生成完整接触矩阵（根据后端自动选择算法）。"""
        if self.backend_mode == "skyfield" and self._sf_backend is not None:
            self._generate_contacts_skyfield()
        else:
            self._generate_contacts_kepler()

    def _generate_contacts_kepler(self):
        """Kepler 后端接触矩阵生成（含 GeoGridFilter 预筛选优化）。

        思路:
        1. GeoGridFilter 预筛：对每颗卫星，筛除轨道不可达的地面站
        2. 仅对可达站点做球面几何判定，跳过永远不可视的星站组合
        """
        multi_vis = self._get_multi_vis()
        n_gs = self.num_ground_stations

        # 对每颗卫星预计算不可达的 GS 集合（仅一次）
        excluded_gs: dict[int, set[int]] = {}
        for sat_id in range(self.num_satellites):
            orbit = self.orbits[sat_id]
            incl = float(orbit.elements.inclination_deg)
            alt = float(orbit.elements.semi_major_axis_km - self.body.radius_km)
            reachable = self._geo_grid_filter.get_gs_ids_for_orbit(incl, alt)
            excluded_gs[sat_id] = {gid for gid in range(n_gs) if gid not in reachable}

        for ts in range(self.num_timeslots):
            if self.verbose and self.num_timeslots > 5000 and ts % (self.num_timeslots // 10) == 0:
                print(
                    f"    [OrbitSim] 生成接触数据: {ts}/{self.num_timeslots} "
                    f"({100 * ts / self.num_timeslots:.0f}%)",
                    flush=True,
                )

            time_min = ts * self.timeslot_duration_min
            all_visible = multi_vis.visible_matrix_at_time(time_min)

            for sat_id, visible_gs in enumerate(all_visible):
                # 过滤掉轨道不可达的地面站
                filtered = [gid for gid in visible_gs if gid not in excluded_gs.get(sat_id, set())]
                self.contact_matrix.set_contacts(sat_id, ts, filtered)

        if self.verbose:
            n_contact = int(np.count_nonzero(self.contact_matrix.simple_matrix >= 0))
            total_slots = self.num_satellites * self.num_timeslots
            print(
                f"    [OrbitSim:kepler] 接触矩阵生成完毕, "
                f"接触率: {n_contact / max(total_slots, 1) * 100:.1f}%"
            )

    def _get_multi_vis(self):
        """延迟初始化 MultiSatVisibility（Kepler 后端按需计算用）。"""
        if self._multi_vis is None:
            self._multi_vis = MultiSatVisibility(
                self.body,
                self.orbits,
                self.ground_network,
            )
        return self._multi_vis

    def _compute_contacts_for_slot(self, timeslot: int) -> list[list[int]]:
        """
        按需计算单个 timeslot 的接触数据并写入矩阵。

        如果 timeslot 超出当前矩阵范围，自动扩展矩阵。

        Returns
        -------
        list[list[int]]
            每个卫星的可见地面站 ID 列表。
        """
        if timeslot >= self.num_timeslots:
            # 扩展矩阵：至少扩到 timeslot+1，每次扩展量不少于一个轨道周期
            min_extension = int(1.2 * self.orbit_period_min / self.timeslot_duration_min)
            target = max(timeslot + 1, self.num_timeslots + min_extension)
            self._extend_to(target)

        time_min = timeslot * self.timeslot_duration_min
        multi_vis = self._get_multi_vis()
        all_visible = multi_vis.visible_matrix_at_time(time_min)

        for sat_id, visible_gs in enumerate(all_visible):
            self.contact_matrix.set_contacts(sat_id, timeslot, visible_gs)

        return all_visible

    def _extend_to(self, target_slots: int) -> None:
        """
        批量扩展接触矩阵至 target_slots，一次性计算新增时隙的接触数据。

        Parameters
        ----------
        target_slots : int
            目标时隙总数。
        """
        if target_slots <= self.num_timeslots:
            return

        old_slots = self.num_timeslots
        new_slots = self.contact_matrix.extend(target_slots)
        added = new_slots - old_slots

        multi_vis = self._get_multi_vis()

        for ts in range(old_slots, new_slots):
            time_min = ts * self.timeslot_duration_min
            all_visible = multi_vis.visible_matrix_at_time(time_min)
            for sat_id, visible_gs in enumerate(all_visible):
                self.contact_matrix.set_contacts(sat_id, ts, visible_gs)

        self.num_timeslots = new_slots

        if self.verbose and added > 100:
            print(
                f"    [OrbitSim] 按需扩展 {added} 时隙 "
                f"({added * self.timeslot_duration_min / 60:.1f}h), "
                f"总计 {new_slots} 时隙"
            )

    def _generate_contacts_skyfield(self):
        """Skyfield 后端接触矩阵生成（使用精确仰角判断）。"""
        sf = self._sf_backend
        yr, mo, dy = self.sim_start_date

        for ts in range(self.num_timeslots):
            if self.verbose and self.num_timeslots > 1000 and ts % (self.num_timeslots // 10) == 0:
                print(
                    f"    [OrbitSim:skyfield] 生成接触数据: {ts}/{self.num_timeslots} "
                    f"({100 * ts / self.num_timeslots:.0f}%)",
                    flush=True,
                )

            minute_of_day = ts * self.timeslot_duration_min
            day_offset = int(minute_of_day / 1440)
            minute_in_day = minute_of_day % 1440
            hour = minute_in_day / 60.0

            # 计算当前UTC日期
            from datetime import datetime, timedelta

            current_dt = datetime(yr, mo, dy) + timedelta(days=day_offset)
            cy, cm, cd = current_dt.year, current_dt.month, current_dt.day

            for sat_id, sat in enumerate(self._sf_satellites):
                visible_gs = []
                for gs_id, gs in enumerate(self.ground_network):
                    try:
                        elev, _, _ = sf.elevation_at(
                            sat,
                            gs.lat_deg,
                            gs.lon_deg,
                            cy,
                            cm,
                            cd,
                            hour,
                        )
                        if elev >= gs.min_elevation_deg:
                            visible_gs.append(gs_id)
                    except Exception:
                        # Skip if SGP4 propagation fails for this time
                        pass

                self.contact_matrix.set_contacts(sat_id, ts, visible_gs)

        if self.verbose:
            print(
                f"    [OrbitSim:skyfield] 接触矩阵生成完毕, "
                f"接触率: {np.mean(np.sum(self.contact_matrix.simple_matrix >= 0, axis=1)) / self.num_timeslots * 100:.1f}%"
            )

    # ============================================================
    # 查询接口（兼容原 orbit_sim_v2.py）
    # ============================================================

    def get_contact_at_timeslot(self, sat_id: int, timeslot: int) -> int:
        """获取简单接触 (gs_id 或 -1)，兼容原接口。"""
        return self.contact_matrix.get_first_contact(sat_id, timeslot)

    def get_all_contacts(self, sat_id: int, timeslot: int) -> list[int]:
        """获取所有可见地面站 ID 列表（自动扩展矩阵）。"""
        if timeslot >= self.num_timeslots:
            all_visible = self._compute_contacts_for_slot(timeslot)
            return all_visible[sat_id] if sat_id < len(all_visible) else []
        return self.contact_matrix.get_all_contacts(sat_id, timeslot)

    def get_next_contact(self, sat_id: int, after_timeslot: int) -> Optional[tuple[int, int]]:
        """
        获取某卫星在指定 timeslot 之后的下一次接触。

        自动扩展矩阵：如果预计算范围内无接触，继续向前搜索。

        Returns
        -------
        Optional[tuple[int, int]]
            (timeslot, gs_id) 或 None。
        """
        # 先在预计算范围内查找
        result = self.contact_matrix.get_next_contact(sat_id, after_timeslot)
        if result is not None:
            return result

        # 按需搜索：每次扩展一个轨道周期的时隙，直到找到接触
        search_chunk = int(1.2 * self.orbit_period_min / self.timeslot_duration_min)
        search_chunk = max(search_chunk, 100)  # 至少100个时隙

        search_start = max(after_timeslot + 1, self.num_timeslots)
        search_end = search_start + search_chunk

        while True:
            self._extend_to(search_end)
            # 在新扩展的范围内查找
            for ts in range(search_start, search_end):
                gs_id = self.contact_matrix.get_first_contact(sat_id, ts)
                if gs_id >= 0:
                    return (ts, int(gs_id))
            # 未找到，继续扩展
            search_start = search_end
            search_end += search_chunk
            # 安全阀：最多搜索 100 个轨道周期（几十年模拟时间）
            if search_end > self.num_timeslots + search_chunk * 100:
                return None

    def get_satellites_in_contact(self, timeslot: int) -> list[int]:
        """获取某时刻可通信的所有卫星（自动扩展矩阵）。"""
        if timeslot >= self.num_timeslots:
            all_visible = self._compute_contacts_for_slot(timeslot)
            return [sat_id for sat_id, visible in enumerate(all_visible) if visible]
        return self.contact_matrix.get_satellites_in_contact(timeslot)

    def get_contact_detail(self, sat_id: int, timeslot: int) -> dict:
        """获取某时刻某卫星的接触详情。"""
        return self.contact_matrix.get_contact_detail(
            sat_id, timeslot, gs_names=self.ground_network.names
        )

    # ============================================================
    # FL 轮次计算
    # ============================================================

    def compute_round_duration_sync(
        self,
        local_epochs: int = 1,
        compute_time_per_epoch_min: float = 2.0,
        start_timeslot: int = 0,
    ) -> tuple[int, int]:
        """
        计算同步 FL 一轮的持续时间。

        同步机制: 所有卫星必须都完成训练并返回模型后才算一轮结束。

        Parameters
        ----------
        local_epochs : int
            本地训练 epoch 数。
        compute_time_per_epoch_min : float
            每 epoch 计算时间 (分钟)。
        start_timeslot : int
            起始 timeslot。

        Returns
        -------
        (round_duration_slots, last_return_timeslot)
        """
        train_time_slots = int(
            local_epochs * compute_time_per_epoch_min / self.timeslot_duration_min
        )

        return_times = []
        for sat_id in range(self.num_satellites):
            # 1) 第一次接触 (接收全局模型)
            first = self.get_next_contact(sat_id, start_timeslot - 1)
            if first is None:
                continue
            recv_ts, _ = first

            # 2) 训练结束后找返回接触
            train_end_ts = recv_ts + train_time_slots
            ret = self.get_next_contact(sat_id, train_end_ts - 1)
            if ret is None:
                continue
            return_ts, _ = ret
            return_times.append(return_ts)

        if not return_times:
            return (0, start_timeslot)

        last_ts = max(return_times)
        duration = last_ts - start_timeslot
        return (duration, last_ts)

    # ============================================================
    # 额外查询（新的增强接口）
    # ============================================================

    def get_sat_position(self, sat_id: int, timeslot: int) -> tuple[float, float]:
        """
        获取卫星在指定 timeslot 的星下点位置。

        Returns
        -------
        (lat_deg, lon_deg)
        """
        time_min = timeslot * self.timeslot_duration_min
        return self.orbits[sat_id].position_at_time_deg(time_min)

    def get_sat_positions_at_timeslot(self, timeslot: int) -> dict[int, tuple[float, float]]:
        """获取某时刻所有卫星位置。"""
        return {
            sat_id: self.get_sat_position(sat_id, timeslot) for sat_id in range(self.num_satellites)
        }

    def get_sat_ecef(self, sat_id: int, timeslot: int) -> tuple[float, float, float]:
        """
        获取卫星在指定 timeslot 的 ECEF 坐标 (km)。

        使用 OrbitCacheManager 缓存，避免重复轨道推算。
        同一卫星同一 timeslot 多次查询仅计算一次。

        Returns
        -------
        (x_km, y_km, z_km)
        """
        cached = self._ecef_cache.get_ecef(sat_id, timeslot)
        if cached is not None:
            return cached
        time_min = timeslot * self.timeslot_duration_min
        ecef = self.orbits[sat_id].ecef_at_time(time_min)
        self._ecef_cache.set_ecef(sat_id, timeslot, ecef)
        return ecef

    def get_all_ecef_at_timeslot(self, timeslot: int) -> dict[int, tuple[float, float, float]]:
        """获取某时刻所有卫星的 ECEF 坐标。"""
        return {
            sat_id: self.get_sat_ecef(sat_id, timeslot) for sat_id in range(self.num_satellites)
        }

    def get_sat_trajectory(
        self, sat_id: int, timeslots: Optional[list[int]] = None
    ) -> list[tuple[float, float]]:
        """
        获取卫星轨迹（一系列星下点）。

        Parameters
        ----------
        sat_id : int
            卫星 ID。
        timeslots : List[int], optional
            时刻列表，默认所有 timeslot。

        Returns
        -------
        List[Tuple[float, float]]
            [(lat_deg, lon_deg), ...]
        """
        if timeslots is None:
            timeslots = list(range(self.num_timeslots))
        return [self.get_sat_position(sat_id, ts) for ts in timeslots]

    def get_communication_record(self, sat_id: int) -> list[dict]:
        """
        获取某卫星的完整通信记录。

        满足用户需求: "每个卫星与哪些基站传输了信息，这些也要记录下来"

        Returns
        -------
        List[Dict]
            每项含 {timeslot, gs_ids, gs_names, sat_position}。
        """
        records = []
        for ts in range(self.num_timeslots):
            gs_ids = self.get_all_contacts(sat_id, ts)
            if gs_ids:
                records.append(
                    {
                        "timeslot": ts,
                        "time_min": ts * self.timeslot_duration_min,
                        "gs_ids": gs_ids,
                        "gs_names": [
                            self.ground_network.names[gid]
                            for gid in gs_ids
                            if 0 <= gid < len(self.ground_network.names)
                        ],
                        "sat_position_deg": self.get_sat_position(sat_id, ts),
                    }
                )
        return records

    # ============================================================
    # 摘要与导出
    # ============================================================

    def summary(self) -> str:
        """返回模拟器配置摘要。"""
        backend_label = {
            "kepler": "Kepler (轻量开普勒力学)",
            "skyfield": "Skyfield (SGP4 + JPL DE421精确星历)",
        }.get(self.backend_mode, self.backend_mode)

        lines = [
            "=" * 60,
            "  SpaceFL 轨道模拟器 配置摘要",
            "=" * 60,
            f"  轨道后端:           {backend_label}",
            f"  中心天体:           {self.body.name}",
            f"    半径:              {self.body.radius_km} km",
            f"    GM:                {self.body.GM:.1f} km^3/s^2",
            f"    自转周期:          {self.body.rotation_period_hours} h",
            f"    大气层高度:        {self.body.atmosphere_height_km} km",
            "",
            f"  卫星数量:           {self.num_satellites}",
            f"  轨道高度:           {self.orbit_altitude_km} km",
            f"  轨道倾角:           {self.orbit_inclination_deg}°",
            f"  轨道周期:           {self.orbit_period_min:.1f} min",
            f"  分布策略:           {self.constellation_config.distribution if self.constellation_config else '自定义轨道'}",
            "",
            f"  地面站数量:         {self.num_ground_stations}",
            f"  Timeslot粒度:       {self.timeslot_duration_min} min",
            f"  总Timeslot数:       {self.num_timeslots}",
            f"  模拟时长:           {self.num_timeslots * self.timeslot_duration_min / 60:.1f} h",
            f"  接触矩阵模式:       {self.contact_mode}",
            f"  模拟起始日期:       {self.sim_start_date}",
            f"  星历来源:           {'JPL DE421 (Skyfield)' if self.backend_mode == 'skyfield' else '开普勒二体'}",
            "",
            "  地面站列表:",
        ]

        for i, gs in enumerate(self.ground_network):
            lines.append(
                f"    [{i}] {gs.name}: "
                f"({gs.lat_deg:+.2f}°, {gs.lon_deg:+.2f}°)"
                + (f" min_el={gs.min_elevation_deg}°" if gs.min_elevation_deg > 0 else "")
            )

        lines.extend(
            [
                "",
                "  接触统计:",
                f"    总接触次数:        {self.stats['total_contacts']}",
                f"    每卫星平均接触:    {self.stats['avg_contacts_per_sat']:.1f}",
                f"    总体接触率:        {self.stats['contact_rate'] * 100:.1f}%",
                f"    生成耗时:          {self._gen_time:.1f}s",
                "",
                f"  ISL 星间链路:       {'启用' if self.isl_config.enabled else '禁用'}",
            ]
        )
        if self.isl_config.enabled:
            isl_info = self.isl_stats
            lines.extend(
                [
                    f"    计算器:            {self.isl_config.calculator}",
                    f"    大气余量:          {self.isl_config.atmosphere_buffer_km} km",
                    f"    窗口数:            {isl_info['total_windows']}",
                    f"    总可见时长:        {isl_info['total_duration_s'] / 3600:.2f} h",
                    f"    独特链路数:        {isl_info['unique_links']}",
                ]
            )
        lines.append("=" * 60)

        return "\n".join(lines)

    def export(self, filepath: str):
        """导出接触矩阵等完整数据到 JSON。"""
        self.contact_matrix.save_json(filepath, gs_names=self.ground_network.names)

    def export_contact_matrix(self, filepath: str):
        """兼容原 orbit_sim_v2.py 的导出接口。"""
        import json

        data = {
            "config": {
                "body": self.body.to_dict(),
                "num_satellites": self.num_satellites,
                "num_ground_stations": self.num_ground_stations,
                "orbit_altitude_km": self.orbit_altitude_km,
                "orbit_period_min": self.orbit_period_min,
                "timeslot_duration_min": self.timeslot_duration_min,
                "num_timeslots": self.num_timeslots,
                "distribution": self.constellation_config.distribution,
            },
            "ground_stations": [
                {"id": i, "name": gs.name, "lat": gs.lat_deg, "lon": gs.lon_deg}
                for i, gs in enumerate(self.ground_network)
            ],
            "contact_matrix_simple": self.contact_matrix.simple_matrix.tolist(),
            "statistics": self.stats,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        return (
            f"OrbitSimulator(backend={self.backend_mode}, body={self.body.name}, "
            f"{self.num_satellites}sats x {self.num_ground_stations}gs, "
            f"alt={self.orbit_altitude_km}km, "
            f"contact_rate={self.stats['contact_rate']:.1%})"
        )


# ============================================================
# 工厂函数
# ============================================================


def create_default_simulator() -> OrbitSimulator:
    """
    创建默认配置模拟器（与原 orbit_sim_v2.py 兼容）。

    Returns
    -------
    OrbitSimulator
        地球 + 3卫星 + 2地面站，24小时模拟。
    """
    return OrbitSimulator(
        body=CelestialBody.earth(),
        num_satellites=3,
        num_ground_stations=2,
        orbit_altitude_km=500.0,
        orbit_inclination_deg=90.0,
        distribution="uniform",
        timeslot_duration_min=1.0,
        num_timeslots=1440,
        contact_mode="full",
        random_seed=42,
    )


def create_mars_simulator() -> OrbitSimulator:
    """创建火星模拟器示例。"""
    from fl_space.environment.ground_station import GroundStation, GroundStationNetwork

    # 火星地面站（假设的未来基站位置）
    mars_gs = GroundStationNetwork(
        [
            GroundStation("Olympus Base", 18.65, -133.8),  # Olympus Mons
            GroundStation("Valles Marineris", -14.0, -59.0),  # 水手谷
            GroundStation("Gale Crater", -4.6, 137.4),  # 好奇号着陆点
            GroundStation("Jezero Crater", 18.4, 77.6),  # 毅力号着陆点
            GroundStation("Utopia Planitia", 49.7, 118.0),  # 祝融号着陆点
        ]
    )

    return OrbitSimulator(
        body=CelestialBody.mars(),
        constellation_config=ConstellationConfig(
            num_satellites=5,
            num_planes=2,
            inclination_deg=60.0,
            altitude_km=300.0,
            distribution="walker",
            phasing_factor=1,
        ),
        ground_station_network=mars_gs,
        timeslot_duration_min=1.0,
        num_timeslots=1440,
        contact_mode="full",
    )
