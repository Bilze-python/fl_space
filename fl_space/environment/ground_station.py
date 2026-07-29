"""
地面站模块 — 地面站定义与网络管理

提供:
    - GroundStation: 单个地面站定义
    - GroundStationNetwork: 地面站网络管理
    - 内置预设地面站集合
"""

from dataclasses import dataclass
import json
import math
from typing import Optional, Union


@dataclass
class GroundStation:
    """
    单个地面站定义。

    Attributes
    ----------
    name : str
        地面站名称。
    lat_deg : float
        地理纬度 (°)，北纬为正。
    lon_deg : float
        地理经度 (°)，东经为正。
    altitude_km : float
        地面站海拔高度 (km)。
    min_elevation_deg : float
        最小通信仰角 (°)，低于此角度视为不可见。
    antenna_gain_dbi : float
        天线增益 (dBi)，用于信号链路预算（可选）。
    num_antennas : int
        天线数量（可并行指向的独立天线数），默认 1。
    max_concurrent_sats : int
        单站同一时刻最大并发接入卫星数，默认等于 num_antennas。
    downlink_rate_mbps : float
        单条链路下行速率上限 (Mbps)，用于预估下行数据量。
    maintenance_windows : list[tuple[int, int]]
        运维/检修禁用时段列表，元素为 (start_timeslot, end_timeslot)（闭区间），
        落入其中的时刻不建立链路。默认无。
    """
    name: str
    lat_deg: float
    lon_deg: float
    altitude_km: float = 0.0
    min_elevation_deg: float = 0.0
    antenna_gain_dbi: float = 0.0
    # --- 可调度资源字段 ---
    num_antennas: int = 1
    max_concurrent_sats: int = 0
    downlink_rate_mbps: float = 100.0
    maintenance_windows: Optional[list[tuple[int, int]]] = None

    def __post_init__(self) -> None:
        # max_concurrent_sats 缺省时取天线数
        if self.max_concurrent_sats <= 0:
            self.max_concurrent_sats = max(1, self.num_antennas)
        if self.maintenance_windows is None:
            self.maintenance_windows = []

    def is_available_at(self, timeslot: int) -> bool:
        """判断该地面站在指定 timeslot 是否处于可用（非检修）状态。"""
        return all(
            not (start <= timeslot <= end)
            for start, end in self.maintenance_windows or []
        )

    @property
    def lat_rad(self) -> float:
        return math.radians(self.lat_deg)

    @property
    def lon_rad(self) -> float:
        return math.radians(self.lon_deg)

    @property
    def min_elevation_rad(self) -> float:
        return math.radians(self.min_elevation_deg)

    @property
    def coords_rad(self) -> tuple[float, float]:
        """返回 (纬度弧度, 经度弧度)。"""
        return (self.lat_rad, self.lon_rad)

    def ecef(self) -> tuple[float, float, float]:
        """返回 WGS84 椭球下的 ECEF 直角坐标 (x, y, z)，单位 km。

        公式：
            N = a / √(1 - e²·sin²φ)
            X = (N + h)·cosφ·cosλ
            Y = (N + h)·cosφ·sinλ
            Z = [N·(1 - e²) + h]·sinφ

        其中 a = 6378.137 km, f = 1/298.257223563。
        """
        from fl_space.utils.coordinate_utils import geodetic_to_ecef

        return geodetic_to_ecef(self.lat_deg, self.lon_deg, self.altitude_km)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lat_deg": self.lat_deg,
            "lon_deg": self.lon_deg,
            "altitude_km": self.altitude_km,
            "min_elevation_deg": self.min_elevation_deg,
            "antenna_gain_dbi": self.antenna_gain_dbi,
            "num_antennas": self.num_antennas,
            "max_concurrent_sats": self.max_concurrent_sats,
            "downlink_rate_mbps": self.downlink_rate_mbps,
            "maintenance_windows": self.maintenance_windows,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GroundStation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def __repr__(self) -> str:
        return f"GS({self.name}, {self.lat_deg:.1f}°, {self.lon_deg:.1f}°)"


class GroundStationNetwork:
    """
    地面站网络 — 管理一组地面站。

    支持:
        - 按索引/名称查询
        - 地理过滤
        - 导入/导出配置
    """

    def __init__(self, stations: Optional[list[GroundStation]] = None):
        """
        Parameters
        ----------
        stations : list of GroundStation, optional
            初始地面站列表。
        """
        self._stations: list[GroundStation] = []
        self._name_index: dict[str, int] = {}
        if stations:
            for gs in stations:
                self.add_station(gs)

    def add_station(self, gs: GroundStation):
        """添加一个地面站。"""
        self._stations.append(gs)
        self._name_index[gs.name] = len(self._stations) - 1

    def remove_station(self, name: str) -> bool:
        """按名称移除地面站。返回是否成功。"""
        if name in self._name_index:
            idx = self._name_index.pop(name)
            self._stations.pop(idx)
            # 重建索引
            self._name_index = {
                gs.name: i for i, gs in enumerate(self._stations)
            }
            return True
        return False

    def __getitem__(self, key: Union[int, str]) -> GroundStation:
        if isinstance(key, int):
            return self._stations[key]
        elif isinstance(key, str):
            return self._stations[self._name_index[key]]
        raise TypeError(f"Unsupported key type: {type(key)}")

    def __len__(self) -> int:
        return len(self._stations)

    def __iter__(self):
        return iter(self._stations)

    def __repr__(self) -> str:
        return f"GroundStationNetwork({len(self)} stations)"

    # --- 属性 ---

    @property
    def count(self) -> int:
        return len(self._stations)

    @property
    def names(self) -> list[str]:
        return [gs.name for gs in self._stations]

    @property
    def coords_rad(self) -> list[tuple[float, float]]:
        """返回所有地面站的 (纬度弧度, 经度弧度) 列表。"""
        return [gs.coords_rad for gs in self._stations]

    @property
    def coords_deg(self) -> list[tuple[float, float]]:
        """返回所有地面站的 (纬度°, 经度°) 列表。"""
        return [(gs.lat_deg, gs.lon_deg) for gs in self._stations]

    def to_list(self) -> list[GroundStation]:
        """返回地面站列表（副本）。"""
        return list(self._stations)

    def to_dict_list(self) -> list[dict]:
        """导出为字典列表。"""
        return [gs.to_dict() for gs in self._stations]

    # --- IO ---

    @classmethod
    def from_tuples(
        cls, stations: list[tuple[str, float, float]]
    ) -> "GroundStationNetwork":
        """
        从 (名称, 纬度, 经度) 元组列表快速创建。

        Parameters
        ----------
        stations : list of (str, float, float)
            地面站列表，每个元素为 (名称, 纬度°, 经度°)。

        Returns
        -------
        GroundStationNetwork
        """
        return cls([
            GroundStation(name=name, lat_deg=lat, lon_deg=lon)
            for name, lat, lon in stations
        ])

    def save_json(self, filepath: str):
        """保存到 JSON 文件。"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict_list(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, filepath: str) -> "GroundStationNetwork":
        """从 JSON 文件加载。"""
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        return cls([GroundStation.from_dict(d) for d in data])


# ============================================================
# 内置预设地面站
# ============================================================

# 论文 Table 3 中的地面站
PAPER_GROUND_STATIONS = [
    ("Sioux Falls", 43.55, -96.72),
    ("Sanya", 18.25, 109.5),
    ("Johannesburg", -26.2, 28.03),
    ("Cordoba", -31.4, -64.18),
    ("Tromso", 69.65, 18.95),
    ("Kashi", 39.1, 77.2),
    ("Beijing", 39.9, 116.4),
    ("Neustrelitz", 53.1, 13.1),
    ("Parepare", -2.99, 119.8),
    ("Alice Springs", -25.1, 133.9),
    ("Fairbanks", 64.8, -147.7),
    ("Prince Albert", 53.2, -105.7),
    ("Shadnagar", 17.4, 78.5),
]

# 扩展的准全球覆盖地面站（论文扩展实验用）
EXTENDED_GROUND_STATIONS = [*PAPER_GROUND_STATIONS, ("Anchorage", 61.2, -149.9), ("Honolulu", 21.3, -157.8), ("Reykjavik", 64.1, -21.9), ("Mumbai", 19.1, 72.9), ("Tokyo", 35.7, 139.7), ("Santiago", -33.4, -70.7)]


def create_default_network(n: int = 7) -> GroundStationNetwork:
    """
    创建默认的 n 个地面站网络。

    Parameters
    ----------
    n : int
        前 n 个地面站（从论文 Table 3 中选取）。

    Returns
    -------
    GroundStationNetwork
    """
    return GroundStationNetwork.from_tuples(PAPER_GROUND_STATIONS[:n])


def create_extended_network(n: int = 13) -> GroundStationNetwork:
    """
    创建扩展的 n 个地面站网络（含准全球覆盖）。

    Parameters
    ----------
    n : int
        前 n 个地面站。

    Returns
    -------
    GroundStationNetwork
    """
    return GroundStationNetwork.from_tuples(EXTENDED_GROUND_STATIONS[:n])
