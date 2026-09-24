"""
SpaceX API 只读适配器 — 将公开目录数据转换为研究对象。

默认服务地址为 ``https://api.spacexdata.com/v4``。适配器不把该网络服务
作为核心依赖；离线研究可以直接传入 API 返回的字典或使用本地快照。
"""

from __future__ import annotations

from collections.abc import Callable
import json
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from fl_space.mission import (
    ClassicalOrbitalElements,
    CommunicationSystem,
    SatelliteDesign,
    VisualAppearance,
)


class SpaceXApiClient:
    """SpaceX API 的最小只读客户端。"""

    def __init__(
        self,
        base_url: str = "https://api.spacexdata.com/v4/",
        *,
        timeout_s: float = 10.0,
        opener: Callable[[Request, float], object] | None = None,
    ) -> None:
        """初始化客户端。"""
        if timeout_s <= 0:
            raise ValueError("timeout_s 必须为正")
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_s = timeout_s
        self._opener = opener or urlopen

    def get(self, resource: str) -> dict | list:
        """获取一个只读资源并解析 JSON。"""
        endpoint = resource.lstrip("/")
        request = Request(
            urljoin(self.base_url, endpoint),
            headers={"Accept": "application/json", "User-Agent": "fl-space/0.1"},
        )
        with self._opener(request, self.timeout_s) as response:
            payload = response.read()
        return json.loads(payload.decode("utf-8"))

    def list_launches(self) -> list[dict]:
        """获取发射记录。"""
        return self._expect_list(self.get("launches"))

    def list_rockets(self) -> list[dict]:
        """获取火箭记录。"""
        return self._expect_list(self.get("rockets"))

    def list_payloads(self) -> list[dict]:
        """获取载荷记录。"""
        return self._expect_list(self.get("payloads"))

    def list_starlink(self) -> list[dict]:
        """获取 Starlink 记录。"""
        return self._expect_list(self.get("starlink"))

    @staticmethod
    def _expect_list(value: dict | list) -> list[dict]:
        """校验列表接口返回格式。"""
        if not isinstance(value, list):
            raise ValueError("SpaceX API 返回的数据不是列表")
        return [item for item in value if isinstance(item, dict)]


def satellite_design_from_payload(record: dict) -> SatelliteDesign:
    """把 SpaceX API payload 记录转换成卫星设计草案。

    API 中没有统一的卫星外形、迎风面积和功率字段，因此这些字段采用
    可追踪的研究默认值，并在 ``metadata`` 中保留数据来源和 API ID。
    """
    name = str(record.get("name") or record.get("type") or "spacex_payload")
    mass_kg = record.get("mass_kg")
    if mass_kg is None and record.get("mass_lbs") is not None:
        mass_kg = float(record["mass_lbs"]) * 0.45359237
    mass_kg = max(float(mass_kg or 1.0), 0.001)
    return SatelliteDesign(
        name=name,
        dry_mass_kg=mass_kg,
        cross_section_area_m2=1.0,
        visual=VisualAppearance(
            primitive="box",
            dimensions_m=(1.0, 1.0, 1.0),
            color_hex="#8ea3b7",
        ),
        metadata={
            "source": "r-spacex/SpaceX-API",
            "source_id": str(record.get("id", "")),
            "source_type": "payload",
            "fidelity": "catalog_mapping",
        },
    )


def orbital_elements_from_starlink(record: dict) -> ClassicalOrbitalElements:
    """从 SpaceX API Starlink 记录提取二体六根数。

    ``r-spacex/SpaceX-API`` 的 Starlink schema 包含 Space-Track OMM 字段：
    ``SEMIMAJOR_AXIS``、``ECCENTRICITY``、``INCLINATION``、``RA_OF_ASC_NODE``、
    ``ARG_OF_PERICENTER`` 和 ``MEAN_ANOMALY``。当半长轴缺失但 ``height_km``
    存在时，本函数退化为圆轨道近似，并在调用方报告中标注保真度。
    """
    space_track = record.get("spaceTrack") or {}
    semi_major_axis_km = space_track.get("SEMIMAJOR_AXIS")
    if semi_major_axis_km is None:
        height_km = record.get("height_km")
        if height_km is None:
            raise ValueError("Starlink 记录缺少 SEMIMAJOR_AXIS 和 height_km，无法构造轨道")
        return ClassicalOrbitalElements.from_circular_altitude(
            float(height_km),
            inclination_deg=float(space_track.get("INCLINATION") or 53.0),
            raan_deg=float(space_track.get("RA_OF_ASC_NODE") or 0.0),
            anomaly_deg=float(space_track.get("MEAN_ANOMALY") or 0.0),
            epoch_utc=space_track.get("EPOCH"),
        )
    return ClassicalOrbitalElements(
        semi_major_axis_km=float(semi_major_axis_km),
        eccentricity=float(space_track.get("ECCENTRICITY") or 0.0),
        inclination_deg=float(space_track.get("INCLINATION") or 0.0),
        raan_deg=float(space_track.get("RA_OF_ASC_NODE") or 0.0),
        arg_perigee_deg=float(space_track.get("ARG_OF_PERICENTER") or 0.0),
        anomaly_deg=float(space_track.get("MEAN_ANOMALY") or 0.0),
        anomaly_kind="mean",
        epoch_utc=space_track.get("EPOCH"),
    )


def public_starlink_optical_link_profile(
    *,
    data_rate_mbps: float = 25_000.0,
    max_range_km: float = 4_000.0,
    source: str = "public_report_assumption",
) -> CommunicationSystem:
    """构造公开资料驱动的 Starlink 光链路候选配置。

    该对象用于仿真初评，不代表 SpaceX 官方工程规格。公开资料通常只给出
    峰值速率、最大距离或终端数量，缺失的波长、发射功率、调制编码和指向
    误差必须保留为假设。
    """
    return CommunicationSystem(
        name="starlink_public_optical_isl",
        link_kind="inter_satellite",
        band="optical",
        frequency_hz=193.5e12,
        channel_bandwidth_mhz=0.0,
        transmit_power_w=1.0,
        transmit_antenna_gain_dbi=0.0,
        receive_antenna_gain_dbi=0.0,
        downlink_rate_mbps=data_rate_mbps,
        uplink_rate_mbps=data_rate_mbps,
        inter_satellite_link=True,
        isl_rate_mbps=data_rate_mbps,
        max_simultaneous_links=1,
        terminal_count=1,
        max_range_km=max_range_km,
        weather_sensitive=False,
        source=source,
    )


__all__ = [
    "SpaceXApiClient",
    "orbital_elements_from_starlink",
    "public_starlink_optical_link_profile",
    "satellite_design_from_payload",
]
