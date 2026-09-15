"""
SpaceX API 只读适配器 — 将公开目录数据转换为研究对象。

默认服务地址为 ``https://api.spacexdata.com/v4``。适配器不把该网络服务
作为核心依赖；离线研究可以直接传入 API 返回的字典或使用本地快照。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from fl_space.mission import SatelliteDesign, VisualAppearance


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
