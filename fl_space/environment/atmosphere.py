"""
大气层模型 — 可扩展的大气层建模框架

用途:
    - 影响卫星可见性计算的地平角
    - 信号衰减估算（未来扩展）
    - 轨道寿命估算（未来扩展）

支持:
    - 无大气层模型 (NoAtmosphere)
    - 恒定高度模型 (ConstantHeightAtmosphere)
    - 指数衰减模型 (ExponentialAtmosphere)
"""

from dataclasses import dataclass
import math


@dataclass
class AtmosphereModel:
    """
    大气层基础模型。

    Attributes
    ----------
    name : str
        模型名称。
    scale_height_km : float
        大气标高 (km)，即密度减小到 1/e 的高度。
    base_density_kgm3 : float
        海平面大气密度 (kg/m³)。
    """

    name: str = "vacuum"
    scale_height_km: float = 0.0
    base_density_kgm3: float = 0.0

    def effective_height_km(self) -> float:
        """
        返回有效大气层高度 (km)。

        对于卫星可见性计算，此高度用于修正地平可见半角。
        """
        return 0.0

    def density_at_altitude(self, altitude_km: float) -> float:
        """
        计算给定高度的大气密度 (kg/m³)。

        Parameters
        ----------
        altitude_km : float
            高度 (km)。

        Returns
        -------
        float
            大气密度。
        """
        return 0.0

    def __repr__(self) -> str:
        return f"AtmosphereModel(name='{self.name}')"


class NoAtmosphere(AtmosphereModel):
    """无大气层模型（适用于月球等无大气天体）。"""
    def __init__(self):
        super().__init__(name="vacuum")

    def effective_height_km(self) -> float:
        return 0.0

    def density_at_altitude(self, altitude_km: float) -> float:
        return 0.0


class ConstantHeightAtmosphere(AtmosphereModel):
    """
    恒定高度大气层模型。

    假设大气层在某个固定高度处突然截断。
    这是最简单、计算最快的大气模型，适合轨道接触判断。
    """

    def __init__(self, height_km: float = 100.0):
        """
        Parameters
        ----------
        height_km : float
            大气层有效高度 (km)。地球默认为 100km (Kármán线)。
        """
        super().__init__(name="constant", scale_height_km=height_km)
        self._height_km = height_km

    def effective_height_km(self) -> float:
        return self._height_km

    def density_at_altitude(self, altitude_km: float) -> float:
        return 1.0 if altitude_km <= self._height_km else 0.0


class ExponentialAtmosphere(AtmosphereModel):
    """
    指数衰减大气层模型。

    密度随高度指数衰减: ρ(h) = ρ₀ · exp(-h/H)
    """

    def __init__(
        self,
        scale_height_km: float = 8.5,
        base_density_kgm3: float = 1.225,
        cutoff_height_km: float = 200.0,
    ):
        """
        Parameters
        ----------
        scale_height_km : float
            大气标高 (km)。地球约 8.5km。
        base_density_kgm3 : float
            海平面密度 (kg/m³)。地球约 1.225。
        cutoff_height_km : float
            截断高度，超过此高度视为真空。
        """
        super().__init__(
            name="exponential",
            scale_height_km=scale_height_km,
            base_density_kgm3=base_density_kgm3,
        )
        self.cutoff_height_km = cutoff_height_km

    def effective_height_km(self) -> float:
        """返回截止高度作为有效大气层高度。"""
        return self.cutoff_height_km

    def density_at_altitude(self, altitude_km: float) -> float:
        if altitude_km > self.cutoff_height_km:
            return 0.0
        return self.base_density_kgm3 * math.exp(
            -altitude_km / self.scale_height_km
        )


def create_atmosphere_for_body(body_name: str) -> AtmosphereModel:
    """
    根据天体名称创建合适的大气模型。

    Parameters
    ----------
    body_name : str
        天体名称 ("earth", "mars", "moon", ...)。

    Returns
    -------
    AtmosphereModel
    """
    name_lower = body_name.lower()
    if name_lower == "earth":
        return ExponentialAtmosphere(
            scale_height_km=8.5,
            base_density_kgm3=1.225,
            cutoff_height_km=100.0,
        )
    elif name_lower == "mars":
        return ExponentialAtmosphere(
            scale_height_km=11.1,
            base_density_kgm3=0.020,
            cutoff_height_km=80.0,
        )
    elif name_lower == "moon":
        return NoAtmosphere()
    else:
        return NoAtmosphere()
