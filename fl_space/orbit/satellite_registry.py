"""
卫星注册表 — 用户可扩展的卫星类型系统。

用户可以通过装饰器或函数调用注册自定义卫星类型，
然后用名称引用即可在模拟中使用。

Examples
--------
>>> from fl_space.orbit.satellite_registry import registry

>>> # 方式1: 装饰器注册
>>> @registry.register("my_leo")
... def my_leo_sats(body):
...     from fl_space.orbit.satellite_config import ClusterSpec, MultiClusterConfig
...     return MultiClusterConfig(clusters=[
...         ClusterSpec("leo", num_satellites=8, altitude_km=400, inclination_deg=51)
...     ])

>>> # 方式2: 函数注册
>>> registry.register_func("starlink_v2", my_starlink_func)

>>> # 方式3: 从字典注册
>>> registry.register_from_dict("custom_geo", {
...     "clusters": [{"name": "geo", "num_satellites": 3, "altitude_km": 35786, "inclination_deg": 0}]
... })

>>> # 使用
>>> config = registry.get("my_leo", body=earth)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from fl_space.environment import CelestialBody

    from .satellite_config import MultiClusterConfig


class SatelliteRegistry:
    """卫星类型注册表。

    全局单例，用户可注册自定义卫星类型并通过名称获取配置。
    """

    def __init__(self):
        self._registry: dict[str, Callable[..., MultiClusterConfig]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

        # 注册内置类型
        self._register_builtins()

    def _register_builtins(self):
        """注册内置卫星类型。"""
        from .satellite_config import MultiClusterConfig

        self._registry["polar_6"] = lambda body=None: MultiClusterConfig.polar_only(6)
        self._metadata["polar_6"] = {"description": "6颗极轨卫星 (500km, 90deg)", "category": "builtin"}

        self._registry["starlink_20"] = lambda body=None: MultiClusterConfig.starlink_like(20)
        self._metadata["starlink_20"] = {"description": "Starlink风格20卫星 (550km, 53deg)", "category": "builtin"}

        self._registry["mixed_10"] = lambda body=None: MultiClusterConfig.mixed_orbit(4, 4, 2)
        self._metadata["mixed_10"] = {"description": "混合轨道10卫星 (极轨4+中倾角4+赤道2)", "category": "builtin"}

        self._registry["demo"] = lambda body=None: MultiClusterConfig.demo_default()
        self._metadata["demo"] = {"description": "演示默认: 极轨4+LEO壳6", "category": "builtin"}

    def register(
        self, name: str, description: str = "", category: str = "custom"
    ) -> Callable:
        """装饰器：注册自定义卫星生成函数。

        Parameters
        ----------
        name : str
            卫星类型名称（唯一标识）。
        description : str
            描述信息。
        category : str
            分类标签。

        Returns
        -------
        decorator : Callable
        """

        def decorator(func: Callable[..., MultiClusterConfig]):
            self._registry[name] = func
            self._metadata[name] = {"description": description, "category": category}
            return func

        return decorator

    def register_func(
        self,
        name: str,
        func: Callable[..., MultiClusterConfig],
        description: str = "",
        category: str = "custom",
    ):
        """函数式注册：注册自定义卫星生成函数。

        Parameters
        ----------
        name : str
            卫星类型名称。
        func : Callable
            生成函数，签名: func(body: CelestialBody | None) -> MultiClusterConfig。
        description : str
            描述信息。
        category : str
            分类标签。
        """
        self._registry[name] = func
        self._metadata[name] = {"description": description, "category": category}

    def register_from_dict(self, name: str, data: dict, description: str = ""):
        """从字典注册：直接提供 MultiClusterConfig 的字典表示。

        这是最简单的注册方式，适合 JSON/YAML 配置文件。

        Parameters
        ----------
        name : str
            卫星类型名称。
        data : dict
            MultiClusterConfig.to_dict() 格式的字典。
        description : str
            描述信息。
        """
        from .satellite_config import MultiClusterConfig

        config = MultiClusterConfig.from_dict(data)
        self._registry[name] = lambda body=None: config
        self._metadata[name] = {"description": description, "category": "dict"}

    def get(
        self, name: str, body: Optional[CelestialBody] = None
    ) -> MultiClusterConfig:
        """获取指定名称的卫星配置。

        Parameters
        ----------
        name : str
            卫星类型名称。
        body : CelestialBody, optional
            中心天体（传递给生成函数）。

        Returns
        -------
        MultiClusterConfig

        Raises
        ------
        KeyError
            如果名称未注册。
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"Unknown satellite type: '{name}'. Available: {available}"
            )
        return self._registry[name](body)

    def list_types(self, category: Optional[str] = None) -> list[dict]:
        """列出所有已注册的卫星类型。

        Parameters
        ----------
        category : str, optional
            按分类过滤。

        Returns
        -------
        list of dict
            [{"name": ..., "description": ..., "category": ...}, ...]
        """
        result = []
        for name, meta in self._metadata.items():
            if category is None or meta.get("category") == category:
                result.append({
                    "name": name,
                    "description": meta.get("description", ""),
                    "category": meta.get("category", "unknown"),
                })
        return sorted(result, key=lambda x: x["name"])

    def unregister(self, name: str):
        """取消注册（仅限用户自定义类型，内置类型不可移除）。

        Parameters
        ----------
        name : str
            卫星类型名称。
        """
        meta = self._metadata.get(name, {})
        if meta.get("category") == "builtin":
            raise ValueError(f"Cannot unregister builtin type: '{name}'")
        self._registry.pop(name, None)
        self._metadata.pop(name, None)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        types = self.list_types()
        lines = [f"SatelliteRegistry({len(types)} types):"]
        for t in types:
            lines.append(f"  {t['name']:20s} [{t['category']}] {t['description']}")
        return "\n".join(lines)


# 全局单例
registry = SatelliteRegistry()
