"""
Flower 联邦学习框架适配器。

将 SpaceFL 的 ContactMatrix + OrbitSimulator 数据结构
映射为 Flower 可消费的调度接口。

设计：
    - 不重算轨道/仰角/LOS — 只消费现有数据
    - 可选 ISL 支持
    - 与 SpaceFL 核心零耦合（纯适配器模式）

用法::

    from fl_space.integrations.flower import FlowerAdapter
    from fl_space.simulator import OrbitSimulator

    sim = OrbitSimulator(...)
    adapter = FlowerAdapter.from_simulator(sim)
    visible = adapter.clients_visible_at(some_time)
"""

from fl_space.integrations.flower.adapter import (
    ContactWindow,
    FlowerAdapter,
    IntraLinkWindow,
)

__all__ = [
    "ContactWindow",
    "FlowerAdapter",
    "IntraLinkWindow",
]
