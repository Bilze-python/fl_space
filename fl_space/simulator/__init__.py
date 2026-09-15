"""
模拟器层 — 接触矩阵管理和轨道模拟主引擎

主要导出:
    OrbitSimulator  — 模块化轨道接触模拟器（主类）
    ContactMatrix   — 接触矩阵（兼容/完整两种模式）
"""

from .contact_matrix import ContactMatrix as ContactMatrix
from .orbit_simulator import (
    OrbitSimulator as OrbitSimulator,
)
from .orbit_simulator import (
    create_default_simulator as create_default_simulator,
)
from .orbit_simulator import (
    create_mars_simulator as create_mars_simulator,
)

__all__ = [
    "ContactMatrix",
    "OrbitSimulator",
    "create_default_simulator",
    "create_mars_simulator",
]
