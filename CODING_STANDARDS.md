# SpaceFL 代码规范与开发指南

> **目标受众**：新加入的开发人员、使用 AI 辅助编程的协作者  
> **最后更新**：2026-06-01  
> **强制级别**：所有新代码和 AI 生成的代码**必须**遵循本规范

---

## 目录

1. [项目架构概述](#1-项目架构概述)
2. [命名规范](#2-命名规范)
3. [类型注解](#3-类型注解)
4. [文档字符串](#4-文档字符串)
5. [导入规范](#5-导入规范)
6. [数据模型模式](#6-数据模型模式)
7. [错误处理](#7-错误处理)
8. [代码质量工具](#8-代码质量工具)
9. [测试规范](#9-测试规范)
10. [AI 辅助开发准则](#10-ai-辅助开发准则)
11. [提交前检查清单](#11-提交前检查清单)

---

## 1. 项目架构概述

### 1.1 四层架构

```
fl_space/
├── environment/   # L1 — 环境模拟层：天体、大气、地面站、坐标工具
├── orbit/         # L2 — 轨道力学层：Kepler/SGP4双后端、星座、可见性
├── simulator/     # L3 — 模拟器层：主模拟引擎、接触矩阵
├── config/        # L4 — 配置层：预设参数、配置加载器
├── mission/       # 任务域层：火箭、载荷、卫星设计、部署与低阶物理
└── viz/           # 可视化层（纯matplotlib）
```

### 1.2 核心设计原则

| 原则 | 说明 |
|------|------|
| **分层解耦** | 每层只依赖下层，使用显式接口 |
| **双后端** | `kepler`（默认，零外部依赖）和 `skyfield`（高精度，可选安装） |
| **工厂优先** | 通过 `@classmethod` 工厂和独立工厂函数创建对象 |
| **序列化友好** | 核心数据类支持 `to_dict()/from_dict()` |
| **中文文档** | 注释和文档字符串使用中文，便于国内研究者阅读 |
| **物理与规则分离** | 物理服务、渲染数据和经营/游戏规则保持独立 |

### 1.3 Python 版本

- **最低版本**：Python 3.9
- **核心依赖**：`numpy>=1.24.0`, `matplotlib>=3.7.0`
- **可选依赖**：`skyfield>=1.48`（高精度后端）, `torch>=2.0.0`（FL算法）

---

## 2. 命名规范

### 2.0 航天物理单位

`fl_space/mission/` 新代码使用 SI 单位，字段必须带单位后缀：
`_m`、`_km`、`_kg`、`_s`、`_mps`、`_mps2`、`_w`、`_m2`。
与既有轨道模块交界时显式转换，禁止复用变量名却更换单位。

### 2.1 基本规则

| 类型 | 规范 | 示例 |
|------|------|------|
| 文件 | `snake_case` | `kepler_orbit.py`, `ground_station.py` |
| 类 | `PascalCase` | `OrbitSimulator`, `ContactMatrix`, `CelestialBody` |
| 函数/方法 | `snake_case` | `position_at_time`, `get_communication_record` |
| 变量 | `snake_case` | `contact_rate`, `earth_radius_km` |
| 常量 | `UPPER_SNAKE_CASE` | `PAPER_GROUND_STATIONS`, `SKYFIELD_AVAILABLE` |
| 私有成员 | 前缀 `_` | `_period_min`, `_sf_backend` |
| 工厂方法 | `from_*` (classmethod) 或 `create_*` (函数) | `CelestialBody.from_dict()`, `create_default_network()` |
| `__all__` | 每个 `__init__.py` 必须定义 | 见 `fl_space/orbit/__init__.py` |

### 2.2 科学符号例外

以下标准轨道力学符号允许违反 `UPPER_SNAKE_CASE` 常量命名规则（已在 Ruff `ignore-names` 中豁免）：

```python
GM  # 引力常数 G × M
M, M0  # 平近点角 (Mean anomaly) 及初值
E, dE  # 偏近点角 (Eccentric anomaly) 及增量
cos_E  # cos(偏近点角)
sin_E  # sin(偏近点角)
N  # 轨道面内卫星数
setUp  # unittest 钩子
tearDown  # unittest 钩子
```

> **规则**：只有上述已豁免的符号可以使用大写命名。其他所有变量/常量必须遵循 `UPPER_SNAKE_CASE`。

### 2.3 禁止事项

- ❌ 单字母变量名（除循环索引 `i`, `j` 和上述豁免符号外）
- ❌ 匈牙利命名法（`strName`, `iCount`）
- ❌ 拼音变量名
- ❌ 无意义的缩写（`gsn`，应写成 `ground_station_network`）

---

## 3. 类型注解

### 3.1 Python 3.9+ 风格（强制）

```python
# ✅ 正确 — Python 3.9+ 内置泛型
def process(data: list[float], mapping: dict[str, int]) -> tuple[float, float]: ...


# ✅ 正确 — | 联合类型
def get_value(key: str | None) -> float | None: ...


# ❌ 错误 — 旧式 typing 泛型
from typing import List, Dict, Tuple, Optional


def process(data: List[float], mapping: Dict[str, int]) -> Tuple[float, float]: ...


# ❌ 错误 — Optional
def get_value(key: Optional[str]) -> Optional[float]: ...
```

> **唯一例外**：`Optional[X]` 仅在 `from __future__ import annotations` 无法使用时保留。

### 3.2 前向引用处理

```python
# 方案1：TYPE_CHECKING 守卫（推荐）
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fl_space.environment.celestial_body import CelestialBody
    from fl_space.environment.ground_station import GroundStationNetwork


# 方案2：字符串注解（备选）
def func(body: "CelestialBody") -> "GroundStationNetwork": ...
```

### 3.3 dataclass 字段类型

```python
@dataclass
class SatelliteSpec:
    name: str
    altitude_km: float
    inclination_deg: float
    ground_stations: list[str] = field(default_factory=list)  # 可变默认值
    optional_param: float | None = None  # 可选字段
```

---

## 4. 文档字符串

### 4.1 格式：NumPy/SciPy 风格（强制）

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    函数功能简述。

    详细描述可以跨多行，说明算法原理、
    注意事项等。

    Parameters
    ----------
    param1 : type1
        参数1的描述。
    param2 : type2
        参数2的描述，可以跨多行。

    Returns
    -------
    return_type
        返回值描述。

    Raises
    ------
    ValueError
        当输入不合法时抛出。

    Notes
    -----
    额外的实现细节或数学背景。

    Examples
    --------
    >>> result = function_name(1.0, 2.0)
    >>> print(result)
    3.0
    """
```

### 4.2 类文档字符串

```python
class OrbitSimulator:
    """
    模块化轨道接触模拟器。

    双后端支持:
        - backend="kepler"  : 轻量开普勒力学（默认，无外部依赖）
        - backend="skyfield": 高精度 SGP4/TLE + JPL 星历

    Attributes
    ----------
    body : CelestialBody
        模拟目标天体。
    contact_matrix : ContactMatrix
        接触矩阵存储。

    Parameters
    ----------
    backend : str
        "kepler" (默认) | "skyfield"。
    num_satellites : int
        卫星数量，默认 3。
    ...
    """
```

### 4.3 模块级文档字符串

每个 `.py` 文件必须在文件头部包含模块文档字符串：

```python
"""
模块名称 — 一句话描述。

支持:
    - 功能点1
    - 功能点2

使用示例::

    from fl_space.xxx import YYY
    result = YYY()
"""
```

### 4.4 注释语言

- **全部使用中文**注释和文档字符串
- 代码标识符使用英文

---

## 5. 导入规范

### 5.1 排序规则（isort）

```
[标准库]
import math
import time

[第三方库]
import numpy as np
import matplotlib.pyplot as plt

[第一方库 — fl_space]
from fl_space.environment import CelestialBody, GroundStationNetwork

[相对导入]
from .contact_matrix import ContactMatrix
```

### 5.2 __init__.py 重导出规范

每个包的 `__init__.py` **必须**同时满足：

```python
# 1. 显式 as 重导出（Ruff I 规则检查）
from .kepler_orbit import KeplerOrbit as KeplerOrbit

# 2. 模块级 docstring
"""
XXX层 — 简短描述
...
"""

# 3. __all__ 列表
__all__ = ["KeplerOrbit", "create_circular_orbit", ...]
```

### 5.3 可选依赖/懒导入

```python
# 方案1：在 __init__.py 中使用 try/except 守卫
try:
    from .skyfield_backend import SkyfieldOrbitBackend as SkyfieldOrbitBackend
except ImportError:
    SkyfieldOrbitBackend = None  # type: ignore


# 方案2：在方法内部懒导入（减少启动开销）
@property
def rotation_rate_rad_per_min(self) -> float:
    import math

    return math.radians(self.rotation_rate_deg_per_min)
```

---

## 6. 数据模型模式

### 6.1 dataclass 为核心数据载体

所有配置和参数对象使用 `@dataclass`：

```python
@dataclass
class GroundStation:
    """地面站定义。"""
    name: str
    lat_deg: float
    lon_deg: float
    altitude_km: float = 0.0
    min_elevation_deg: float = 10.0

    def __post_init__(self):
        """初始化后验证。"""
        if not -90 <= self.lat_deg <= 90:
            raise ValueError(f"纬度 {self.lat_deg} 超出范围 [-90, 90]")

    @property
    def lat_rad(self) -> float:
        """纬度（弧度）。"""
        return math.radians(self.lat_deg)

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "name": self.name,
            "lat_deg": self.lat_deg,
            ...
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GroundStation":
        """从字典反序列化。"""
        return cls(**data)
```

### 6.2 工厂模式

```python
# @classmethod 工厂 — 用于预设实例
@classmethod
def earth(cls, precise: bool = False) -> "CelestialBody": ...


# 独立工厂函数 — 用于组合创建
def create_default_network(n: int = 7) -> GroundStationNetwork: ...


def create_circular_orbit(
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    true_anomaly_deg: float,
    body: CelestialBody,
) -> KeplerOrbit: ...
```

### 6.3 任务与卫星扩展

- 火箭级、运载火箭、卫星设计、目标轨道和任务结果使用 dataclass；
- `SatelliteDesign.visual` 只描述渲染，不隐式修改质量、阻力或功率；
- 外部 SpaceX API 只能通过 `fl_space.integrations` 适配器进入；
- 新任务和卫星行为优先通过注册表/插件扩展；
- 结果中记录 `toy`、`first_order`、`kepler`、`sgp4` 或 `high_fidelity` 标签；
- 详细规则见 `docs/SPACE_MISSION_FRAMEWORK.md` 和
  `docs/references/space-satellite-physics.md`。

### 6.4 单例模式

```python
class SatelliteRegistry:
    """全局卫星注册表单例。"""

    _instance: "SatelliteRegistry | None" = None

    def __new__(cls) -> "SatelliteRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance


# 模块级单例
registry = SatelliteRegistry()
```

---

## 7. 错误处理

### 7.1 可选依赖静默回退

```python
SKYFIELD_AVAILABLE = False
try:
    from skyfield.api import load

    SKYFIELD_AVAILABLE = True
except ImportError:
    pass
```

### 7.2 优雅的默认值

```python
def __init__(self, body: CelestialBody | None = None):
    if body is None:
        body = CelestialBody.earth()  # 工厂回退
    self.body = body
```

### 7.3 输入验证

```python
def __post_init__(self):
    if self.altitude_km <= 0:
        raise ValueError(f"轨道高度必须为正，当前值: {self.altitude_km}")
```

### 7.4 调试输出控制

```python
def simulate(self, verbose: bool = True):
    if verbose:
        print(f"模拟进度: {progress:.1f}%")
```

---

## 8. 代码质量工具

### 8.1 Ruff（唯一工具）

本项目**统一使用 Ruff** 进行 linting 和 formatting。配置位于 `pyproject.toml` 的 `[tool.ruff]` 下。

**常用命令：**

```bash
# 代码检查（必须通过）
ruff check fl_space/

# 自动修复
ruff check --fix fl_space/

# 代码格式化（必须通过）
ruff format fl_space/

# 检查特定文件
ruff check fl_space/orbit/kepler_orbit.py
```

### 8.2 启用的规则集

| 规则 | 说明 | 严重性 |
|------|------|--------|
| `F` | Pyflakes — 未定义名称、未使用导入 | 必须修复 |
| `E`/`W` | pycodestyle — 风格错误/警告 | 必须修复 |
| `I` | isort — 导入排序 | 必须修复 |
| `N` | pep8-naming — 命名规范 | 必须修复 |
| `B` | flake8-bugbear — 常见bug | 必须修复 |
| `SIM` | 代码简化建议 | 建议修复 |
| `UP` | pyupgrade — 现代语法 | 建议修复 |
| `C4` | 推导式优化 | 建议修复 |
| `PERF` | 性能优化 | 建议修复 |
| `RUF` | Ruff特有规则 | 建议修复 |

### 8.3 忽略的规则及原因

| 规则 | 原因 |
|------|------|
| `E501` | 行宽由 formatter 处理，100字符限制 |
| `B008` | dataclass 中 `field(default_factory=list)` 需要函数调用 |
| `B028` | 测试中需要无消息 assert |
| `SIM108` | 不强制三元表达式 |
| `PERF203` | 部分场景刻意在循环内 try-except |
| `RUF002` | 中文文档的正常全角标点 |
| `RUF003` | 中文注释的正常使用 |

### 8.4 格式化配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `target-version` | `py39` | 目标 Python 3.9 |
| `line-length` | `100` | 科研代码适当放宽 |
| `quote-style` | `"double"` | 双引号 |
| `indent-style` | `"space"` | 空格缩进 |
| `skip-magic-trailing-comma` | `false` | 保留尾部逗号魔法 |

---

## 9. 测试规范

### 9.1 测试文件位置

```
tests/
├── test_quick.py    # 快速冒烟测试（每次提交前运行）
└── verify_all.py    # 全量验证测试（发版前运行）
```

### 9.2 测试风格

当前使用纯 `assert` + `print` 风格（无 pytest 框架），未来可迁移到 pytest：

```python
def check(name: str, condition: bool, detail: str = "") -> int:
    """检查条件，失败时收集错误信息。"""
    if condition:
        print(f"  ✅ {name}")
        return 0
    else:
        error_msg = f"  ❌ {name}"
        if detail:
            error_msg += f" — {detail}"
        print(error_msg)
        return 1


# 使用
errors = []
errors.append(check("天体创建", body is not None, "body is None"))
errors.append(check("半径正确", body.radius_km == 6371))

# 汇总
if errors:
    print(f"\n❌ {sum(errors)} 项测试失败")
else:
    print("\n✅ 全部测试通过")
```

### 9.3 测试原则

- 冒烟测试覆盖所有公共 API 的**基本调用**
- 验证测试覆盖**边界条件**和**典型场景**
- 测试 `kepler` 和 `skyfield` 两个后端
- 使用 `verbose=False` 加速测试
- 生成的临时文件在测试结束后清理

### 9.4 测试执行命令

```bash
# 快速检查
python tests/test_quick.py

# 完整验证（含 skyfield）
python tests/verify_all.py
```

---

## 10. AI 辅助开发准则

### 10.1 核心原则

> **所有 AI 生成的代码必须与人工编写的代码遵循完全相同的规范。**

### 10.2 AI 提示词模板

在让 AI 修改或生成代码时，使用以下标准提示词前缀：

```
你正在为 SpaceFL (太空联邦学习研究框架) 项目编写代码。
请严格遵循以下规范：
- Python 3.9+, 所有注释和文档字符串使用中文
- 类型注解使用 Python 3.9+ 风格 (list[X], dict[K,V], X | None)
- 文档字符串使用 NumPy/SciPy 风格
- 导入顺序：标准库 -> 第三方 -> fl_space -> 相对导入
- 命名：snake_case 函数/变量, PascalCase 类, UPPER_SNAKE_CASE 常量
- 使用 @dataclass 定义数据类
- 遵循 Ruff 规则（行宽100，双引号，空格缩进）
- 可选依赖使用 try/except 静默回退
- __init__.py 必须包含 __all__ 和显式 as 重导出
```

### 10.3 AI 生成代码检查流程

**每段 AI 生成的代码必须经过以下检查：**

1. **命名检查**：是否符合 `snake_case`/`PascalCase`/`UPPER_SNAKE_CASE`？
2. **类型注解**：是否使用 `list[X]`/`X | None`？（不是 `List[X]`/`Optional[X]`）
3. **文档字符串**：是否有 NumPy 风格 docstring？是否用中文？
4. **导入排序**：是否正确分层排序？
5. **Ruff 检查**：`ruff check` 是否通过？
6. **Ruff 格式化**：`ruff format` 是否通过？
7. **循环依赖**：是否引入了跨层循环依赖？
8. **`__all__` 更新**：新增公开 API 是否加入了 `__init__.py` 的 `__all__`？

### 10.4 AI 严禁事项

- ❌ 不要引入新的代码格式化工具（black, isort 等），统一使用 Ruff
- ❌ 不要在非 `__init__.py` 文件中使用 `from module import *`
- ❌ 不要使用 `typing.List`, `typing.Dict`, `typing.Optional` 等旧式泛型
- ❌ 不要跨层直接引用（如 simulator 不应 import environment 的内部实现）
- ❌ 不要引入 `cartopy`, `astropy` 等重型依赖（除非经过讨论同意）
- ❌ 不要修改 `pyproject.toml` 中的 Ruff 配置（除非经过讨论同意）

---

## 11. 提交前检查清单

在提交任何代码（含 AI 生成代码）之前，确认以下全部通过：

```bash
# [ ] 1. Ruff 检查零错误
ruff check fl_space/

# [ ] 2. Ruff 格式化检查零变更
ruff format --check fl_space/

# [ ] 3. 快速冒烟测试通过
python tests/test_quick.py

# [ ] 4. 新增公开API已加入 __all__
grep "__all__" fl_space/*/__init__.py

# [ ] 5. 无循环依赖
python -c "import fl_space; print('OK')"

# [ ] 6. 文档字符串完整（每个公共函数/类）
ruff check --select D fl_space/  # (如果启用 pydocstyle)
```

### 额外的人工检查项

- [ ] 新模块是否有模块级 docstring？
- [ ] dataclass 是否有 `to_dict()`/`from_dict()`？
- [ ] 可选依赖是否正确使用 try/except 守卫？
- [ ] 新增文件是否遵循了命名规范？
- [ ] 测试是否覆盖了新功能？

---

## 附录 A：常见模式速查

### A.1 创建新的子模块

```python
# fl_space/new_module/my_feature.py
"""
新功能模块 — 简要描述。

支持:
    - 功能1
    - 功能2
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fl_space.environment import CelestialBody


@dataclass
class NewFeature:
    """新功能描述。"""

    name: str
    param: float

    def run(self) -> list[float]:
        """
        执行核心逻辑。

        Returns
        -------
        list[float]
            计算结果列表。
        """
        ...
```

### A.2 __init__.py 模板

```python
"""
new_module — 模块简要描述。
"""

from .my_feature import NewFeature as NewFeature

__all__ = [
    "NewFeature",
]
```

### A.3 测试文件模板

```python
"""
new_module 模块测试。
"""

import sys

sys.path.insert(0, "..")

from fl_space.new_module import NewFeature


def test_basic():
    """基本功能测试。"""
    obj = NewFeature(name="test", param=1.0)
    result = obj.run()
    assert len(result) > 0, "结果不应为空"
    print("✅ 基本功能测试通过")


if __name__ == "__main__":
    test_basic()
    print("\n✅ 全部测试通过")
```

---

## 附录 B：常见错误示例

### B.1 类型注解错误

```python
# ❌ 错误
from typing import List, Dict, Optional


def bad(sats: List[str], data: Dict[str, float]) -> Optional[float]: ...


# ✅ 正确
def good(sats: list[str], data: dict[str, float]) -> float | None: ...
```

### B.2 导入顺序错误

```python
# ❌ 错误 — 顺序混乱
from fl_space.environment import CelestialBody
import numpy as np
from .utils import helper
import math

# ✅ 正确 — 标准库 → 第三方 → fl_space → 相对导入
import math

import numpy as np

from fl_space.environment import CelestialBody

from .utils import helper
```

### B.3 缺少 __all__

```python
# ❌ __init__.py 缺少 __all__
from .module import MyClass as MyClass

# ✅ 必须有 __all__
from .module import MyClass as MyClass

__all__ = ["MyClass"]
```

### B.4 dataclass 可变默认值

```python
# ❌ 错误 — 可变默认值导致共享引用
@dataclass
class Bad:
    items: list[str] = []


# ✅ 正确 — 使用 field(default_factory=list)
@dataclass
class Good:
    items: list[str] = field(default_factory=list)
```

---

*本规范随项目演进持续更新。如有疑问或改进建议，请提交 Issue 或联系项目维护者。*
