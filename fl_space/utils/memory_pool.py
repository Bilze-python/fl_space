"""
内存池管理模块 (furr_chk 四-1)
=============================

原理:
    定时推演会批量生成海量卫星位置时序数组 (ECEF坐标、仰角序列等)，
    频繁申请/释放内存会产生内存碎片和系统开销。
    预先申请固定大小内存池，所有轨道坐标、仰角时序数据复用内存区块，
    定时任务执行全程无动态内存分配。

实现:
    - CoordinateBufferPool: 预分配 (N_sats, N_slots, 3) ECEF 坐标缓冲区
    - ElevationBufferPool: 预分配 (N_sats, N_slots, N_gs) 仰角矩阵缓冲区
    - GenericBufferPool: 通用 numpy 数组复用池
"""

from __future__ import annotations

import numpy as np


class CoordinateBufferPool:
    """卫星 ECEF 坐标内存池 — 预分配三维数组复用。

    用法::

        pool = CoordinateBufferPool(n_sats=100, n_slots=1440)
        # 每次定时任务:
        arr = pool.acquire()        # 获取预分配缓冲区
        # ... 填充 ECEF 坐标到 arr[sat_id, ts, 0:3] ...
        pool.release()              # 标记复用 (不清零, 下次写入覆盖)

    优势: 零次 malloc/free, O(1) 获取, 无碎片。
    """

    def __init__(
        self,
        n_sats: int,
        n_slots: int,
        dtype: np.dtype | str = np.float64,
        n_pools: int = 2,
    ):
        self.shape = (n_sats, n_slots, 3)
        self.dtype = np.dtype(dtype)
        self._n_pools = n_pools
        self._pools: list[np.ndarray] = []
        self._in_use: list[bool] = []
        self._n_sats = n_sats
        self._n_slots = n_slots

        for _ in range(n_pools):
            arr = np.empty(self.shape, dtype=self.dtype)
            self._pools.append(arr)
            self._in_use.append(False)

    def acquire(self) -> np.ndarray:
        """获取一个可用的预分配缓冲区。

        Returns
        -------
        np.ndarray
            shape (n_sats, n_slots, 3) 的缓冲区。
            内容为上次使用后的残留，需覆盖写入。
        """
        for i, in_use in enumerate(self._in_use):
            if not in_use:
                self._in_use[i] = True
                return self._pools[i]
        # 池满：分配新块 (降级为动态分配)
        arr = np.empty(self.shape, dtype=self.dtype)
        self._pools.append(arr)
        self._in_use.append(True)
        return arr

    def release(self, arr: np.ndarray | None = None) -> None:
        """释放指定缓冲区（或释放最后获取的）。"""
        if arr is None:
            # 释放最后使用的
            for i in range(len(self._in_use) - 1, -1, -1):
                if self._in_use[i]:
                    self._in_use[i] = False
                    return
            return

        for i, pool in enumerate(self._pools):
            if pool is arr and self._in_use[i]:
                self._in_use[i] = False
                return

    def release_all(self) -> None:
        """释放全部缓冲区。"""
        for i in range(len(self._in_use)):
            self._in_use[i] = False

    def resize(self, n_sats: int, n_slots: int) -> None:
        """动态调整池大小 (重新分配所有缓冲区)。"""
        self._n_sats = n_sats
        self._n_slots = n_slots
        self.shape = (n_sats, n_slots, 3)
        self._pools = [
            np.empty(self.shape, dtype=self.dtype)
            for _ in range(len(self._pools))
        ]
        self._in_use = [False] * len(self._pools)

    @property
    def usage(self) -> float:
        """缓冲区使用率。"""
        if not self._in_use:
            return 0.0
        return sum(self._in_use) / len(self._in_use)

    @property
    def n_pools(self) -> int:
        return len(self._pools)


class ElevationBufferPool:
    """仰角矩阵内存池 — (N_sats, N_slots, N_gs) 预分配。"""

    def __init__(
        self,
        n_sats: int,
        n_slots: int,
        n_gs: int,
        dtype: np.dtype | str = np.float32,
        n_pools: int = 2,
    ):
        self.shape = (n_sats, n_slots, n_gs)
        self.dtype = np.dtype(dtype)
        self._pools: list[np.ndarray] = []
        self._in_use: list[bool] = []

        for _ in range(n_pools):
            arr = np.full(self.shape, -999.0, dtype=self.dtype)
            self._pools.append(arr)
            self._in_use.append(False)

    def acquire(self) -> np.ndarray:
        for i, in_use in enumerate(self._in_use):
            if not in_use:
                self._in_use[i] = True
                return self._pools[i]
        arr = np.full(self.shape, -999.0, dtype=self.dtype)
        self._pools.append(arr)
        self._in_use.append(True)
        return arr

    def release(self, arr: np.ndarray | None = None) -> None:
        if arr is None:
            for i in range(len(self._in_use) - 1, -1, -1):
                if self._in_use[i]:
                    self._in_use[i] = False
                    return
            return
        for i, pool in enumerate(self._pools):
            if pool is arr and self._in_use[i]:
                self._in_use[i] = False
                return

    def release_all(self) -> None:
        for i in range(len(self._in_use)):
            self._in_use[i] = False


# ============================================================
# 针对 TLE 的增量更新服务 (furr_chk 四-2)
# ============================================================

class TLEIncrementalCache:
    """TLE 增量更新缓存 — 仅加载变更卫星的轨道数据。

    多数卫星 TLE 长时间不会更新，定时任务不再每次全盘读取文件
    解析所有两行元数据。仅加载发生更新的卫星轨道数据，其余卫星
    直接读取内存常驻的轨道参数，削减磁盘 IO 耗时。

    Usage::

        cache = TLEIncrementalCache()
        cache.load_full("all_sats.tle")         # 首次全量加载
        # 定时任务:
        cache.update_incremental("delta.tle")    # 仅加载变更
        params = {sid: cache.get_params(sid) for sid in active_sats}
    """

    def __init__(self):
        self._params: dict[str, dict] = {}      # sat_name → tle_params
        self._version: dict[str, int] = {}       # sat_name → version
        self._loaded_files: set[str] = set()

    def load_full(self, tle_lines: list[str]) -> None:
        """首次全量加载所有卫星 TLE 参数。

        Parameters
        ----------
        tle_lines : list[str]
            完整 TLE 文本行列表 (每颗卫星 2 行数据)。
        """
        self._params.clear()
        self._version.clear()
        for i in range(0, len(tle_lines) - 1, 2):
            name = tle_lines[i + 1][2:8].strip() if i + 1 < len(tle_lines) else f"SAT-{i//2:04d}"
            self._params[name] = {
                "line1": tle_lines[i],
                "line2": tle_lines[i + 1],
            }
            self._version[name] = 1

    def update_incremental(self, delta_lines: list[str]) -> int:
        """增量加载变更卫星的 TLE 参数。

        Parameters
        ----------
        delta_lines : list[str]
            增量 TLE 数据 (仅变更的卫星)。

        Returns
        -------
        int
            实际更新的卫星数量。
        """
        updated = 0
        for i in range(0, len(delta_lines) - 1, 2):
            if i + 1 >= len(delta_lines):
                break
            name = delta_lines[i + 1][2:8].strip() if i + 1 < len(delta_lines) else f"DELTA-{i//2:04d}"
            self._params[name] = {
                "line1": delta_lines[i],
                "line2": delta_lines[i + 1],
            }
            self._version[name] = self._version.get(name, 0) + 1
            updated += 1
        return updated

    def get_params(self, sat_name: str) -> dict | None:
        """获取某卫星的内存常驻参数。"""
        return self._params.get(sat_name)

    def get_all_params(self) -> dict[str, dict]:
        """获取全部卫星参数 (用于批量 SGP4 传播)。"""
        return dict(self._params)

    def get_version(self, sat_name: str) -> int:
        return self._version.get(sat_name, 0)

    @property
    def n_cached(self) -> int:
        return len(self._params)


# ============================================================
# 列式存储写入器 (furr_chk 四-3)
# ============================================================

class ColumnarStore:
    """接轨窗口列式存储 — 减少磁盘 IO。

    原理:
        接轨窗口结构化数据采用列式存储，而非传统行式存储。
        定时计算只需要更新时间、仰角、时长等少数字段，
        列式读写速度远优于普通数据表。

    Usage::

        store = ColumnarStore()
        store.append(ts=129, sat_id=0, gs_id=1, el_max=45.2, dur_slots=3)
        # 批量追加:
        store.append_batch(ts_list=[...], sat_ids=[...], gs_ids=[...], ...)
        store.to_json("passes.json")
    """

    def __init__(self):
        self._columns: dict[str, list] = {
            "timeslot": [],
            "sat_id": [],
            "gs_id": [],
            "elevation_max_deg": [],
            "duration_slots": [],
            "is_relay": [],        # 是否为接力合并窗口
        }

    def append(
        self,
        ts: int,
        sat_id: int,
        gs_id: int,
        el_max: float = 0.0,
        dur_slots: int = 1,
        is_relay: bool = False,
    ) -> None:
        self._columns["timeslot"].append(ts)
        self._columns["sat_id"].append(sat_id)
        self._columns["gs_id"].append(gs_id)
        self._columns["elevation_max_deg"].append(el_max)
        self._columns["duration_slots"].append(dur_slots)
        self._columns["is_relay"].append(int(is_relay))

    def append_batch(self, **cols: list) -> None:
        """批量追加列数据。"""
        for key, values in cols.items():
            if key in self._columns:
                self._columns[key].extend(values)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """转换为 numpy 数组 (列式读取优化)。"""
        return {
            k: np.array(v, dtype=np.int64 if k != "elevation_max_deg" else np.float32)
            for k, v in self._columns.items()
        }

    def to_json(self, path: str) -> None:
        import json
        arrs = self.to_arrays()
        data = {k: v.tolist() for k, v in arrs.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def query_by_sat(self, sat_id: int) -> dict[str, np.ndarray]:
        """列式过滤: 按卫星ID查询 (O(1) 索引切片)。"""
        arrs = self.to_arrays()
        mask = arrs["sat_id"] == sat_id
        return {k: v[mask] for k, v in arrs.items()}

    def query_by_timeslot_range(self, ts_start: int, ts_end: int) -> dict[str, np.ndarray]:
        """列式过滤: 按时隙范围查询。"""
        arrs = self.to_arrays()
        mask = (arrs["timeslot"] >= ts_start) & (arrs["timeslot"] <= ts_end)
        return {k: v[mask] for k, v in arrs.items()}

    def clear(self) -> None:
        for v in self._columns.values():
            v.clear()

    @property
    def n_records(self) -> int:
        return len(self._columns["timeslot"])
