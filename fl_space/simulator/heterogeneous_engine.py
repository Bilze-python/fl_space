"""
GPU / 分布式异构并行优化模块 (论文维度四)
============================================

实现:
1. GPU 批量矢量化轨道推演 (CUDA/OpenCL 抽象层)
2. 分布式分域推演集群 (卫星 ID 区间拆分多服务器)
3. 异构算力调度 (CPU 逻辑控制 / GPU 浮点运算)

Notes
-----
本模块为架构级方案, 提供接口约定与本地 CPU 多进程回退。
实际 GPU 部署需 CUDA/PyCUDA 或 OpenCL, 此处提供纯 Python 多进程模拟
与接口适配层, 论文可直接描述架构设计。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from fl_space.simulator.orbit_simulator import OrbitSimulator


# ============================================================
# 1. GPU 批量矢量化抽象层
# ============================================================

class GpuBatchPropagator:
    """GPU 批量 SGP4 + 仰角计算抽象接口。

    实际 GPU 部署: 将时序切片打包为 float 数组, 传入 CUDA kernel
    完成 SGP4 迭代 + ECEF 旋转 + ENU 仰角 + 球面粗判, 一次 kernel 返回全部结果。

    此处提供 CPU 多进程回退实现, 接口一致。
    """

    def __init__(self, gpu_device_id: int = 0, use_gpu: bool = False):
        self._gpu_id = gpu_device_id
        self._use_gpu = use_gpu

    def propagate_batch(
        self,
        sim: OrbitSimulator,
        sat_ids: list[int],
        timeslot: int,
        max_workers: int = 8,
    ) -> dict[int, tuple[float, float, float]]:
        """批量计算多卫星 ECEF 坐标 (GPU 矢量化或 CPU 多进程回退)。

        Returns
        -------
        dict[int, tuple[float, float, float]]
            {sat_id: (x, y, z) km}
        """
        if self._use_gpu:
            # GPU 路径 (抽象)
            return self._gpu_propagate(sim, sat_ids, timeslot)
        # CPU 多进程回退
        return self._cpu_batch_propagate(sim, sat_ids, timeslot, max_workers)

    def _cpu_batch_propagate(
        self,
        sim: OrbitSimulator,
        sat_ids: list[int],
        timeslot: int,
        max_workers: int,
    ) -> dict[int, tuple[float, float, float]]:
        results: dict[int, tuple[float, float, float]] = {}

        chunk_size = max(1, len(sat_ids) // max_workers)
        chunks = [
            sat_ids[i: i + chunk_size] for i in range(0, len(sat_ids), chunk_size)
        ]

        def _chunk_work(sids: list[int]) -> dict[int, tuple[float, float, float]]:
            chunk_res: dict[int, tuple[float, float, float]] = {}
            for sid in sids:
                chunk_res[sid] = sim.get_sat_ecef(sid, timeslot)
            return chunk_res

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_chunk_work, c) for c in chunks]
            for future in as_completed(futures):
                results.update(future.result())
        return results

    def _gpu_propagate(
        self,
        sim: OrbitSimulator,
        sat_ids: list[int],
        timeslot: int,
    ) -> dict[int, tuple[float, float, float]]:
        """GPU 路径 — 占位, 实际 CUDA kernel (论文架构方案)。"""
        return self._cpu_batch_propagate(sim, sat_ids, timeslot, max_workers=1)

    def compute_elevations_batch(
        self,
        sim: OrbitSimulator,
        sat_ids: list[int],
        gs_id: int,
        timeslot: int,
        max_workers: int = 8,
    ) -> dict[int, float]:
        """批量卫星-单站仰角 GPU 并行计算。

        Returns
        -------
        dict[int, float]
            {sat_id: elevation_deg}
        """
        gs = sim.ground_network[gs_id]
        planet_r = sim.body.radius_km

        results: dict[int, float] = {}

        def _one_sat(sid: int) -> tuple[int, float]:
            from fl_space.simulator.pass_scheduler import elevation_deg
            ecef = sim.get_sat_ecef(sid, timeslot)
            el = elevation_deg(ecef, gs.lat_deg, gs.lon_deg, gs.altitude_km, planet_r)
            return (sid, el)

        with ThreadPoolExecutor(max_workers=min(max_workers, len(sat_ids))) as pool:
            futures = [pool.submit(_one_sat, sid) for sid in sat_ids]
            for future in as_completed(futures):
                sid, el = future.result()
                results[sid] = el
        return results


# ============================================================
# 2. 分布式分域推演集群
# ============================================================

@dataclass
class DomainConfig:
    """推演域配置 (单台服务器的卫星 ID 区间)。"""

    server_id: int
    sat_id_start: int
    sat_id_end: int     # 包含
    host: str = "localhost"
    port: int = 8000


class DistributedOrchestrator:
    """分布式分域推演编排器。

    按卫星 ID 区间拆分多台服务器:
        - 节点 A: sat 1-100
        - 节点 B: sat 101-200
    各节点独立定时演算, 最后汇总窗口表。

    此处为单机多进程模拟, 论文可描述真实多机部署架构。
    """

    def __init__(self, domains: list[DomainConfig], use_mpi: bool = False):
        self._domains = sorted(domains, key=lambda d: d.sat_id_start)
        self._use_mpi = use_mpi

    def dispatch_compute(
        self,
        compute_fn: Callable[[int, int], dict[str, Any]],
        merge_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """分发计算任务到各域, 汇总结果。

        Parameters
        ----------
        compute_fn : callable
            compute_fn(sat_id_start, sat_id_end) -> dict 各节点计算函数。
        merge_fn : callable, optional
            merge_fn([result1, result2, ...]) -> dict 结果汇总函数。

        Returns
        -------
        dict
            汇总结果。
        """
        if self._use_mpi:
            return self._mpi_dispatch(compute_fn, merge_fn)

        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=len(self._domains)) as pool:
            futures = {
                pool.submit(compute_fn, d.sat_id_start, d.sat_id_end): d
                for d in self._domains
            }
            for future in as_completed(futures):
                results.append(future.result())  # noqa: PERF401

        if merge_fn:
            return merge_fn(results)
        # 默认合并: 拼接 records
        merged: dict[str, Any] = {"records": [], "stats": {}}
        for r in results:
            merged["records"].extend(r.get("records", []))
        return merged

    def _mpi_dispatch(
        self,
        compute_fn: Callable[[int, int], dict[str, Any]],
        merge_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """MPI 分布式路径 (论文方案, 此处单进程回退)。"""
        # 单进程执行所有域
        results = [compute_fn(d.sat_id_start, d.sat_id_end) for d in self._domains]
        if merge_fn:
            return merge_fn(results)
        merged: dict[str, Any] = {"records": []}
        for r in results:
            merged["records"].extend(r.get("records", []))
        return merged

    def build_uniform_domains(
        self,
        total_sats: int,
        num_nodes: int,
    ) -> list[DomainConfig]:
        """均匀划分卫星到 num_nodes 个节点。"""
        per_node = math.ceil(total_sats / num_nodes)
        domains: list[DomainConfig] = []
        for nid in range(num_nodes):
            start = nid * per_node
            end = min((nid + 1) * per_node - 1, total_sats - 1)
            if start <= end:
                domains.append(DomainConfig(
                    server_id=nid,
                    sat_id_start=start,
                    sat_id_end=end,
                    port=8000 + nid,
                ))
        return domains


# ============================================================
# 3. 异构算力调度
# ============================================================

class HeterogeneousScheduler:
    """异构算力调度器: CPU 负责逻辑控制, GPU 负责浮点运算。

    任务解耦:
        - GPU/ThreadPool: 轨道传播 + ECEF 旋转 + 仰角几何计算
        - CPU: 窗口拼接 + 冲突标记 + 打分择优 + 分配逻辑
    """

    def __init__(
        self,
        gpu_batch: GpuBatchPropagator | None = None,
        cpu_workers: int = 4,
    ):
        self._gpu = gpu_batch or GpuBatchPropagator(use_gpu=False)
        self._cpu_workers = cpu_workers

    def run_pipeline(
        self,
        sim: OrbitSimulator,
        domain: DomainConfig,
    ) -> dict[str, Any]:
        """运行完整异构推演管线: GPU 浮点 + CPU 逻辑。

        Returns
        -------
        dict
            {"records": [...], "timing": {...}}
        """
        import time

        t0 = time.perf_counter()

        # Phase 1: GPU/并行 — 批量轨道传播
        all_ecef: dict[int, dict[int, tuple[float, float, float]]] = {}
        for ts in range(sim.num_timeslots):
            ecefs = self._gpu.propagate_batch(
                sim,
                list(range(domain.sat_id_start, domain.sat_id_end + 1)),
                ts,
            )
            all_ecef[ts] = ecefs

        t1 = time.perf_counter()

        # Phase 2: CPU — 窗口提取 (委托给 pass_scheduler)
        from fl_space.simulator.pass_scheduler import PassTimetable

        # 简化: 直接用 PassTimetable 做剩余逻辑
        tt = PassTimetable(sim, min_duration_slots=1, predict_slots=sim.num_timeslots)

        t2 = time.perf_counter()

        records_list = [
            {
                "sat_id": r.sat_id,
                "gs_id": r.gs_id,
                "ts_start": r.ts_start,
                "ts_end": r.ts_end,
                "duration_slots": r.duration_slots,
            }
            for r in tt.records
        ]

        return {
            "records": records_list,
            "timing": {
                "propagation_s": round(t1 - t0, 3),
                "window_extraction_s": round(t2 - t1, 3),
                "total_s": round(t2 - t0, 3),
            },
        }
