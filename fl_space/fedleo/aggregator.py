"""
FedLEOAggregator — 分层聚合器（同轨 → 跨轨）

实现 FedLEO 的去中心化聚合：
    Phase 1: 同轨 Ring-Allreduce（同一轨道面内环形聚合）
    Phase 2: 跨轨 Ring-Allreduce（轨道面之间环形聚合）

MVP 实现：用"分组加权平均"模拟 Ring-Allreduce。
加权平均等价于 Ring-Allreduce 的最终数学结果（Allreduce 语义）。

与 FedAvg 的 SyncWeightedAggregator 对比:
    - SyncWeightedAggregator: 所有选中客户端 → 一次加权平均
    - FedLEOAggregator: 先按轨道面分组聚合 → 再跨轨聚合
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FedLEOAggregator:
    """
    FedLEO 分层聚合器。

    两步聚合流程:
        1. intra_orbit_aggregate(): 同轨加权平均
        2. inter_orbit_aggregate(): 跨轨加权平均

    Parameters
    ----------
    plane_map : dict[int, int]
        卫星 ID → 轨道面 ID 的映射。
    use_weighted_average : bool
        True = 加权平均模拟 Allreduce；False = 简单平均。
    """

    def __init__(
        self,
        plane_map: dict[int, int],
        use_weighted_average: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("FedLEOAggregator 需要 PyTorch")
        self.plane_map = plane_map
        self.use_weighted_average = use_weighted_average

    # ── 主入口 ──────────────────────────────────────────────

    def aggregate(
        self,
        local_weights_list: list[tuple[int, list[Any], int]],
    ) -> list[Any]:
        """
        执行完整的两层聚合。

        Parameters
        ----------
        local_weights_list : list[tuple[int, list, int]]
            [(sat_id, local_weights, data_size), ...]
            每个卫星训练后的本地权重和样本数。

        Returns
        -------
        list
            全局聚合后的权重。
        """
        # Phase 1: 同轨聚合
        plane_weights, plane_sizes = self.intra_orbit_aggregate(local_weights_list)

        # Phase 2: 跨轨聚合
        global_weights = self.inter_orbit_aggregate(plane_weights, plane_sizes)

        return global_weights

    # ── Phase 1: 同轨聚合 ───────────────────────────────────

    def intra_orbit_aggregate(
        self,
        local_weights_list: list[tuple[int, list[Any], int]],
    ) -> tuple[dict[int, list[Any]], dict[int, int]]:
        """
        同轨加权平均聚合。

        将同一轨道面的卫星权重聚合为轨道面代表权重。

        Parameters
        ----------
        local_weights_list : list[tuple[int, list, int]]
            [(sat_id, weights, data_size), ...]

        Returns
        -------
        plane_weights : dict[int, list]
            {plane_id: aggregated_weights}
        plane_sizes : dict[int, int]
            {plane_id: total_data_size}
        """
        # 按轨道面分组
        plane_updates: dict[int, list[tuple[list[Any], int]]] = {}
        for sat_id, weights, data_size in local_weights_list:
            plane_id = self.plane_map.get(sat_id, 0)
            if plane_id not in plane_updates:
                plane_updates[plane_id] = []
            plane_updates[plane_id].append((weights, data_size))

        plane_weights: dict[int, list[Any]] = {}
        plane_sizes: dict[int, int] = {}

        for plane_id, updates in plane_updates.items():
            total_size = sum(sz for _, sz in updates)
            if total_size == 0:
                # 取第一个非空权重作为回退
                plane_weights[plane_id] = [
                    w.clone() if isinstance(w, torch.Tensor) else w
                    for w in updates[0][0]
                ]
                plane_sizes[plane_id] = 1
                continue

            # 加权平均
            aggregated = self._weighted_sum(updates, total_size)
            plane_weights[plane_id] = aggregated
            plane_sizes[plane_id] = total_size

        return plane_weights, plane_sizes

    # ── Phase 2: 跨轨聚合 ───────────────────────────────────

    def inter_orbit_aggregate(
        self,
        plane_weights: dict[int, list[Any]],
        plane_sizes: dict[int, int],
    ) -> list[Any]:
        """
        跨轨加权平均聚合。

        将所有轨道面的代表权重聚合为全局权重。

        Parameters
        ----------
        plane_weights : dict[int, list]
            {plane_id: weights}
        plane_sizes : dict[int, int]
            {plane_id: total_data_size}

        Returns
        -------
        list
            全局聚合权重。
        """
        if not plane_weights:
            return []

        total_size = sum(plane_sizes.values())
        if total_size == 0:
            # 简单平均回退
            return self._simple_average(list(plane_weights.values()))

        if self.use_weighted_average:
            # 加权平均: W_global = Σ (n_plane / N) × W_plane
            first_weights = next(iter(plane_weights.values()))
            aggregated = [
                torch.zeros_like(w, dtype=torch.float32)
                if isinstance(w, torch.Tensor)
                else 0.0
                for w in first_weights
            ]
            for plane_id, weights in plane_weights.items():
                ratio = plane_sizes[plane_id] / total_size
                for i, (agg_w, pw) in enumerate(zip(aggregated, weights)):
                    if isinstance(pw, torch.Tensor):
                        agg_w.add_(pw.float() * ratio)
                    else:
                        aggregated[i] += pw * ratio
            return aggregated
        else:
            return self._simple_average(list(plane_weights.values()))

    # ── 辅助 ────────────────────────────────────────────────

    def _weighted_sum(
        self,
        updates: list[tuple[list[Any], int]],
        total_size: int,
    ) -> list[Any]:
        """按数据量加权求和。"""
        first_weights = updates[0][0]
        aggregated = [
            torch.zeros_like(w, dtype=torch.float32)
            if isinstance(w, torch.Tensor)
            else 0.0
            for w in first_weights
        ]
        for weights, data_size in updates:
            ratio = data_size / total_size if total_size > 0 else 1.0 / len(updates)
            for i, (agg_w, w) in enumerate(zip(aggregated, weights)):
                if isinstance(w, torch.Tensor):
                    agg_w.add_(w.float() * ratio)
                else:
                    aggregated[i] += w * ratio
        return aggregated

    def _simple_average(self, weights_list: list[list[Any]]) -> list[Any]:
        """简单平均（回退）。"""
        if not weights_list:
            return []
        first = weights_list[0]
        n = len(weights_list)
        avg = []
        for i in range(len(first)):
            summed = sum(
                w[i].clone() if isinstance(w[i], torch.Tensor) else w[i]
                for w in weights_list
            )
            if isinstance(summed, torch.Tensor):
                avg.append(summed / n)
            else:
                avg.append(summed / n)
        return avg
