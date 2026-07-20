"""
FedLEOPlanner — 离散卸载决策引擎

实现论文 Algorithm 2 (阈值卸载) 和 Algorithm 3 (贪心迭代) 的离散简化版。

核心思想:
    把"卸载量"当作同时影响训练时延和权重散度的控制旋钮:
    - 卸载更多 → 慢节点变快, 数据更均衡, 散度降低
    - 但通信开销上升 → 系统总时延可能增加

离散简化:
    枚举有限个卸载比例 [0, 0.25, 0.5, 0.75, 1.0]
    对每个候选评估: score = w_d · Δdelay + w_div · Δdivergence - w_c · comm_cost
    选择 score 最高的比例。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class OffloadAction:
    """单次卸载动作。"""

    from_sat: int          # 源卫星 ID
    to_sat: int            # 目标卫星 ID
    offload_ratio: float   # 卸载比例 (0.0 ~ 1.0)
    offload_samples: int   # 实际卸载样本数
    comm_cost_slots: int   # 通信开销 (timeslots)
    score: float = 0.0     # 综合评分（越高越好）


@dataclass
class OffloadPlan:
    """一轮完整的卸载计划。"""

    round_num: int
    actions: list[OffloadAction] = field(default_factory=list)
    total_offloaded: int = 0  # 本轮总卸载样本数
    score: float = 0.0

    @property
    def num_actions(self) -> int:
        return len(self.actions)


# ═══════════════════════════════════════════════════════════════
# FedLEOPlanner
# ═══════════════════════════════════════════════════════════════


class FedLEOPlanner:
    """
    FedLEO 卸载规划器 — 离散版 Algorithm 2 + Algorithm 3。

    Parameters
    ----------
    num_satellites : int
        总卫星数。
    num_planes : int
        轨道面数。
    plane_map : dict[int, int]
        卫星 ID → 轨道面 ID 的映射。
    adjacency : dict[int, list[int]]
        邻接表: 卫星 ID → 相邻卫星 ID 列表（同轨 + 跨轨）。
    discrete_ratios : list[float]
        离散卸载比例候选集。
    bandwidth_mbps : float
        ISL 带宽 (Mbps)，用于计算通信开销。
    bytes_per_sample : int
        每样本字节数 (MNIST: 784)。
    timeslot_duration_sec : float
        每 timeslot 秒数，用于将通信时延转为 slots。
    max_offload_iter : int
        Algorithm 3 的最大卸载迭代次数。
    delay_weight : float
        时延降低的权重 w_d。
    divergence_weight : float
        散度改善的权重 w_div。
    comm_cost_weight : float
        通信成本的权重 w_c。
    seed : int | None
        随机种子。
    """

    def __init__(
        self,
        num_satellites: int,
        num_planes: int,
        plane_map: dict[int, int],
        adjacency: dict[int, list[int]],
        discrete_ratios: list[float] | None = None,
        bandwidth_mbps: float = 10.0,
        bytes_per_sample: int = 784,
        timeslot_duration_sec: float = 60.0,
        max_offload_iter: int = 5,
        delay_weight: float = 1.0,
        divergence_weight: float = 0.5,
        comm_cost_weight: float = 0.3,
        seed: int | None = 42,
    ):
        self.num_satellites = num_satellites
        self.num_planes = num_planes
        self.plane_map = plane_map
        self.adjacency = adjacency
        self.discrete_ratios = discrete_ratios or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.bandwidth_mbps = bandwidth_mbps
        self.bytes_per_sample = bytes_per_sample
        self.timeslot_duration_sec = timeslot_duration_sec
        self.max_offload_iter = max_offload_iter
        self.w_delay = delay_weight
        self.w_div = divergence_weight
        self.w_comm = comm_cost_weight
        self._rng = np.random.default_rng(seed)

    # ── 主入口 ──────────────────────────────────────────────

    def plan(
        self,
        current_data_sizes: list[int],
        compute_powers: list[float] | None = None,
        current_divergence: float = 0.0,
        round_num: int = 0,
    ) -> OffloadPlan:
        """
        执行 Algorithm 3：贪心迭代卸载规划。

        每轮先找"最慢卫星"，再选最佳邻居按 Algorithm 2 决定卸载量，
        虚拟更新数据分布后继续迭代。

        Parameters
        ----------
        current_data_sizes : list[int]
            各卫星当前持有的样本数。
        compute_powers : list[float] | None
            各卫星的相对算力 (1.0 = 基准)，None 表示全部相同。
        current_divergence : float
            当前权重散度。
        round_num : int
            当前训练轮次。

        Returns
        -------
        OffloadPlan
            本轮卸载计划。
        """
        if compute_powers is None:
            compute_powers = [1.0] * self.num_satellites

        # 复制数据分布以进行虚拟更新
        virtual_sizes = list(current_data_sizes)
        plan = OffloadPlan(round_num=round_num)
        used_pairs: set[tuple[int, int]] = set()

        for _iteration in range(self.max_offload_iter):
            # Step 1: 找到当前"最慢"卫星
            slowest = self._find_slowest(virtual_sizes, compute_powers)
            if slowest is None:
                break

            # Step 2: 评估所有邻居的卸载收益
            neighbors = self.adjacency.get(slowest, [])
            if not neighbors:
                break

            best_action: OffloadAction | None = None
            best_score = -float("inf")

            for neighbor in neighbors:
                pair = (slowest, neighbor)  # slowest → neighbor
                if pair in used_pairs:
                    continue
                if virtual_sizes[slowest] <= 0:
                    continue

                action = self._evaluate_offload(
                    from_sat=slowest,
                    to_sat=neighbor,
                    virtual_sizes=virtual_sizes,
                    compute_powers=compute_powers,
                    current_divergence=current_divergence,
                )
                if action is not None and action.score > best_score:
                    best_score = action.score
                    best_action = action

            # Step 3: 如果收益为正，应用最佳卸载
            if best_action is not None and best_action.score > 0:
                plan.actions.append(best_action)
                plan.total_offloaded += best_action.offload_samples
                plan.score += best_action.score
                # 虚拟更新数据分布
                virtual_sizes[best_action.from_sat] -= best_action.offload_samples
                virtual_sizes[best_action.to_sat] += best_action.offload_samples
                used_pairs.add((best_action.from_sat, best_action.to_sat))
            else:
                break  # 无正收益，停止迭代

        return plan

    # ── 内部方法 ────────────────────────────────────────────

    def _find_slowest(
        self,
        data_sizes: list[int],
        compute_powers: list[float],
    ) -> int | None:
        """
        找到训练时延最长的卫星。

        训练时延 ∝ data_size / compute_power
        """
        max_delay = -1.0
        slowest = -1
        for sat_id, (size, power) in enumerate(zip(data_sizes, compute_powers)):
            if size <= 0:
                continue
            effective_size = size / max(power, 0.01)
            if effective_size > max_delay:
                max_delay = effective_size
                slowest = sat_id
        return slowest if slowest >= 0 else None

    def _evaluate_offload(
        self,
        from_sat: int,
        to_sat: int,
        virtual_sizes: list[int],
        compute_powers: list[float],
        current_divergence: float,
    ) -> OffloadAction | None:
        """
        对一个 (from_sat → to_sat) 卸载对，评估所有离散比例的收益。

        Algorithm 2 离散版：
            for ratio in [0, 0.25, 0.5, 0.75, 1.0]:
                score = w_d·Δdelay + w_div·Δdiv - w_c·comm_cost
            return best

        其中:
            Δdelay = 卸载前最慢时延 - 卸载后最慢时延（正数 = 改善）
            Δdiv ≈ 数据均衡度提升（正数 = 改善，data_balance_entropy 增量）
            comm_cost = 卸载样本数的通信开销
        """
        source_size = virtual_sizes[from_sat]
        if source_size <= 0:
            return None

        # 卸载前基准
        old_balance = self._data_balance(virtual_sizes)
        old_slowest_delay = max(
            (s / max(p, 0.01)) for s, p in zip(virtual_sizes, compute_powers) if s > 0
        )

        best_action: OffloadAction | None = None
        best_score = -float("inf")

        for ratio in self.discrete_ratios:
            if ratio <= 0:
                continue

            n_samples = int(source_size * ratio)
            if n_samples <= 0:
                continue

            # 模拟卸载后的数据分布
            sim_sizes = list(virtual_sizes)
            sim_sizes[from_sat] -= n_samples
            sim_sizes[to_sat] += n_samples

            # Δdelay
            new_slowest = max(
                (s / max(p, 0.01))
                for s, p in zip(sim_sizes, compute_powers)
                if s > 0
            )
            delta_delay = old_slowest_delay - new_slowest  # 正数 = 改善

            # Δdivergence proxy = 数据均衡度变化
            new_balance = self._data_balance(sim_sizes)
            delta_balance = new_balance - old_balance

            # comm_cost
            comm_cost = self._offload_comm_cost(n_samples)

            # 综合打分
            score = (
                self.w_delay * delta_delay
                + self.w_div * delta_balance
                - self.w_comm * comm_cost
            )

            if score > best_score:
                best_score = score
                best_action = OffloadAction(
                    from_sat=from_sat,
                    to_sat=to_sat,
                    offload_ratio=ratio,
                    offload_samples=n_samples,
                    comm_cost_slots=comm_cost,
                    score=score,
                )

        return best_action

    def _data_balance(self, data_sizes: list[int]) -> float:
        """计算数据均衡度（归一化熵，0~1）。"""
        total = sum(data_sizes)
        if total == 0:
            return 0.0
        n = len(data_sizes)
        if n <= 1:
            return 1.0
        max_ent = math.log2(n)
        entropy = 0.0
        for s in data_sizes:
            if s > 0:
                p = s / total
                entropy -= p * math.log2(p)
        return entropy / max_ent

    def _offload_comm_cost(self, n_samples: int) -> int:
        """
        计算卸载 n_samples 个样本的通信开销 (timeslots)。

        cost = n_samples × bytes_per_sample × 8 / (bandwidth_mbps × 1e6)
        然后转换为 timeslots（向上取整）。
        """
        if n_samples <= 0 or self.bandwidth_mbps <= 0:
            return 0
        total_bits = n_samples * self.bytes_per_sample * 8
        cost_seconds = total_bits / (self.bandwidth_mbps * 1e6)
        return max(1, math.ceil(cost_seconds / self.timeslot_duration_sec))

    # ── 工厂方法 ────────────────────────────────────────────

    @classmethod
    def from_orbit_config(
        cls,
        num_satellites: int,
        num_planes: int,
        sats_per_plane: int,
        **kwargs,
    ) -> FedLEOPlanner:
        """
        从轨道配置快速创建 Planner。

        自动构建 plane_map 和环形邻接表（同轨相邻 + 跨轨对应位相邻）。

        Parameters
        ----------
        num_satellites : int
            总卫星数。
        num_planes : int
            轨道面数。
        sats_per_plane : int
            每轨道面卫星数。
        **kwargs
            传递给 FedLEOPlanner 的其他参数。
        """
        # 构建 plane_map: sat_id → plane_id
        plane_map: dict[int, int] = {}
        for sat_id in range(num_satellites):
            plane_map[sat_id] = sat_id // sats_per_plane

        # 构建环形邻接表
        adjacency: dict[int, list[int]] = {}
        for sat_id in range(num_satellites):
            plane = plane_map[sat_id]
            pos_in_plane = sat_id % sats_per_plane
            adj: list[int] = []

            # 同轨相邻（环形）
            prev_sat = plane * sats_per_plane + (pos_in_plane - 1) % sats_per_plane
            next_sat = plane * sats_per_plane + (pos_in_plane + 1) % sats_per_plane
            adj.extend([prev_sat, next_sat])

            # 跨轨相邻（对应 slot 位置的相邻轨）
            if num_planes > 1:
                prev_plane = (plane - 1) % num_planes
                next_plane = (plane + 1) % num_planes
                adj.append(prev_plane * sats_per_plane + pos_in_plane)
                adj.append(next_plane * sats_per_plane + pos_in_plane)

            # 去重
            adjacency[sat_id] = sorted(set(adj))

        return cls(
            num_satellites=num_satellites,
            num_planes=num_planes,
            plane_map=plane_map,
            adjacency=adjacency,
            **kwargs,
        )
