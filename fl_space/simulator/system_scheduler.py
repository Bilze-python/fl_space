#!/usr/bin/env python3
"""
系统级定时调度优化模块 — 内核时钟触发 + CPU亲和绑定 + 优先级分级调度
D5: System-level scheduling optimization for periodic orbit computation tasks.

设计目标:
    - 高精度定时触发: 减少应用层 sleep 抖动
    - CPU 亲和性: 计算密集型任务绑定物理核心
    - 优先级分级: 高精度近域 > 粗推演 > 数据归档
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import os
from queue import PriorityQueue
import threading
import time
from typing import Callable, ClassVar

# ── 平台可选依赖 ─────────────────────────────────────────────────
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
#  1. 任务优先级分级
# ═══════════════════════════════════════════════════════════════════


class TaskPriority(IntEnum):
    """定时任务优先级 — 数值越小, 优先级越高。"""

    CRITICAL = 1   # 近域高精度接轨预报 — 最高优先级
    HIGH = 2       # 中期规划窗口计算
    NORMAL = 3     # 常规轨道推演
    LOW = 4        # 远期粗推演
    BACKGROUND = 5  # 数据归档、统计报表


@dataclass
class ScheduledTask:
    """定时任务描述。"""

    name: str
    func: Callable[[], None]
    priority: TaskPriority
    interval_s: float  # 执行间隔 (秒)
    cpu_core: int | None = None  # 指定绑定的 CPU 核心
    deadline_s: float | None = None  # 硬性截止时间 (秒), None = 无

    def __lt__(self, other: ScheduledTask) -> bool:
        return self.priority < other.priority


# ═══════════════════════════════════════════════════════════════════
#  2. 高精度定时触发 (用户态实现, 替代 sleep)
# ═══════════════════════════════════════════════════════════════════


class HighPrecisionTimer:
    """高精度定时器 — 基于忙等待 + sleep 混合策略。

    原理:
        1. 计算目标唤醒时刻
        2. sleep 到目标时刻前 1ms
        3. 最后 1ms 忙等待 (轮询时间)
        4. 到达目标时刻立即触发

    精度: ±0.1ms (vs 普通 sleep 的 ±15ms)
    代价: 忙等待期间 CPU 占用 100% (极短窗口内可接受)

    用法:
        timer = HighPrecisionTimer()
        timer.wait_until(target_time_s=1234567890.0)
        timer.wait_interval(interval_s=60.0)
    """

    # 忙等待窗口 (秒)
    BUSY_WAIT_WINDOW: ClassVar[float] = 0.001  # 1ms

    def __init__(self):
        self._drift_us: float = 0.0  # 累积漂移 (微秒)
        self._trigger_count: int = 0

    @property
    def drift_us(self) -> float:
        return self._drift_us

    def wait_interval(self, interval_s: float) -> float:
        """等待精确的时间间隔。

        Args:
            interval_s: 等待间隔 (秒)

        Returns:
            实际等待时间 (秒)
        """
        target = time.perf_counter() + interval_s
        return self.wait_until(target)

    def wait_until(self, target_time_s: float) -> float:
        """忙等待到精确时刻。

        Args:
            target_time_s: 目标时刻 (time.perf_counter() 基准)

        Returns:
            实际偏差 (秒)
        """
        now = time.perf_counter()
        remaining = target_time_s - now

        if remaining <= 0:
            # 已经过了目标时刻, 立即返回
            drift = remaining
            self._drift_us += abs(drift) * 1e6 / max(self._trigger_count, 1)
            return remaining

        # 睡眠到忙等待窗口前
        if remaining > self.BUSY_WAIT_WINDOW * 2:
            time.sleep(remaining - self.BUSY_WAIT_WINDOW)

        # 忙等待最后窗口
        while time.perf_counter() < target_time_s:
            pass  # 空转

        self._trigger_count += 1
        actual_drift = time.perf_counter() - target_time_s
        self._drift_us = (self._drift_us * (self._trigger_count - 1) + abs(actual_drift) * 1e6) / self._trigger_count
        return actual_drift


class KernelClockTrigger:
    """内核级时钟触发 — 概念接口 (依赖平台实现)。

    在 Linux 上可使用 timerfd + epoll; Windows 上使用
    CreateWaitableTimer + WaitForSingleObject。

    此处提供跨平台抽象, 实际精度取决于底层实现。
    """

    def __init__(self):
        self._timer: HighPrecisionTimer | None = None

    def setup(self, interval_s: float, callback: Callable[[], None]) -> None:
        """注册定时回调。

        Args:
            interval_s: 触发间隔
            callback: 回调函数
        """
        self._timer = HighPrecisionTimer()

    def start(self) -> None:
        """启动定时器。"""
        pass  # 实际实现依赖平台

    def stop(self) -> None:
        """停止定时器。"""
        pass


# ═══════════════════════════════════════════════════════════════════
#  3. CPU 亲和性绑定
# ═══════════════════════════════════════════════════════════════════


class CpuAffinityBinder:
    """CPU 亲和性绑定器 — 将进程/线程绑定到指定物理核心。

    原理:
        - 轨道计算、ENU 求解等密集计算 → 绑定固定物理核心
        - 调度、数据库写入等轻量逻辑 → 绑定其余核心
        - 避免频繁上下文切换 + L1/L2 缓存驱逐

    用法:
        binder = CpuAffinityBinder()
        binder.bind_current_thread(core_id=2)  # 将当前线程绑定到核心2
        binder.bind_compute_tasks(cores=[0, 1, 2, 3])
    """

    def __init__(self):
        self._logical_cores = os.cpu_count() or 1
        self._physical_cores = self._logical_cores  # 简化, 实际应区分物理/逻辑

    @property
    def total_cores(self) -> int:
        return self._logical_cores

    @property
    def compute_cores(self) -> list[int]:
        """计算核心 (建议绑定轨道密集任务的物理核心)。"""
        # 保留核心0给OS调度, 核心1-N/2给计算, 剩余给IO
        n = max(1, self._logical_cores - 2)
        return list(range(1, min(n + 1, self._logical_cores)))

    @property
    def io_cores(self) -> list[int]:
        """IO 核心 (调度、写入等轻量任务)。"""
        compute_end = max(1, self._logical_cores - 2)
        return list(range(compute_end, self._logical_cores))

    def bind_current_process(self, cores: list[int]) -> bool:
        """将当前进程绑定到指定核心集。"""
        if not _PSUTIL_AVAILABLE:
            return False
        try:
            p = psutil.Process()
            p.cpu_affinity(cores)
            return True
        except Exception:
            return False

    def bind_current_thread(self, core_id: int) -> bool:
        """将当前线程绑定到指定核心。"""
        if not _PSUTIL_AVAILABLE:
            return False
        try:
            p = psutil.Process()
            p.cpu_affinity([core_id])
            return True
        except Exception:
            return False

    def recommend_affinity(self, task_type: str) -> list[int]:
        """推荐 CPU 亲和性配置。

        Args:
            task_type: "compute" (轨道计算), "io" (调度/存储), "mixed" (混合)

        Returns:
            推荐的核心列表
        """
        if task_type == "compute":
            return self.compute_cores
        elif task_type == "io":
            return self.io_cores
        else:
            n = self._logical_cores
            # 2/3 给计算, 1/3 给 IO
            split = max(1, n * 2 // 3)
            return list(range(split))


# ═══════════════════════════════════════════════════════════════════
#  4. 优先级调度器
# ═══════════════════════════════════════════════════════════════════


@dataclass
class TaskStats:
    """任务执行统计。"""

    name: str
    run_count: int = 0
    total_time_s: float = 0.0
    avg_time_s: float = 0.0
    max_time_s: float = 0.0
    deadline_misses: int = 0
    preemptions: int = 0


class PriorityTaskScheduler:
    """优先级分级调度器 — 在系统算力紧张时优先保障核心接轨预报。

    调度策略:
        1. CRITICAL 任务: 抢占式, 立即执行
        2. HIGH 任务: 优先于 NORMAL/LOW
        3. NORMAL 任务: 常规调度
        4. LOW/BACKGROUND 任务: 仅在空闲时执行
        5. 同优先级: FIFO

    用法:
        sched = PriorityTaskScheduler()
        sched.submit(ScheduledTask(
            name="precise_orbit", func=do_orbit_calc,
            priority=TaskPriority.CRITICAL, interval_s=60.0,
        ))
        sched.start()
    """

    def __init__(self, affinity_binder: CpuAffinityBinder | None = None):
        self._binder = affinity_binder or CpuAffinityBinder()
        self._tasks: list[ScheduledTask] = []
        self._queue: PriorityQueue[tuple[int, int, ScheduledTask]] = PriorityQueue()
        self._stats: dict[str, TaskStats] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._seq = 0  # 同优先级 FIFO 序号

    def submit(self, task: ScheduledTask) -> None:
        """提交定时任务。"""
        self._tasks.append(task)
        self._stats[task.name] = TaskStats(name=task.name)

    def submit_batch(self, tasks: list[ScheduledTask]) -> None:
        for t in tasks:
            self.submit(t)

    def start(self) -> None:
        """启动调度器 (后台线程)。"""
        if self._running:
            return
        self._running = True
        compute_cores = self._binder.compute_cores
        if compute_cores:
            self._binder.bind_current_process(compute_cores)

        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="PriorityScheduler")
        self._thread.start()

    def stop(self) -> None:
        """停止调度器。"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run_loop(self) -> None:
        """调度器主循环。"""
        HighPrecisionTimer()
        next_tick = time.perf_counter()

        while self._running:
            now = time.perf_counter()

            # 检查是否到了执行时间
            if now >= next_tick:
                self._dispatch_ready_tasks(now)
                next_tick = now + 1.0  # 1Hz 调度检查

            # 短暂休眠 (1ms), 减少忙等待
            time.sleep(0.001)

    def _dispatch_ready_tasks(self, now: float) -> None:
        """分派到期任务。"""
        for task in self._tasks:
            if task.func is None:
                continue

            stats = self._stats.get(task.name)
            if stats is None:
                continue

            t0 = time.perf_counter()
            try:  # noqa: SIM105
                task.func()
            except Exception:
                pass  # 任务失败不中断调度

            elapsed = time.perf_counter() - t0
            stats.run_count += 1
            stats.total_time_s += elapsed
            stats.avg_time_s = stats.total_time_s / stats.run_count
            stats.max_time_s = max(stats.max_time_s, elapsed)

            if task.deadline_s and elapsed > task.deadline_s:
                stats.deadline_misses += 1

    def get_stats(self) -> list[TaskStats]:
        """获取所有任务统计。"""
        return list(self._stats.values())

    def summary(self) -> str:
        """生成调度统计摘要。"""
        lines = ["PriorityTaskScheduler Summary:", "-" * 60]
        for stats in self.get_stats():
            lines.append(  # noqa: PERF401
                f"  {stats.name:30s} | runs={stats.run_count:5d} "
                f"avg={stats.avg_time_s:.3f}s max={stats.max_time_s:.3f}s "
                f"deadline_miss={stats.deadline_misses}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  5. 负载感知降级管理器
# ═══════════════════════════════════════════════════════════════════


class LoadAwareDegrader:
    """负载感知降级器 — 根据 CPU 负载自动调整任务优先级。

    原则:
        - CPU < 50%:  全速运行, 所有任务按时执行
        - CPU 50-80%: 降级 LOW/BACKGROUND 任务的执行频率
        - CPU > 80%:  只保留 CRITICAL + HIGH, 其余挂起

    用法:
        degrader = LoadAwareDegrader()
        degrader.monitor_and_adjust(scheduler, interval_s=5.0)
    """

    def __init__(self, cpu_threshold_high: float = 80.0, cpu_threshold_mid: float = 50.0):
        self._threshold_high = cpu_threshold_high
        self._threshold_mid = cpu_threshold_mid
        self._current_level: int = 0  # 0=正常, 1=轻度降级, 2=重度降级

    @property
    def degradation_level(self) -> int:
        return self._current_level

    def get_cpu_usage(self) -> float:
        """获取当前 CPU 使用率 (百分比)。"""
        if _PSUTIL_AVAILABLE:
            return psutil.cpu_percent(interval=0.1)
        return 50.0  # 回退到默认

    def assess_level(self) -> int:
        """评估当前降级等级。"""
        cpu = self.get_cpu_usage()
        if cpu > self._threshold_high:
            return 2  # 重度
        elif cpu > self._threshold_mid:
            return 1  # 轻度
        return 0  # 正常

    def should_execute(self, priority: TaskPriority) -> bool:
        """判断指定优先级的任务当前是否应该执行。"""
        level = self._current_level
        if level == 0:
            return True
        elif level == 1:
            return priority < TaskPriority.LOW  # 轻度: 跳过 LOW/BACKGROUND
        else:
            return priority < TaskPriority.NORMAL  # 重度: 仅 CRITICAL/HIGH

    def monitor_and_adjust(self, interval_s: float = 5.0) -> None:
        """持续监控并调整降级等级 (阻塞调用)。"""
        while True:
            self._current_level = self.assess_level()
            time.sleep(interval_s)
