"""
一维卡尔曼滤波器 — 轨道预报时序偏差在线补偿 (furr_chk 第三章)

原理:
    采集历史多组「预推接轨时刻 - 实际过境时刻」误差序列 (ΔT)，
    搭建一维卡尔曼滤波器对时间偏差做在线预测补偿。
    定时演算输出的原始接轨时间叠加滤波预测的偏差值，
    拉长定时刷新周期，依靠滤波修正漂移误差，不用频繁全量重算轨道。

状态向量 x = [偏差(ts), 偏差变化率(ts/slot)]
观测值 z = 本次测得的 ΔT 误差

支持:
- 一维/二维状态向量自适应
- 过程噪声 Q 自适应（偏差大→大Q，跟踪快）
- 误差趋势一阶/二阶多项式拟合
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanConfig:
    """卡尔曼滤波器配置参数。"""

    # 过程噪声协方差 Q — 越大越信任观测值（跟踪快但抖动大）
    q_bias: float = 1.0        # 偏差状态过程噪声
    q_rate: float = 0.01       # 速率状态过程噪声

    # 观测噪声协方差 R — 越大越不信任观测值（平滑但滞后）
    r_obs: float = 4.0

    # 初始估计误差协方差 P0
    p0_diag: float = 100.0

    # 自适应 Q 开关：偏差 > adaptive_q_threshold 时放大 Q 以加速跟踪
    adaptive_q: bool = True
    adaptive_q_threshold: float = 3.0         # timeslots
    adaptive_q_scale: float = 10.0            # Q 放大倍数

    # 维度: 1=仅偏差 (标量), 2=偏差+速率
    dim: int = 2


class KalmanTimingFilter:
    """一维/二维卡尔曼滤波器 — 轨道接轨时间偏差在线预测。

    用于拉长定时刷新周期:
        原始接轨时刻 t_raw + filter.predict() → 修正后接轨时刻 t_corrected

    Usage::

        kf = KalmanTimingFilter(KalmanConfig(dim=2))
        kf.update(measured_delta=3.0)      # 测得本次偏差 3 slots
        correction = kf.predict()          # 预测下次偏差补偿值
        t_corrected = t_raw + correction
    """

    def __init__(self, config: KalmanConfig | None = None, slot_duration_min: float = 1.0):
        self.cfg = config or KalmanConfig()
        self.slot_min = slot_duration_min

        n = self.cfg.dim
        # 状态向量 x: [bias, rate] 或 [bias]
        self._x = np.zeros((n, 1), dtype=np.float64)

        # 状态转移矩阵 F (恒定速率模型)
        # x_{k+1} = bias_k + rate_k * Δt, rate_{k+1} = rate_k
        self._F = np.eye(n, dtype=np.float64)
        if n == 2:
            self._F[0, 1] = 1.0  # bias += rate * 1 step

        # 观测矩阵 H (观测到偏差)
        self._H = np.zeros((1, n), dtype=np.float64)
        self._H[0, 0] = 1.0

        # 误差协方差矩阵
        self._P = np.eye(n, dtype=np.float64) * self.cfg.p0_diag

        # 过程噪声协方差
        self._Q = np.diag([self.cfg.q_bias] + [self.cfg.q_rate] * (n - 1)).astype(np.float64)

        # 观测噪声协方差
        self._R = np.array([[self.cfg.r_obs]], dtype=np.float64)

        self._initialized = False
        self._history: list[float] = []       # 最近偏差历史
        self._max_history = 20

    # ── 核心 API ────────────────────────────────────────────

    def update(self, measured_delta: float) -> float:
        """输入本次测得的偏差 (timeslots)，返回滤波后的偏差估计。

        Parameters
        ----------
        measured_delta : float
            本次实测误差 = actual_ts - predicted_ts。

        Returns
        -------
        float
            滤波后的偏差估计值。
        """
        z = np.array([[measured_delta]], dtype=np.float64)

        if not self._initialized:
            # 首次观测直接初始化状态
            self._x[0, 0] = measured_delta
            if self.cfg.dim >= 2:
                self._x[1, 0] = 0.0
            self._initialized = True
        else:
            # ── 预测步 ──
            self._x = self._F @ self._x
            self._P = self._F @ self._P @ self._F.T + self._Q

            # ── 自适应 Q 放大 ──
            if self.cfg.adaptive_q:
                current_err = abs(measured_delta - self._x[0, 0])
                if current_err > self.cfg.adaptive_q_threshold:
                    q_scaled = self._Q * self.cfg.adaptive_q_scale
                else:
                    q_scaled = self._Q
                self._P = self._F @ self._P @ self._F.T + q_scaled

            # ── 更新步 ──
            y = z - self._H @ self._x                  # 创新
            S = self._H @ self._P @ self._H.T + self._R  # noqa: N806
            K = self._P @ self._H.T @ np.linalg.inv(S)   # noqa: N806
            self._x = self._x + K @ y
            self._P = (np.eye(self.cfg.dim) - K @ self._H) @ self._P

        # 记录历史
        self._history.append(measured_delta)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return float(self._x[0, 0])

    def predict(self, steps_ahead: int = 1) -> float:
        """预测 steps_ahead 步后的偏差补偿值。

        Parameters
        ----------
        steps_ahead : int
            向前预测步数（默认 1，即下一次测量）。

        Returns
        -------
        float
            预测偏差 (timeslots)。
        """
        if not self._initialized:
            return 0.0

        if self.cfg.dim == 1:
            return float(self._x[0, 0])

        # 恒定速率外推: bias_pred = bias + rate * steps
        F_n = np.eye(self.cfg.dim, dtype=np.float64)  # noqa: N806
        F_n[0, 1] = float(steps_ahead)
        x_pred = F_n @ self._x
        return float(x_pred[0, 0])

    def reset(self) -> None:
        """重置滤波器状态。"""
        self._x = np.zeros((self.cfg.dim, 1), dtype=np.float64)
        self._P = np.eye(self.cfg.dim, dtype=np.float64) * self.cfg.p0_diag
        self._initialized = False
        self._history.clear()

    # ── 属性 ─────────────────────────────────────────────────

    @property
    def bias(self) -> float:
        return float(self._x[0, 0])

    @property
    def rate(self) -> float:
        return float(self._x[1, 0]) if self.cfg.dim >= 2 else 0.0

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    @property
    def history(self) -> list[float]:
        return self._history.copy()


class MultiSatKalmanTracker:
    """多卫星卡尔曼滤波追踪器。

    为每颗卫星维护独立的 KalmanTimingFilter，
    用于 SGP4/Kepler 递推时叠加时序补偿。

    Usage::

        tracker = MultiSatKalmanTracker(num_sats=10)
        tracker.update(0, predicted_ts=129, actual_ts=132)  # 偏差 3 slots
        correction = tracker.predict_correction(0)          # → 下次补偿值
    """

    def __init__(self, num_satellites: int, config: KalmanConfig | None = None):
        self._filters: dict[int, KalmanTimingFilter] = {
            sid: KalmanTimingFilter(config) for sid in range(num_satellites)
        }
        self.num_satellites = num_satellites

    def update(self, sat_id: int, predicted_ts: int, actual_ts: int) -> float:
        """记录一次预推偏差并返回滤波估计。"""
        delta = float(actual_ts - predicted_ts)
        return self._filters[sat_id].update(delta)

    def predict_correction(self, sat_id: int, steps_ahead: int = 1) -> float:
        """获取某卫星的预测时序补偿 (timeslots)。"""
        return self._filters[sat_id].predict(steps_ahead)

    def predict_all_corrections(self, steps_ahead: int = 1) -> dict[int, float]:
        """获取所有卫星的预测时序补偿。"""
        return {
            sid: kf.predict(steps_ahead)
            for sid, kf in self._filters.items()
        }

    def get_filter(self, sat_id: int) -> KalmanTimingFilter:
        return self._filters[sat_id]

    def reset_all(self) -> None:
        for kf in self._filters.values():
            kf.reset()


# ============================================================
# 误差趋势多项式拟合 (furr_chk 第三章2)
# ============================================================

def fit_error_trend(
    errors: list[float],
    degree: int = 2,
) -> tuple[list[float], float]:
    """对误差序列做多项式拟合，返回系数和未来一步预测。

    卫星轨道误差随时间近似单调漂移 (J2 长期项主导)，
    使用 1~2 阶多项式拟合误差变化曲线。
    两次定时任务间隔内依靠误差曲线对窗口时刻做动态修正。

    Parameters
    ----------
    errors : list[float]
        历史误差序列 (timeslots)，按时间从旧到新排列。
    degree : int
        拟合多项式阶数 (1=线性, 2=抛物线)。

    Returns
    -------
    (coeffs, next_prediction)
        coeffs: 多项式系数 [a0, a1, ...] (最高次到最低次)
        next_prediction: 下一步预测值 (timeslots)
    """
    n = len(errors)
    if n < 2:
        return ([0.0], 0.0)

    degree = min(degree, n - 1)
    t = np.arange(n, dtype=np.float64)
    y = np.array(errors, dtype=np.float64)

    coeffs = np.polyfit(t, y, degree)
    next_t = float(n)
    next_pred = float(np.polyval(coeffs, next_t))

    return (coeffs.tolist(), next_pred)


def fit_error_trend_weighted(
    errors: list[float],
    degree: int = 2,
    decay: float = 0.9,
) -> tuple[list[float], float]:
    """加权误差趋势拟合 — 近期样本权重更高。

    Parameters
    ----------
    decay : float
        权重衰减因子：第 i 个样本权重 = decay^(n-1-i)。
    """
    n = len(errors)
    if n < 2:
        return ([0.0], 0.0)

    degree = min(degree, n - 1)
    t = np.arange(n, dtype=np.float64)
    y = np.array(errors, dtype=np.float64)
    w = np.power(decay, np.arange(n - 1, -1, -1, dtype=np.float64))

    coeffs = np.polyfit(t, y, degree, w=w)
    next_t = float(n)
    next_pred = float(np.polyval(coeffs, next_t))

    return (coeffs.tolist(), next_pred)
