"""含接收端干扰的 RF 一阶预算；非波形、非实测编码性能。"""

from __future__ import annotations

import math

from .models import GeometrySample, LinkState, RFConfig

LIGHT_SPEED_MPS = 299792458.0
BOLTZMANN_J_PER_K = 1.380649e-23


def evaluate_link(geometry: GeometrySample, config: RFConfig) -> LinkState:
    """按 Friis 损耗、kTB 和接收机噪声因子计算速率。

    全部对数功率为 dBW；天线增益 dBi。指向损失是外部给定的非负 dB，
    不模拟姿态控制。无同频发射机时 SINR=SNR。固定基线保留遮挡，
    故意不使用 SNR 门限，便于隔离固定速率假设的影响。
    """
    fspl_db = 20 * math.log10(
        4 * math.pi * geometry.distance_m * config.frequency_hz / LIGHT_SPEED_MPS
    )
    received_dbw = (
        10 * math.log10(config.tx_power_w)
        + config.tx_gain_dbi
        + config.rx_gain_dbi
        - fspl_db
        - config.system_loss_db
        - geometry.pointing_loss_db
    )
    noise_dbw = (
        10 * math.log10(BOLTZMANN_J_PER_K * config.noise_temperature_k * config.bandwidth_hz)
        + config.noise_figure_db
    )
    snr_db = received_dbw - noise_dbw
    noise_w = 10 ** (noise_dbw / 10)
    sinr_db = received_dbw - 10 * math.log10(noise_w + geometry.interference_w)
    reason = "available"
    if not geometry.visible:
        rate = 0.0
        reason = "not_visible"
    elif config.mode == "fixed":
        rate = config.fixed_rate_bps
    elif sinr_db < config.min_snr_db:
        rate = 0.0
        reason = "below_snr_threshold" if geometry.interference_w == 0 else "below_sinr_threshold"
    else:
        # softplus 避免极高 SNR 的指数溢出。
        snr_log = sinr_db * math.log(10) / 10
        log_capacity = max(snr_log, 0) + math.log1p(math.exp(-abs(snr_log)))
        rate = min(
            config.max_rate_bps,
            config.efficiency * config.bandwidth_hz * log_capacity / math.log(2),
        )
    return LinkState(
        geometry.start_s,
        geometry.end_s,
        geometry.link_id,
        geometry.distance_m,
        geometry.visible,
        fspl_db,
        received_dbw,
        noise_dbw,
        snr_db,
        rate,
        geometry.distance_m / LIGHT_SPEED_MPS,
        reason,
        geometry.interference_w,
        sinr_db,
    )
