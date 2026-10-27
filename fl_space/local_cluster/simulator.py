"""Deterministic local satellite-cluster benchmark."""

from __future__ import annotations

import math
from typing import Any

from .models import LocalClusterConfig, LocalClusterResult


def _dbm_to_watts(dbm: float) -> float:
    return 1e-3 * 10.0 ** (dbm / 10.0)


def _link_budget(config: LocalClusterConfig) -> dict[str, float]:
    """Estimate optical power with a Friis-style far-field approximation.

    The paper's 5 km example is near-field sensitive, so this is a bounded
    comparison metric, not a replacement for an optical propagation solver.
    """
    link = config.link
    diameter_m = link.aperture_diameter_cm / 100.0
    wavelength_m = link.wavelength_um * 1e-6
    distance_m = link.link_distance_km * 1e3
    aperture_gain = link.aperture_efficiency * (math.pi * diameter_m / wavelength_m) ** 2
    free_space_factor = (wavelength_m / (4.0 * math.pi * distance_m)) ** 2
    loss_factor = 10.0 ** (-link.other_loss_db / 10.0)
    received_w = link.transmit_power_w * aperture_gain**2 * free_space_factor * loss_factor
    required_w = _dbm_to_watts(link.required_receive_power_dbm_per_channel) * link.channels_per_link
    divergence_urad = wavelength_m / (math.pi * diameter_m / 2.0) * 1e6
    pointing_loss_db = 4.343 * (link.pointing_accuracy_urad / max(divergence_urad, 1e-9)) ** 2
    received_w *= 10.0 ** (-pointing_loss_db / 10.0)
    margin_db = 10.0 * math.log10(max(received_w, 1e-30) / max(required_w, 1e-30))
    budget_factor = min(1.0, max(0.0, 1.0 + margin_db / 20.0))
    return {
        "received_power_mw": received_w * 1e3,
        "link_margin_db": margin_db,
        "pointing_loss_db": pointing_loss_db,
        "budget_factor": budget_factor,
        "divergence_urad": divergence_urad,
        "fresnel_number": diameter_m**2 / (wavelength_m * distance_m),
    }


def _topology_hops(config: LocalClusterConfig) -> float:
    n = config.num_satellites
    if config.topology == "mesh":
        return 1.0
    if config.topology == "grid":
        return max(1.0, math.sqrt(n) / 2.0)
    return max(1.0, n / 4.0)


class LocalClusterSimulator:
    """Run a local cluster without touching the global FL session or state."""

    def __init__(self, config: LocalClusterConfig | None = None):
        self.config = config or LocalClusterConfig.paper_reference()

    def run(self) -> LocalClusterResult:
        config = self.config
        workload = config.workload_profile()
        budget = _link_budget(config)
        nominal_gbps = config.link.nominal_bandwidth_gbps
        effective_gbps = nominal_gbps * budget["budget_factor"]
        training_s = (
            config.samples_per_satellite
            * config.local_epochs
            * float(workload["flops_per_sample"])
            / (config.compute_gflops * 1e9)
        )
        model_bits = float(workload["model_size_mb"]) * 1024.0**2 * 8.0
        aggregation_s = (
            model_bits
            * 2.0
            * max(config.num_satellites - 1, 1)
            / max(effective_gbps * 1e9, 1e-9)
            * _topology_hops(config)
        )
        round_s = training_s + aggregation_s
        total_s = round_s * config.rounds
        samples_per_second = (
            config.num_satellites
            * config.samples_per_satellite
            * config.local_epochs
            * config.rounds
        ) / max(total_s, 1e-9)
        components = self._score_components(
            config,
            effective_gbps,
            nominal_gbps,
            training_s,
            aggregation_s,
            budget["link_margin_db"],
        )
        score = round(sum(components.values()), 3)
        profile = {
            "num_satellites": config.num_satellites,
            "num_planes": 1,
            "sats_per_plane": config.num_satellites,
            "topology": config.topology,
            "cluster_radius_km": config.cluster_radius_km,
            "altitude_km": config.altitude_km,
            "parallel_links": config.link.parallel_links,
            "visual": {
                **config.visual.to_dict(),
                "telescope_diameter_m": (
                    config.link.aperture_diameter_cm * config.visual.telescope_scale_m_per_cm
                ),
                "beam_count": config.link.parallel_links,
            },
            "effective_bandwidth_mbps": effective_gbps * 1000.0,
            "cluster_efficiency_score": score,
        }
        assumptions = [
            "论文参数用于可重复的概念级链路预算，不代表公开工程规格。",
            "采用解析式本地训练和环形/网格/全连接通信时延，不启动全局FL训练。",
            "光链路采用Friis风格估算；5 km案例的近场传播、抖动和捕获过程仍需高保真模型。",
            "cluster_score是实验排序指标，不是卫星作战或真实业务能力值。",
        ]
        return LocalClusterResult(
            name=config.name,
            dataset=config.dataset,
            num_satellites=config.num_satellites,
            effective_bandwidth_gbps=round(effective_gbps, 6),
            nominal_bandwidth_gbps=round(nominal_gbps, 6),
            received_power_mw=round(budget["received_power_mw"], 9),
            link_margin_db=round(budget["link_margin_db"], 4),
            pointing_loss_db=round(budget["pointing_loss_db"], 4),
            training_time_per_round_s=round(training_s, 6),
            aggregation_time_per_round_s=round(aggregation_s, 6),
            per_round_latency_s=round(round_s, 6),
            total_time_s=round(total_s, 6),
            samples_per_second=round(samples_per_second, 6),
            cluster_score=score,
            score_components=components,
            profile=profile,
            assumptions=assumptions,
            config=config.to_dict(),
        )

    @staticmethod
    def _score_components(
        config: LocalClusterConfig,
        effective_gbps: float,
        nominal_gbps: float,
        training_s: float,
        aggregation_s: float,
        margin_db: float,
    ) -> dict[str, float]:
        link_score = 25.0 * min(1.0, effective_gbps / max(nominal_gbps, 1e-9))
        latency_score = 25.0 * min(1.0, 10.0 / max(training_s + aggregation_s, 1e-9))
        scale_score = 25.0 * min(1.0, config.num_satellites / 81.0)
        margin_score = 25.0 * min(1.0, max(0.0, margin_db / 20.0))
        return {
            "link_utilization": round(link_score, 3),
            "round_latency": round(latency_score, 3),
            "cluster_scale": round(scale_score, 3),
            "link_margin": round(margin_score, 3),
        }


def run_local_cluster(config: LocalClusterConfig | None = None) -> LocalClusterResult:
    """Convenience entry point for scripts, Web, and future plugins."""
    return LocalClusterSimulator(config).run()


def profile_to_fedleo_overrides(profile: dict[str, Any]) -> dict[str, Any]:
    """Extract only explicit local-cluster inputs accepted by global FedLEO."""
    allowed = {"num_planes", "sats_per_plane", "bandwidth_mbps", "timeslot_duration_sec"}
    return {key: profile[key] for key in allowed if key in profile}
