"""FedLEO paper-conformance metadata exposed to the CLI and result files."""

from __future__ import annotations

from typing import Any

IMPLEMENTATION_PROFILE: dict[str, Any] = {
    "paper": "Zhai et al. (2024), FedLEO",
    "implementation_level": "lightweight_discrete_simulation",
    "implemented": [
        "non-IID satellite datasets",
        "ISL-neighbor offloading workflow",
        "greedy iterative offloading",
        "intra-plane then inter-plane weighted aggregation",
        "accuracy, delay, offloading, balance, and divergence metrics",
    ],
    "approximated": [
        "Algorithm 2 uses discrete ratio search instead of the closed-form threshold solution",
        "data-balance entropy is used as the planner's divergence proxy",
        "Ring-Allreduce is simulated by its mathematically equivalent weighted-average result",
        "training and aggregation delays use timeslot estimates",
        "the Walker-star ISL graph is static during an experiment",
    ],
    "not_implemented": [
        "KKT-derived communication-power optimization",
        "multi-hop streaming contention and channel-gain model",
        "dynamic cross-seam and orbital contact constraints",
        "the paper's full continuous P1-P4 optimization",
    ],
    "external_backend": {
        "repository": "https://github.com/teleportup/FedLEO-Federated-Learning",
        "integrated": False,
        "reason": (
            "The repository implements centralized ground-station FedAvg, not the paper's "
            "offloading-assisted decentralized FedLEO algorithm."
        ),
    },
}


def get_implementation_profile() -> dict[str, Any]:
    """Return a copy suitable for JSON output without exposing mutable module state."""
    import copy

    return copy.deepcopy(IMPLEMENTATION_PROFILE)


def format_implementation_profile() -> str:
    """Format the implementation boundary for terminal users."""
    profile = get_implementation_profile()
    labels = (
        ("implemented", "已实现"),
        ("approximated", "近似实现"),
        ("not_implemented", "未实现"),
    )
    lines = [
        "FedLEO 实现说明",
        f"论文: {profile['paper']}",
        f"实现级别: {profile['implementation_level']}",
    ]
    for key, label in labels:
        lines.append(f"\n{label}:")
        lines.extend(f"  - {item}" for item in profile[key])
    backend = profile["external_backend"]
    lines.extend(
        [
            "\n外部仓库后端:",
            f"  - 地址: {backend['repository']}",
            f"  - 已接入: {'是' if backend['integrated'] else '否'}",
            f"  - 原因: {backend['reason']}",
        ]
    )
    return "\n".join(lines)
