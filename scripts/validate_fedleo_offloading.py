"""Run a bounded FedLEO offloading on/off validation experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any

from fl_space.fedleo.experiment import run_fedleo_experiment


def _history_checks(history: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [sum(row.get("data_sizes", [])) for row in history]
    accuracies = [float(row["accuracy"]) for row in history]
    losses = [float(row["train_loss"]) for row in history]
    return {
        "sample_totals": totals,
        "samples_conserved": bool(totals) and len(set(totals)) == 1,
        "all_metrics_finite": all(math.isfinite(value) for value in accuracies + losses),
        "initial_accuracy": accuracies[0],
        "final_accuracy": accuracies[-1],
        "accuracy_improved": accuracies[-1] > accuracies[0],
        "initial_balance": float(history[0]["data_balance_entropy"]),
        "final_balance": float(history[-1]["data_balance_entropy"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="fedleo_local_validation")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    common = {
        "num_planes": 3,
        "sats_per_plane": 4,
        "num_rounds": args.rounds,
        "local_epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.02,
        "dataset": "mnist",
        "data_dir": "./data",
        "device": "cpu",
        "offload_every_n_rounds": 2,
        "max_offload_iter": 4,
        "bandwidth_mbps": 10.0,
        "timeslot_duration_sec": 60.0,
        "discrete_ratios": [0.0, 0.25, 0.5],
        "delay_weight": 1.0,
        "divergence_weight": 0.5,
        "comm_cost_weight": 0.3,
        "eval_every_n_rounds": 1,
        "classes_per_client": 1,
        "max_samples_per_client": 300,
        "sample_imbalance": 0.9,
        "seed": args.seed,
        "verbose": False,
    }

    started = time.time()
    offload = run_fedleo_experiment(
        **common,
        enable_offloading=True,
        output_dir=os.path.join(args.output, "offload_on"),
    )
    no_offload = run_fedleo_experiment(
        **common,
        enable_offloading=False,
        output_dir=os.path.join(args.output, "offload_off"),
    )
    elapsed = time.time() - started

    on_checks = _history_checks(offload.history)
    off_checks = _history_checks(no_offload.history)
    gates = {
        "same_initial_model": on_checks["initial_accuracy"] == off_checks["initial_accuracy"],
        "offload_actions_executed": offload.total_offloaded > 0,
        "offload_samples_conserved": on_checks["samples_conserved"],
        "control_samples_conserved": off_checks["samples_conserved"],
        "offload_metrics_finite": on_checks["all_metrics_finite"],
        "control_metrics_finite": off_checks["all_metrics_finite"],
        "offload_training_progressed": on_checks["accuracy_improved"],
        "control_training_progressed": off_checks["accuracy_improved"],
        "offloading_improved_balance": (
            on_checks["final_balance"] > off_checks["final_balance"]
        ),
        "offload_accuracy_not_worse": (
            offload.final_accuracy >= no_offload.final_accuracy
        ),
        "completed_within_30_minutes": elapsed < 1800,
    }
    summary = {
        "experiment": "FedLEO offloading on/off functional validation",
        "config": common,
        "elapsed_sec": round(elapsed, 3),
        "offload_on": {
            "final_accuracy": offload.final_accuracy,
            "peak_accuracy": offload.peak_accuracy,
            "total_offloaded": offload.total_offloaded,
            "total_delay_slots": offload.total_delay_slots,
            **on_checks,
        },
        "offload_off": {
            "final_accuracy": no_offload.final_accuracy,
            "peak_accuracy": no_offload.peak_accuracy,
            "total_offloaded": no_offload.total_offloaded,
            "total_delay_slots": no_offload.total_delay_slots,
            **off_checks,
        },
        "deltas_on_minus_off": {
            "final_accuracy": round(offload.final_accuracy - no_offload.final_accuracy, 6),
            "final_balance": round(
                on_checks["final_balance"] - off_checks["final_balance"], 6
            ),
            "total_delay_slots": offload.total_delay_slots - no_offload.total_delay_slots,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    summary_path = os.path.join(args.output, "validation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
