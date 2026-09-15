"""Deterministic validation suite for FedAvg, FedProx, and FedBuff."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fl_space.fl.core import ClientUpdate
from fl_space.fl.fedavg import FixedEpochTrainer, create_fedavg_components
from fl_space.fl.fedbuff import BufferAggregator, create_fedbuff_components
from fl_space.fl.fedprox import ProximalTrainer, create_fedprox_components
from fl_space.fl.server import FLConfig, FLServer


def _make_dataset(seed: int, num_clients: int = 6) -> tuple[dict[int, DataLoader], DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    train_loaders: dict[int, DataLoader] = {}

    for client_id in range(num_clients):
        preferred_label = client_id % 2
        labels = torch.tensor(
            [preferred_label] * 64 + [1 - preferred_label] * 16,
            dtype=torch.long,
        )
        centers = torch.where(labels[:, None] == 0, -1.0, 1.0).repeat(1, 2)
        features = centers + 0.9 * torch.randn(80, 2, generator=generator)
        loader_generator = torch.Generator().manual_seed(seed + client_id)
        train_loaders[client_id] = DataLoader(
            TensorDataset(features, labels),
            batch_size=16,
            shuffle=True,
            generator=loader_generator,
        )

    test_labels = torch.arange(400, dtype=torch.long) % 2
    test_centers = torch.where(test_labels[:, None] == 0, -1.0, 1.0).repeat(1, 2)
    test_features = test_centers + 0.9 * torch.randn(400, 2, generator=generator)
    test_loader = DataLoader(
        TensorDataset(test_features, test_labels),
        batch_size=128,
        shuffle=False,
    )
    return train_loaders, test_loader


def _make_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))


def _contract_checks() -> dict[str, dict[str, Any]]:
    torch.manual_seed(101)
    model = nn.Sequential(nn.Linear(2, 2))
    global_weights = [parameter.detach().clone() for parameter in model.parameters()]
    features = torch.randn(96, 2)
    labels = (features[:, 0] + features[:, 1] > 0).long()
    dataset = TensorDataset(features, labels)

    avg_update = FixedEpochTrainer(
        local_epochs=3,
        learning_rate=0.05,
    ).train(
        0,
        model,
        DataLoader(dataset, batch_size=16, shuffle=False),
        global_weights,
        0,
    )
    prox_zero_update = ProximalTrainer(
        local_epochs=3,
        learning_rate=0.05,
        mu=0.0,
    ).train(
        0,
        model,
        DataLoader(dataset, batch_size=16, shuffle=False),
        global_weights,
        0,
    )
    prox_update = ProximalTrainer(
        local_epochs=3,
        learning_rate=0.05,
        mu=1.0,
    ).train(
        0,
        model,
        DataLoader(dataset, batch_size=16, shuffle=False),
        global_weights,
        0,
    )

    zero_diff = max(
        float((left - right).abs().max())
        for left, right in zip(avg_update.weights, prox_zero_update.weights)
    )

    def drift(update: ClientUpdate) -> float:
        return float(
            torch.sqrt(
                sum(
                    torch.sum((local - global_weight) ** 2)
                    for local, global_weight in zip(update.weights, global_weights)
                )
            )
        )

    avg_drift = drift(avg_update)
    prox_drift = drift(prox_update)

    buffer = BufferAggregator(buffer_size=2, server_learning_rate=0.5)
    for client_id, delta in enumerate((1.0, 2.0, 3.0)):
        buffer.add_update(
            ClientUpdate(
                client_id=client_id,
                weights=[torch.tensor([0.0])],
                data_size=1,
                train_loss=0.0,
                round_num=0,
                model_delta=[torch.tensor([delta])],
                base_version=0,
            )
        )
    updated_weight = buffer.aggregate([torch.tensor([0.0])], [], 0)[0].item()
    buffer_status = buffer.buffer_status()

    return {
        "fedprox_mu_zero_equals_fedavg": {
            "passed": zero_diff <= 1e-7,
            "max_parameter_difference": zero_diff,
        },
        "fedprox_reduces_client_drift": {
            "passed": prox_drift < avg_drift,
            "fedavg_drift": avg_drift,
            "fedprox_mu_1_drift": prox_drift,
        },
        "fedbuff_fifo_conservation": {
            "passed": (
                buffer_status["last_client_ids"] == [0, 1]
                and buffer_status["current_count"] == 1
            ),
            "first_batch_client_ids": buffer_status["last_client_ids"],
            "remaining_updates": buffer_status["current_count"],
        },
        "fedbuff_delta_and_server_lr": {
            "passed": abs(updated_weight - (-0.75)) <= 1e-7,
            "actual_weight": updated_weight,
            "expected_weight": -0.75,
        },
    }


def _run_one(algorithm: str, seed: int, rounds: int) -> tuple[dict[str, Any], list[dict], list[dict]]:
    train_loaders, test_loader = _make_dataset(seed)
    model = _make_model(seed)
    common = {
        "fraction": 1.0,
        "min_clients": 6,
        "local_epochs": 3,
        "batch_size": 16,
        "learning_rate": 0.05,
        "device": "cpu",
    }

    if algorithm == "fedavg":
        components = create_fedavg_components(**common, seed=seed)
    elif algorithm == "fedprox":
        components = create_fedprox_components(**common, mu=0.5, seed=seed)
    elif algorithm == "fedbuff":
        fedbuff_common = dict(common)
        fedbuff_common.pop("fraction")
        components = create_fedbuff_components(
            **fedbuff_common,
            buffer_size=2,
            staleness_weight=True,
            server_learning_rate=0.5,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    config = FLConfig(
        algorithm=algorithm,
        num_rounds=rounds,
        num_clients=6,
        timeslots_per_round=2,
        fraction=1.0,
        local_epochs=3,
        batch_size=16,
        learning_rate=0.05,
        mu=0.5,
        buffer_size=2,
        staleness_weight=True,
        server_learning_rate=0.5,
        async_eval_every=1,
        seed=seed,
        time_model="slot",
        time_model_kwargs={"slots_per_epoch": 1},
    )
    server = FLServer(config, *components)
    server.run(model, train_loaders, test_loader, verbose=False)
    history = server.get_history_dict()
    events = server.get_event_history()
    trained_history = [row for row in history if row["round"] >= 0]
    final_row = trained_history[-1]
    aggregate_events = [event for event in events if event["event"] == "server_aggregate"]
    staleness = [
        value
        for event in aggregate_events
        for value in event.get("staleness", [])
    ]
    client_updates = (
        sum(row["num_clients"] for row in trained_history)
        if algorithm != "fedbuff"
        else next(
            event["arrivals"]
            for event in events
            if event["event"] == "run_complete"
        )
    )
    summary = {
        "algorithm": algorithm,
        "seed": seed,
        "final_accuracy": final_row.get("accuracy", 0.0),
        "max_accuracy": max(row.get("accuracy", 0.0) for row in trained_history),
        "server_updates": len(aggregate_events),
        "client_updates": client_updates,
        "final_timeslot": final_row["timeslot"],
        "mean_staleness": sum(staleness) / len(staleness) if staleness else 0.0,
    }
    return summary, history, events


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_histories(
    path: Path,
    histories: dict[str, list[dict]],
    x_axis: str,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    for label, history in histories.items():
        trained = [row for row in history if row["round"] >= 0 and "accuracy" in row]
        x_values = (
            list(range(1, len(trained) + 1))
            if x_axis == "server_update"
            else [row["timeslot"] for row in trained]
        )
        axis.plot(
            x_values,
            [row["accuracy"] for row in trained],
            linewidth=1.2,
            alpha=0.8,
            label=label,
        )
    axis.set_xlabel("Server update" if x_axis == "server_update" else "Simulated timeslot")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_algorithm_validation(
    output_dir: str = "algorithm_validation_output",
    rounds: int = 12,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Run contract checks and controlled learning experiments."""
    if seeds is None:
        seeds = [7, 17, 27]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    contracts = _contract_checks()
    _write_json(output / "contracts.json", contracts)

    summaries: list[dict[str, Any]] = []
    plot_histories: dict[str, list[dict]] = {}
    for seed in seeds:
        for algorithm in ("fedavg", "fedprox", "fedbuff"):
            summary, history, events = _run_one(algorithm, seed, rounds)
            summaries.append(summary)
            run_name = f"{algorithm}_seed{seed}"
            _write_json(output / f"{run_name}_history.json", history)
            _write_json(output / f"{run_name}_events.json", events)
            plot_histories[run_name] = history

    _write_summary_csv(output / "summary.csv", summaries)
    aggregate: dict[str, dict[str, float]] = {}
    for algorithm in ("fedavg", "fedprox", "fedbuff"):
        rows = [row for row in summaries if row["algorithm"] == algorithm]
        accuracies = [float(row["final_accuracy"]) for row in rows]
        aggregate[algorithm] = {
            "final_accuracy_mean": statistics.mean(accuracies),
            "final_accuracy_std": statistics.pstdev(accuracies),
            "client_updates_mean": statistics.mean(
                float(row["client_updates"]) for row in rows
            ),
            "final_timeslot_mean": statistics.mean(
                float(row["final_timeslot"]) for row in rows
            ),
            "mean_staleness": statistics.mean(
                float(row["mean_staleness"]) for row in rows
            ),
        }
    report = {
        "contracts_passed": all(item["passed"] for item in contracts.values()),
        "rounds": rounds,
        "seeds": seeds,
        "aggregate": aggregate,
        "runs": summaries,
    }
    _write_json(output / "summary.json", report)
    _plot_histories(
        output / "accuracy_by_server_update.png",
        plot_histories,
        "server_update",
    )
    _plot_histories(
        output / "accuracy_by_timeslot.png",
        plot_histories,
        "timeslot",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="algorithm_validation_output")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    args = parser.parse_args(argv)
    report = run_algorithm_validation(args.output, args.rounds, args.seeds)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["contracts_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
