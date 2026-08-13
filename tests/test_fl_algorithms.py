from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fl_space.fl.core import ClientState, ClientUpdate
from fl_space.fl.fedavg import EarliestReturnSelector, FixedEpochTrainer, RandomSelector
from fl_space.fl.fedbuff import AsyncTrainer, BufferAggregator
from fl_space.fl.fedprox import ProximalTrainer
from fl_space.fl.server import FLConfig


def _single_client_problem() -> tuple[nn.Module, TensorDataset, list[torch.Tensor]]:
    torch.manual_seed(3)
    model = nn.Linear(2, 2)
    features = torch.randn(64, 2)
    labels = (features.sum(dim=1) > 0).long()
    weights = [parameter.detach().clone() for parameter in model.parameters()]
    return model, TensorDataset(features, labels), weights


def test_fedprox_mu_zero_matches_fedavg() -> None:
    model, dataset, weights = _single_client_problem()
    avg = FixedEpochTrainer(local_epochs=3, learning_rate=0.05).train(
        0,
        model,
        DataLoader(dataset, batch_size=8, shuffle=False),
        weights,
        0,
    )
    prox = ProximalTrainer(local_epochs=3, learning_rate=0.05, mu=0.0).train(
        0,
        model,
        DataLoader(dataset, batch_size=8, shuffle=False),
        weights,
        0,
    )
    assert all(torch.equal(left, right) for left, right in zip(avg.weights, prox.weights))


def test_sync_selector_samples_round_set_without_current_contact() -> None:
    clients = [ClientState(client_id=index, is_connected=False) for index in range(4)]
    selector = RandomSelector(fraction=1.0, min_clients=4, seed=1)
    assert sorted(selector.select(clients, 0)) == [0, 1, 2, 3]


def test_fedbuff_keeps_updates_beyond_first_buffer() -> None:
    aggregator = BufferAggregator(buffer_size=2, server_learning_rate=0.5)
    for client_id, delta in enumerate((1.0, 2.0, 3.0)):
        aggregator.add_update(
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

    result = aggregator.aggregate([torch.tensor([0.0])], [], 0)
    status = aggregator.buffer_status()
    assert result[0].item() == -0.75
    assert status["last_client_ids"] == [0, 1]
    assert status["current_count"] == 1


def test_fedbuff_reports_version_staleness() -> None:
    aggregator = BufferAggregator(
        buffer_size=2,
        staleness_weight=True,
        server_learning_rate=1.0,
    )
    for client_id, base_version in ((0, 0), (1, 2)):
        aggregator.add_update(
            ClientUpdate(
                client_id=client_id,
                weights=[torch.tensor([0.0])],
                data_size=1,
                train_loss=0.0,
                round_num=base_version,
                model_delta=[torch.tensor([1.0])],
                base_version=base_version,
            )
        )
    aggregator.aggregate([torch.tensor([0.0])], [], 3)
    assert aggregator.buffer_status()["last_staleness"] == [3, 1]


def test_earliest_return_selector_uses_completion_order() -> None:
    clients = [ClientState(client_id=index, is_connected=True) for index in range(4)]
    selector = EarliestReturnSelector(fraction=0.5, min_clients=2)
    selected = selector.select(
        clients,
        0,
        completion_times={0: 20, 1: 9, 2: 12, 3: 5},
    )
    assert selected == [3, 1]


def test_local_trainers_accept_contact_epoch_override() -> None:
    model, dataset, weights = _single_client_problem()
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    avg = FixedEpochTrainer(local_epochs=5, learning_rate=0.05).train(
        0, model, loader, weights, 0, local_epochs_override=2,
    )
    prox = ProximalTrainer(local_epochs=5, learning_rate=0.05, mu=0.1).train(
        0, model, loader, weights, 0, local_epochs_override=3,
    )
    async_update = AsyncTrainer(local_epochs=5, learning_rate=0.05, mu=0.1).train(
        0, model, loader, weights, 0, local_epochs_override=4,
    )
    assert avg.metadata["actual_local_epochs"] == 2
    assert prox.metadata["actual_local_epochs"] == 3
    assert async_update.metadata == {"actual_local_epochs": 4, "proximal_mu": 0.1}


def test_fedbuff_drops_updates_beyond_staleness_bound() -> None:
    aggregator = BufferAggregator(buffer_size=1, max_staleness=2)
    aggregator.add_update(
        ClientUpdate(
            client_id=0,
            weights=[torch.tensor([0.0])],
            data_size=1,
            train_loss=0.0,
            round_num=0,
            model_delta=[torch.tensor([1.0])],
            base_version=0,
        )
    )
    assert aggregator.should_aggregate([], 3) is False
    assert aggregator.buffer_status()["dropped_stale"] == 1


def test_paper_approx_config_applies_algorithm_specific_defaults() -> None:
    avg = FLConfig(algorithm="fedavg", protocol_mode="paper_approx")
    prox = FLConfig(algorithm="fedprox", protocol_mode="paper_approx")
    buff = FLConfig(algorithm="fedbuff", protocol_mode="paper_approx", mu=0.02)
    assert avg.selection_strategy == "earliest_return"
    assert avg.contact_adaptive_epochs is False
    assert prox.selection_strategy == "earliest_return"
    assert prox.contact_adaptive_epochs is True
    assert buff.fedbuff_mu == 0.02
    assert buff.max_staleness == 4
