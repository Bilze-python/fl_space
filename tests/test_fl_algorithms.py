from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fl_space.fl.core import ClientState, ClientUpdate
from fl_space.fl.fedavg import FixedEpochTrainer, RandomSelector
from fl_space.fl.fedbuff import BufferAggregator
from fl_space.fl.fedprox import ProximalTrainer


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
