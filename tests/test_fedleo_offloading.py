from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from fl_space.fedleo.planner import OffloadAction, OffloadPlan
from fl_space.fedleo.scheduler import FedLEOConfig, FedLEOScheduler


def test_offload_plan_moves_real_samples_and_conserves_total() -> None:
    dataset = TensorDataset(torch.arange(12).float().unsqueeze(1), torch.arange(12))
    loaders = {
        0: DataLoader(Subset(dataset, list(range(8))), batch_size=2),
        1: DataLoader(Subset(dataset, list(range(8, 12))), batch_size=2),
    }
    scheduler = FedLEOScheduler(
        FedLEOConfig(num_satellites=2, num_planes=1, sats_per_plane=2, batch_size=2),
        plane_map={0: 0, 1: 0},
    )
    shared, memberships = scheduler._extract_offload_memberships(loaders)
    plan = OffloadPlan(
        round_num=1,
        actions=[OffloadAction(0, 1, 0.25, 2, 1)],
        total_offloaded=2,
    )

    rebuilt = scheduler._apply_offload_plan(loaders, shared, memberships, plan, 1)

    assert [len(memberships[index]) for index in range(2)] == [6, 6]
    assert sum(len(indices) for indices in memberships.values()) == 12
    assert set(memberships[0]).isdisjoint(memberships[1])
    assert len(rebuilt[0].dataset) == 6
    assert len(rebuilt[1].dataset) == 6
