from __future__ import annotations

import numpy as np

from fl_space.fl.runner import FLRunner
from fl_space.fl.server import FLConfig


def _runner(seed: int = 7) -> FLRunner:
    return object.__new__(FLRunner)


def test_satellite_profiles_bias_classes_and_conserve_samples() -> None:
    runner = _runner()
    runner.config = FLConfig(num_clients=2, seed=7)
    targets = np.asarray([0] * 2000 + [1] * 2000)
    profiles = {
        "0": {"preferred_classes": [0], "preference_probability": 0.95},
        "1": {"preferred_classes": [1], "preference_probability": 0.95},
    }
    partitions = runner._partition_satellite_profiles(
        targets=targets,
        n_clients=2,
        profiles=profiles,
        default_probability=0.8,
    )
    assert sum(map(len, partitions)) == len(targets)
    assert sum(targets[index] == 0 for index in partitions[0]) > 1800
    assert sum(targets[index] == 1 for index in partitions[1]) > 1800
    assert sum(targets[index] == 1 for index in partitions[0]) < 200
    assert sum(targets[index] == 0 for index in partitions[1]) < 200


def test_satellite_profile_sample_caps_are_independent() -> None:
    partitions = [list(range(100)), list(range(100, 200)), list(range(200, 300))]
    capped = FLRunner._cap_satellite_profiles(
        partitions,
        {"0": {"max_samples": 20}, "2": {"max_samples": 35}},
    )
    assert [len(indices) for indices in capped] == [20, 100, 35]
