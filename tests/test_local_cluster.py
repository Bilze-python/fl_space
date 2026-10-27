from __future__ import annotations

import pytest

from fl_space.local_cluster import (
    LocalClusterConfig,
    OpticalLinkConfig,
    profile_to_fedleo_overrides,
    run_local_cluster,
)


def test_paper_reference_is_disabled_and_serializable() -> None:
    config = LocalClusterConfig.paper_reference()
    restored = LocalClusterConfig.from_dict(config.to_dict())

    assert config.enabled is False
    assert restored.num_satellites == 81
    assert restored.altitude_km == pytest.approx(650.0)
    assert restored.link.aperture_diameter_cm == pytest.approx(10.0)
    assert restored.link.nominal_bandwidth_gbps == pytest.approx(9.6)
    assert restored.visual.to_dict()["telescope_scale_m_per_cm"] == pytest.approx(0.01)


def test_local_cluster_run_is_deterministic_and_has_explicit_profile() -> None:
    config = LocalClusterConfig(
        enabled=True,
        num_satellites=9,
        cluster_radius_km=1.0,
        altitude_km=650.0,
        compute_gflops=20.0,
        samples_per_satellite=100,
        rounds=3,
        link=OpticalLinkConfig(parallel_links=2),
    )

    first = run_local_cluster(config).to_dict()
    second = run_local_cluster(config).to_dict()

    assert first == second
    assert first["schema_version"] == "local_cluster_result_v1"
    assert first["cluster_score"] >= 0.0
    assert first["cluster_score"] <= 100.0
    assert first["federated_profile"]["schema_version"] == "local_cluster_profile_v1"
    assert first["federated_profile"]["bandwidth_mbps"] > 0
    assert first["profile"]["visual"]["telescope_diameter_m"] == pytest.approx(0.1)


def test_parallel_links_scale_nominal_bandwidth() -> None:
    one = run_local_cluster(
        LocalClusterConfig(num_satellites=4, samples_per_satellite=10, rounds=1)
    )
    four = run_local_cluster(
        LocalClusterConfig(
            num_satellites=4,
            samples_per_satellite=10,
            rounds=1,
            link=OpticalLinkConfig(parallel_links=4),
        )
    )

    assert four.nominal_bandwidth_gbps == pytest.approx(one.nominal_bandwidth_gbps * 4)


def test_global_override_is_explicit_and_narrow() -> None:
    overrides = profile_to_fedleo_overrides(
        {
            "num_planes": 2,
            "sats_per_plane": 8,
            "bandwidth_mbps": 123.0,
            "cluster_score": 99.0,
            "unrelated": "ignored",
        }
    )

    assert overrides == {
        "num_planes": 2,
        "sats_per_plane": 8,
        "bandwidth_mbps": 123.0,
    }


def test_custom_dataset_requires_workload_parameters() -> None:
    with pytest.raises(ValueError):
        LocalClusterConfig(dataset="my_dataset")
