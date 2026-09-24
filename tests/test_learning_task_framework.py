"""学习任务顶层框架测试。"""

import pytest

from fl_space.integrations import (
    orbital_elements_from_starlink,
    public_starlink_optical_link_profile,
)
from fl_space.mission import ClassicalOrbitalElements, CommunicationSystem, SatelliteDesign
from fl_space.study import (
    CommunicationLinkSpec,
    DatasetSpec,
    GroundStationSpec,
    InfrastructureSpec,
    LearningTaskScenario,
    ModelSpec,
    SchedulerSpec,
    SpaceNodeSpec,
)


def test_classical_orbital_elements_round_trip_and_target_compatibility() -> None:
    """验证六根数可序列化，并能转换到旧目标轨道模型。"""
    elements = ClassicalOrbitalElements.from_circular_altitude(
        550.0,
        inclination_deg=53.0,
        raan_deg=120.0,
        anomaly_deg=30.0,
        epoch_utc="2026-01-01T00:00:00+00:00",
    )
    restored = ClassicalOrbitalElements.from_dict(elements.to_dict())
    target = restored.to_orbital_target()
    assert restored.perigee_altitude_km == pytest.approx(550.0)
    assert restored.apogee_altitude_km == pytest.approx(550.0)
    assert target.altitude_km == pytest.approx(550.0)
    assert target.raan_deg == pytest.approx(120.0)


def test_communication_system_carries_link_simulation_parameters() -> None:
    """验证通信分系统可表达频段、带宽、距离和终端数量。"""
    comms = CommunicationSystem(
        name="ka_band_downlink",
        band="Ka",
        frequency_hz=26.5e9,
        channel_bandwidth_mhz=250.0,
        downlink_rate_mbps=500.0,
        uplink_rate_mbps=25.0,
        terminal_count=2,
        max_range_km=2_000.0,
        source="assumption",
    )
    restored = CommunicationSystem.from_dict(comms.to_dict())
    assert restored.channel_bandwidth_mhz == pytest.approx(250.0)
    assert restored.terminal_count == 2
    assert restored.max_range_km == pytest.approx(2_000.0)


def test_learning_task_scenario_uses_learning_as_top_level_unit() -> None:
    """验证学习任务组合数据、模型、卫星轨道、基础设施和调度。"""
    scenario = LearningTaskScenario(
        name="earth-observation-fl",
        dataset=DatasetSpec(
            name="eo-cloud",
            data_type="image",
            distribution="non_iid_label_skew",
            sample_count=10_000,
            size_mb=512.0,
            shard_count=3,
        ),
        model=ModelSpec(
            name="tiny-resnet",
            task_kind="classification",
            parameter_count=250_000,
            model_size_mb=2.0,
            update_size_mb=2.0,
        ),
        nodes=[
            SpaceNodeSpec(
                node_id="sat-1",
                satellite=SatelliteDesign(name="research-sat-1"),
                orbit=ClassicalOrbitalElements.from_circular_altitude(550.0),
            ),
            SpaceNodeSpec(
                node_id="sat-2",
                satellite=SatelliteDesign(name="research-sat-2"),
                orbit=ClassicalOrbitalElements.from_circular_altitude(
                    560.0,
                    raan_deg=30.0,
                    anomaly_deg=180.0,
                ),
            ),
        ],
        infrastructure=InfrastructureSpec(
            ground_stations=[GroundStationSpec(name="ksc", latitude_deg=28.5, longitude_deg=-80.6)],
            links=[
                CommunicationLinkSpec(
                    name="s_to_ground",
                    band="S",
                    frequency_hz=2.2e9,
                    bandwidth_mhz=5.0,
                    data_rate_mbps=10.0,
                )
            ],
        ),
        scheduler=SchedulerSpec(
            policy="contact_aware",
            round_duration_s=600.0,
            participant_fraction=0.5,
        ),
    )
    restored = LearningTaskScenario.from_dict(scenario.to_dict())
    assert restored.satellite_count == 2
    assert restored.per_round_update_volume_mb == pytest.approx(2.0)
    assert restored.infrastructure.ground_stations[0].name == "ksc"


def test_learning_task_rejects_duplicate_node_ids() -> None:
    """验证重复航天器节点会被拒绝。"""
    node = SpaceNodeSpec(node_id="sat-1")
    with pytest.raises(ValueError):
        LearningTaskScenario(nodes=[node, node])


def test_spacex_starlink_catalog_mapping_is_explicit() -> None:
    """验证 SpaceX API 目录字段到六根数和光链路假设的映射。"""
    elements = orbital_elements_from_starlink(
        {
            "spaceTrack": {
                "SEMIMAJOR_AXIS": 6928.137,
                "ECCENTRICITY": 0.001,
                "INCLINATION": 53.0,
                "RA_OF_ASC_NODE": 10.0,
                "ARG_OF_PERICENTER": 20.0,
                "MEAN_ANOMALY": 30.0,
                "EPOCH": "2026-01-01T00:00:00+00:00",
            }
        }
    )
    assert elements.anomaly_kind == "mean"
    assert elements.perigee_altitude_km == pytest.approx(543.071863)

    optical_link = public_starlink_optical_link_profile()
    assert optical_link.band == "optical"
    assert optical_link.isl_rate_mbps == pytest.approx(25_000.0)
    assert optical_link.max_range_km == pytest.approx(4_000.0)
    assert optical_link.source == "public_report_assumption"
