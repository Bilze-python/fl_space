from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pytest

from fl_space.environment import CelestialBody, GroundStation, GroundStationNetwork
from fl_space.environment.coordinate_utils import ground_central_angle_for_elevation_rad
from fl_space.fl.scheduler import CommunicationScheduler
from fl_space.orbit import KeplerOrbit, OrbitalElements, VisibilityEngine
from fl_space.simulator.orbit_simulator import OrbitSimulator


def test_elevation_mask_uses_exact_spherical_relation_without_implicit_atmosphere() -> None:
    threshold = ground_central_angle_for_elevation_rad(
        6371.0,
        6921.0,
        math.radians(10.0),
    )

    assert math.degrees(threshold) == pytest.approx(14.9676, abs=0.001)


def test_elevation_mask_accounts_for_station_altitude() -> None:
    sea_level = ground_central_angle_for_elevation_rad(6371.0, 6921.0, 0.0)
    high_station = ground_central_angle_for_elevation_rad(
        6371.0,
        6921.0,
        0.0,
        station_altitude_km=4.0,
    )

    assert high_station < sea_level


def test_eccentric_orbit_radius_reaches_perigee_and_apogee() -> None:
    earth = CelestialBody.earth()
    orbit = KeplerOrbit(
        OrbitalElements(
            semi_major_axis_km=8000.0,
            eccentricity=0.1,
            inclination_deg=45.0,
        ),
        earth,
    )

    perigee = orbit.radius_at_time_km(0.0)
    apogee = orbit.radius_at_time_km(orbit.period_min / 2.0)
    ecef_perigee = orbit.ecef_at_time(0.0)
    ecef_apogee = orbit.ecef_at_time(orbit.period_min / 2.0)

    assert perigee == pytest.approx(7200.0, abs=1e-6)
    assert apogee == pytest.approx(8800.0, abs=1e-6)
    assert math.dist((0.0, 0.0, 0.0), ecef_perigee) == pytest.approx(perigee)
    assert math.dist((0.0, 0.0, 0.0), ecef_apogee) == pytest.approx(apogee)


def test_no_contact_search_has_fixed_absolute_bound() -> None:
    network = GroundStationNetwork([GroundStation("north-pole", 90.0, 0.0, min_elevation_deg=10.0)])
    simulator = OrbitSimulator(
        num_satellites=1,
        orbit_altitude_km=500.0,
        orbit_inclination_deg=0.0,
        ground_station_network=network,
        num_timeslots=1,
        verbose=False,
    )

    result = simulator.get_next_contact(0, 0)
    search_chunk = max(int(1.2 * simulator.orbit_period_min), 100)

    assert result is None
    assert simulator.num_timeslots <= 1 + search_chunk * 100


class _FakeSimulator:
    num_satellites = 2
    num_timeslots = 6

    contacts: ClassVar[dict[int, dict[int, list[int]]]] = {
        0: {1: [0], 2: [0], 3: [0]},
        1: {2: [0], 3: [0], 4: [0]},
    }

    contact_matrix = type(
        "FakeContactMatrix",
        (),
        {"simple_matrix": np.full((num_satellites, num_timeslots), -1, dtype=int)},
    )()

    def get_all_contacts(self, sat_id: int, timeslot: int) -> list[int]:
        return list(self.contacts.get(sat_id, {}).get(timeslot, []))

    def get_satellites_in_contact(self, timeslot: int) -> list[int]:
        return [
            sat_id
            for sat_id in range(self.num_satellites)
            if self.get_all_contacts(sat_id, timeslot)
        ]


def test_transfer_window_requires_continuity_and_respects_station_reservations() -> None:
    scheduler = CommunicationScheduler(_FakeSimulator())

    first = scheduler.find_transfer_window(0, -1, duration_slots=3)
    assert first == (1, 0, 4)

    reservations = {0: [(1, 4)]}
    competing = scheduler.find_transfer_window(
        1,
        -1,
        duration_slots=2,
        reservations=reservations,
    )
    assert competing is None

    interrupted = scheduler.find_transfer_window(1, 0, duration_slots=4)
    assert interrupted is None


def test_visibility_engine_does_not_use_full_atmosphere_height_as_solid_body() -> None:
    earth = CelestialBody.earth()
    network = GroundStationNetwork([GroundStation("origin", 0.0, 0.0)])
    orbit = KeplerOrbit(
        OrbitalElements(
            semi_major_axis_km=earth.radius_km + 500.0,
            inclination_deg=0.0,
        ),
        earth,
    )
    engine = VisibilityEngine(earth, orbit, network)

    expected = ground_central_angle_for_elevation_rad(
        earth.radius_km,
        orbit.radius_at_time_km(0.0),
        0.0,
    )

    assert engine.horizon_angle_rad == pytest.approx(expected)
