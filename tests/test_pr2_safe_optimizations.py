from __future__ import annotations

import unittest

from fl_space.environment import GroundStation, GroundStationNetwork
from fl_space.simulator.contact_matrix import ContactMatrix
from fl_space.simulator.orbit_simulator import OrbitSimulator


class ContactMatrixOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = ContactMatrix(3, 5, mode="full")
        self.matrix.set_contacts(0, 1, [2, 3])
        self.matrix.set_contacts(1, 1, [0])
        self.matrix.set_contacts(1, 4, [2])

    def test_vectorized_queries_preserve_results(self) -> None:
        self.assertEqual(self.matrix.get_next_contact(0, 0), (1, 2))
        self.assertIsNone(self.matrix.get_next_contact(0, 1))
        self.assertEqual(self.matrix.get_satellites_in_contact(1), [0, 1])
        self.assertEqual(self.matrix.get_satellites_in_contact(2), [])

    def test_vectorized_statistics_preserve_counts(self) -> None:
        self.assertEqual(
            self.matrix.compute_statistics(),
            {
                "total_contacts": 3,
                "sat_contact_counts": [1, 2, 0],
                "gs_contact_counts": [1, 0, 2],
                "avg_contacts_per_sat": 1.0,
                "contact_rate": 0.2,
            },
        )

    def test_contact_lists_remain_defensive_copies(self) -> None:
        contacts = self.matrix.get_all_contacts(0, 1)
        contacts.append(99)
        self.assertEqual(self.matrix.get_all_contacts(0, 1), [2, 3])


class OrbitSimulatorInputTests(unittest.TestCase):
    def test_plain_ground_station_list_is_normalized(self) -> None:
        stations = [
            GroundStation("A", 0.0, 0.0),
            GroundStation("B", 30.0, 120.0),
        ]
        simulator = OrbitSimulator(
            num_satellites=1,
            ground_station_network=stations,
            num_timeslots=2,
            verbose=False,
        )

        self.assertIsInstance(simulator.ground_network, GroundStationNetwork)
        self.assertEqual(simulator.num_ground_stations, 2)
        self.assertEqual(simulator.ground_network.names, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
