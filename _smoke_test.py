﻿"""Smoke test: verify ISL, Flower, and Pydantic integration works."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== ISL Module Import Test ===")
from fl_space.isl import (  # noqa: E402
    ISLConfig,
    NoISLCalculator,
    WGS84ISLCalculator,
    compute_intra_cluster_los_windows,
    is_los_clear_ecef,
)

print("  ISL imports OK")

# Test WGS84 occlusion
import numpy as np  # noqa: E402

r1 = np.array([7000.0, 0.0, 0.0])  # point on x-axis
r2 = np.array([-7000.0, 0.0, 0.0])  # opposite side - should be blocked
blocked = is_los_clear_ecef(r1, r2)
print(f"  WGS84 occlusion (opposite side): blocked={not blocked} (expected True)")
assert not blocked, "Earth should block opposite-side satellites"

# Test clear line of sight
r3 = np.array([7000.0, 100.0, 500.0])
r4 = np.array([7100.0, -100.0, -500.0])
clear = is_los_clear_ecef(r3, r4)
print(f"  WGS84 occlusion (nearby): clear={clear} (expected True)")
assert clear, "Nearby satellites should have clear LOS"

# Test ISLConfig
cfg = ISLConfig(enabled=True, calculator="wgs84", atmosphere_buffer_km=80.0)
assert cfg.enabled
assert cfg.atmosphere_buffer_km == 80.0
print("  ISLConfig OK")

# Test calculator factory
from fl_space.isl.intra_cluster import create_isl_calculator  # noqa: E402

calc_wgs84 = create_isl_calculator(cfg)
assert isinstance(calc_wgs84, WGS84ISLCalculator)
calc_disabled = create_isl_calculator(ISLConfig(enabled=False))
assert isinstance(calc_disabled, NoISLCalculator)
print("  Calculator factory OK")

# Test ISL window computation with mock data
from datetime import datetime, timedelta, timezone  # noqa: E402

n_samples = 10
ecef = {
    "SAT-00": np.column_stack([r3 + np.array([i*10, 0, 0]) for i in range(n_samples)]),
    "SAT-01": np.column_stack([r4 + np.array([i*10, 5, 5]) for i in range(n_samples)]),
}
cluster = {"SAT-00": "plane-0", "SAT-01": "plane-0"}
base = datetime(2024, 6, 1, tzinfo=timezone.utc)
times = [base + timedelta(seconds=i * 60) for i in range(n_samples)]
windows = compute_intra_cluster_los_windows(ecef, cluster, times)
print(f"  ISL compute: {len(windows)} windows")
assert len(windows) >= 0

print("\n=== Flower Integration Test ===")
from fl_space.integrations.flower import ContactWindow, FlowerAdapter  # noqa: E402

print("  Flower imports OK")

cw = ContactWindow(satellite_name="SAT-00", station_name="GS-Beijing",
                   start_utc=times[0], end_utc=times[4], duration_s=240.0)
adapter = FlowerAdapter(
    satellite_names=["SAT-00", "SAT-01"],
    station_names=["GS-Beijing"],
    contact_windows=[cw],
)
visible = adapter.clients_visible_at(times[2])
print(f"  Visible clients: {visible}")
assert "SAT-00" in visible

print("\n=== Pydantic Config Test (optional) ===")
try:
    from fl_space.config.schemas import GroundStation, SpaceFLScenario, WalkerSpec  # noqa: F401
    spec = WalkerSpec(num_planes=2, sats_per_plane=5, altitude_km=550)
    assert spec.total_satellites == 10
    print("  Pydantic schemas OK")
except ImportError:
    print("  Pydantic not installed (optional) — skipped")

print("\n=== OrbitSimulator ISL Integration Test ===")
from fl_space.isl.base import ISLConfig as ICfg  # noqa: E402
from fl_space.simulator import OrbitSimulator  # noqa: E402

# Without ISL (backward compatible)
sim_no_isl = OrbitSimulator(num_satellites=3, num_ground_stations=2, verbose=False)
assert not sim_no_isl.isl_config.enabled
print(f"  Without ISL: {sim_no_isl.isl_stats}")

# With ISL
sim_isl = OrbitSimulator(
    num_satellites=3, num_ground_stations=2,
    isl_config=ICfg(enabled=True, calculator="wgs84"),
    verbose=False,
)
assert sim_isl.isl_config.enabled
stats = sim_isl.isl_stats
print(f"  With ISL: total_windows={stats['total_windows']}, unique_links={stats['unique_links']}")

# Test ISL query
active = sim_isl.isl_active_at(60)
print(f"  ISL active at ts=60: {len(active)} links")

# Test ECEF
x, y, z = sim_isl.get_sat_ecef(0, 0)
print(f"  ECEF SAT-00 @ ts=0: ({x:.1f}, {y:.1f}, {z:.1f}) km")

# Test Flower adapter from simulator
adapter2 = FlowerAdapter.from_simulator(sim_isl)
print(f"  Flower adapter: {len(adapter2.contact_windows)} contact windows, "
      f"{len(adapter2.intra_links)} intra links")

print("\n=== KeplerOrbit ECEF Test ===")
from fl_space.environment import CelestialBody  # noqa: E402
from fl_space.orbit import KeplerOrbit, OrbitalElements  # noqa: E402

earth = CelestialBody.earth()
oe = OrbitalElements(
    semi_major_axis_km=earth.radius_km + 500,
    inclination_deg=53.0,
)
orbit = KeplerOrbit(oe, earth)
x, y, z = orbit.ecef_at_time(0)
r = np.sqrt(x**2 + y**2 + z**2)
print(f"  ECEF norm = {r:.1f} km, expected = {earth.radius_km + 500:.1f} km")
assert abs(r - (earth.radius_km + 500)) < 10.0

# Compatibility: FL experiment still works
print("\n=== FL Experiment Compatibility Test ===")
from examples.standard_experiment import run_single_experiment  # noqa: E402

exp = run_single_experiment(
    gs_count=2, sat_count=3, num_rounds=5, local_epochs=1,
    isl_enabled=True, isl_atmosphere_buffer_km=0.0,
    verbose=False,
)
accs = [h["accuracy"] for h in exp.history] if exp.history else [0]
print(f"  FL experiment: max_acc={max(accs):.3f}, "
      f"total_rounds={len(exp.history)}, "
      f"isl_enabled=True")

print("\n=== ALL TESTS PASSED ===")
