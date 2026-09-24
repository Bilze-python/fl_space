# SpaceFL Code Quality And Contact-Model Audit

Audit date: 2026-09-20

## Scope And Evidence

This audit covers the current working tree, including uncommitted mission, study, SpaceX adapter,
and web changes. It distinguishes source quality, executable test evidence, physical fidelity, and
public Starlink architecture. The bundled `SpaceX-API-master` repository is reference data only.

Commands and outcomes:

- `venv\\Scripts\\python.exe -m ruff check .`: **0 findings**.
- `venv\\Scripts\\python.exe -m ruff format --check .`: **222 files already formatted**.
- Full local suite with external plugin autoload disabled: **54 passed**.
- `tests/test_time_model.py` is a legacy executable script with top-level work, so it is excluded
  from pytest collection by `tests/conftest.py`; run it directly when its script checks are needed.
- `tests/conftest.py` now removes stale `fl_space` modules and external checkout paths, then asserts
  that pytest imports `D:\\fl_space-master\\fl_space`. This prevents the separate `D:\\fl_space`
  checkout from silently changing test results.
- The ECI/ECEF regression suite (`tests/test_eci_ecef_fix.py`) passes 3 tests. Its implementation
  uses a short-term GMST rotation fallback and should not be treated as a full ITRF/UT1/teme
  reference for long-duration high-precision work.
- `tests/verify_all.py` remains an executable integration script. The standard quality gate is
  `scripts/quality_gate.py`; it does not load JPL DE421. DE421 remains an explicit optional
  resource for workflows that request planetary ephemeris.

## Findings

### Resolved P1 - Skyfield Contact Coordinates Mix Reference Frames

`OrbitSimulator` creates `_SGP4Satellite` objects, whose propagation returns TEME/ECI coordinates.
`SkyfieldOrbitBackend.elevation_at()` passes those coordinates directly to `_compute_topocentric()`,
which subtracts a ground station expressed in Earth-fixed coordinates. No time-dependent Earth
rotation converts TEME/ECI to ITRS/ECEF. Generated-satellite elevation and contact results from this
path are therefore not physically reliable. The native `EarthSatellite` branch does use Skyfield's
frame handling, but the simulator does not use that branch for generated constellations.

Resolution: `fl_space/orbit/skyfield_backend.py` now applies a GMST-based ECI/TEME-to-ECEF
rotation before topocentric calculations, with regression coverage in
`tests/test_eci_ecef_fix.py`. This is a simplified fallback, not a replacement for a full
Skyfield ITRS transform in long-duration/high-precision studies.

### Resolved P1 - No-Contact Search Safety Limit Can Move Forever

`OrbitSimulator.get_next_contact()` extends `self.num_timeslots` inside its loop, then compares
`search_end` with `self.num_timeslots + search_chunk * 100`. Because the right-hand side advances
with every extension, the stated 100-orbit safety limit may never be reached. A satellite that can
never see any configured station can hold a scheduler or experiment in unbounded computation.

Resolution: `get_next_contact()` now captures a fixed absolute search end before extending the
matrix and returns `None` after at most 100 search chunks. Regression coverage is in
`tests/test_contact_and_orbit_fixes.py`.

### Resolved P1 - A Visible Instant Is Treated As Enough For A Transfer

The synchronous FL server finds the first visible slot, adds separately calculated download or
upload slots, and accepts the resulting completion time. It does not verify that visibility remains
continuous for the whole transfer. It also does not reserve ground-station capacity or model two
satellites competing for one beam. Large model updates can therefore finish after the geometric
window has already closed.

Resolution: `CommunicationScheduler.find_transfer_window()` requires one common ground station
to remain visible for every occupied slot. The synchronous server uses it for download and upload,
and keeps a half-open station reservation ledger for the current round. A conflict rejects that
candidate or forces the round to wait; this is a one-terminal-per-station model, not a full beam
or gateway scheduler. Regression coverage is in `tests/test_contact_and_orbit_fixes.py`.

### Resolved P2 - Kepler Elevation Mask Is A Conservative Approximation

The Kepler visibility path uses `horizon_angle - min_elevation`. The exact spherical relation for
ground central angle is `acos((R/r) * cos(elevation)) - elevation`; subtracting the elevation angle
is not equivalent. The implementation also adds the full configured atmosphere height to the
opaque body radius. With the current Earth defaults, a 550 km circular orbit and 10 degree mask has
a threshold of 10.775 degrees instead of 14.968 degrees. An ideal overhead window is about 6.12
minutes instead of 8.51 minutes, an approximate 28 percent underestimate for that example.

Resolution: Kepler visibility uses the exact spherical relation
`acos((R/r) * cos(elevation)) - elevation`, includes station altitude, and does not silently
use the full atmosphere-height field as a solid-body radius. Atmospheric attenuation remains a
separate link-model concern. Regression coverage includes the 550 km / 10 degree case.

### Resolved P2 - Eccentric Orbit Position Uses A Constant Radius

The anomaly solver changes angular position for eccentric orbits, but `ecef_at_time()` always uses
the semi-major axis as radius. Visibility and ISL geometry therefore do not represent perigee and
apogee distance changes. The project accepts six elements, but the runtime does not yet realize a
complete six-element orbit in position space.

Resolution: `KeplerOrbit.radius_at_time_km()` evaluates the instantaneous radius from true anomaly,
and `ecef_at_time()` uses it for eccentric-orbit geometry. Near- and far-point tests verify the
position norm.

### Remaining P2 - New Link And Learning-Task Models Are Not Runtime Inputs Yet

`CommunicationLinkSpec` and `CommunicationSystem` expose frequency, bandwidth, rate, range,
terminal count, acquisition, pointing, and weather fields. The web learning-task snapshot preserves
and displays them, but contact generation and `PhysicsTimeModel` do not consume those objects. The
new interface is useful as a canonical scenario description, not yet as simulation behavior.

Evidence: `fl_space/study/models.py:172`, `fl_space/mission/subsystems.py:124`, and no runtime
reference from `fl_space/simulator/` or `fl_space/fl/time_model.py`.

### Resolved P2 - Test And Offline Reproducibility Need A Single Entry Point

Resolution: `scripts/quality_gate.py` clears `PYTHONPATH`, disables third-party pytest autoload,
asserts the imported `fl_space` package root, and runs Ruff plus pytest from the repository root.
`SkyfieldProvider` loads only its built-in time scale by default; `load_ephemeris=True` explicitly
requests `de421.bsp`. This makes the normal gate offline, while preserving an opt-in precise
planetary-ephemeris path.

### Resolved P3 - Ruff Debt Obscures Higher-Value Signals

The Ruff debt has been cleared without widening the ignore list. The two legacy helper names that
looked like pytest tests were changed to `run_command_case` and `run_model_case`; the latter script
is now explicitly excluded from automatic collection.

## Current Contact Pipeline

```text
orbit elements / constellation
        -> sampled satellite position per timeslot
        -> geometric elevation/central-angle visibility per ground station
        -> ContactMatrix: first station and optionally all visible stations
        -> CommunicationScheduler: merge consecutive visible slots into windows and find
           single-station continuous transfer intervals
        -> FL server: reserve station intervals for download/upload, then train between them
        -> TimeModel: add nominal transfer duration from model bytes and fixed Mbps
```

The matrix still records geometric opportunity rather than RF link margin or delivered bytes. The
new scheduler reservation is a bounded one-terminal-per-station transfer guard; it is not yet a
capacity ledger driven by frequency, bandwidth, weather, acquisition, or queue state.

## Public Starlink Architecture And Gap

Public Starlink descriptions show a multi-layer network: electronically steered RF user terminals
track and hand over between moving satellites; satellites communicate with RF gateway stations; and
optical inter-satellite links route traffic across the constellation. Real availability also depends
on beam scheduling, spectrum and interference constraints, gateway capacity, weather, acquisition
and tracking, traffic demand, and dynamic routing. The community SpaceX API provides catalog and
Space-Track OMM/TLE fields, not operational gateway schedules, beam assignments, link telemetry, or
laser engineering parameters.

Gap assessment:

| Capability | Current SpaceFL | Public Starlink-like simulator target | Gap |
| --- | --- | --- | --- |
| Orbit propagation | Two-body sampling; generated SGP4 path with simplified ECI/ECEF fallback | Epoch-consistent SGP4/OMM with validated frames | Moderate: replace fallback with full frame/time handling for high fidelity |
| Ground contact | Boolean sampled visibility | Continuous windows, RF link margin, capacity, handover | Two model layers |
| Data transfer | Fixed-rate duration after a contact start | Bytes constrained by each active link/window | Major |
| Ground network | Position and elevation mask | Gateway beams, concurrency, queues, weather and spectrum | Major |
| ISL | Geometric LOS windows and assumed rate | Terminal topology, acquisition, pointing, routing and queues | Major |
| SpaceX data | Catalog mapping and explicit assumptions | Time-versioned OMM plus sourced public configuration | Moderate |

For relative FL algorithm experiments, the project is at a useful geometric-baseline stage after the
frame-conversion and code-quality gates are fixed. For a defensible public Starlink-like network simulator, two major layers are
still missing: link-capacity/resource simulation and dynamic network routing/handover. A true SpaceX
digital twin is not a realistic claim without private operational data; the achievable goal is a
source-traceable public approximation with uncertainty ranges.

## Recommended Order

1. Map learning-task link and ground-station specifications into simulator runtime configuration.
2. Replace the bounded one-terminal guard with a `ContactWindow` capacity ledger driven by
   bandwidth, acquisition, weather/link margin, and queue state; support transfers split across
   windows where the experiment requires it.
3. Add gateway concurrency, handover, weather/link margin, then optical-ISL routing.
4. Replace the simplified ECI/ECEF fallback with a validated high-precision frame/time path where needed.
