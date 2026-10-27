# SpaceFL Project Map

Last verified: 2026-09-20

## Purpose

SpaceFL is a modular research framework for federated learning in space and LEO satellite-network settings. This document is the entry point for agents and contributors. It records current code boundaries and evidence, not aspirational features.

## Architecture

```text
integrations -> study -> mission
                  |        |
                  v        v
config -> environment + orbit -> simulator -> FL core -> experiments/results
                              |              |
                              v              v
                           ISL model       FedLEO extension
                                              |
                                              v
                                         CLI / web / visualization
```

| Area | Ownership | Key paths | Current role |
| --- | --- | --- | --- |
| Configuration | Loading and schemas | `fl_space/config/` | Defaults, schema, and YAML/config loading |
| Environment | Earth and ground stations | `fl_space/environment/` | Celestial body, atmosphere, coordinate, ground-station models |
| Orbit | Propagation and visibility | `fl_space/orbit/` | Kepler and optional Skyfield backends, satellite registry/configuration, visibility |
| Simulation | Contacts and scheduling | `fl_space/simulator/` | Orbit simulation and contact matrix; scheduling-related utilities |
| Inter-satellite links | ISL abstraction | `fl_space/isl/` | LOS and intra-cluster link behavior |
| Mission | Launch and spacecraft domain | `fl_space/mission/` | Launch vehicles and staging, satellite designs and subsystems, launch windows, deployment plans, constraints, first-order physics/budget |
| Study | Top-level research scenario | `fl_space/study/` | Dataset, model, spacecraft/orbit, infrastructure, links, and scheduler declarations |
| Local cluster lab | Isolated paper-inspired cluster benchmark | `fl_space/local_cluster/` | Optical-link budget, analytical local workload, quantified profile, and opt-in global adapter |
| FL core | Algorithms and timing | `fl_space/fl/` | FedAvg, FedProx, FedBuff, scheduler, time model, runner, validation |
| FedLEO | Research extension | `fl_space/fedleo/` | Decentralized offloading planner, aggregation, metrics, experiment, conformance metadata |
| Integrations | External adapters | `fl_space/integrations/` | Flower adapter and read-only r-spacex catalog mapping |
| Interfaces | CLI, web, visualization | `fl_space/cli.py`, `web/`, `fl_space/viz/` | Experiment and visualization access paths |

## Evidence And Entry Points

| Need | Start here | Evidence/output |
| --- | --- | --- |
| Package and basic algorithms | `tests/test_fl_algorithms.py`, `tests/test_quick.py` | Algorithm and smoke-test coverage |
| Orbit/time behavior | `tests/test_time_model.py`, `tests/test_satellite_data_profiles.py` | Timing and satellite data behavior |
| FedLEO fidelity boundary | `fl_space/fedleo/conformance.py` | Explicit implemented, approximated, and missing behavior |
| FedLEO offloading validation | `scripts/validate_fedleo_offloading.py` | On/off results in `fedleo_web_validation_*/` |
| FedLEO tests | `tests/test_fedleo_offloading.py`, `tests/test_fedleo_conformance.py` | Planner and boundary metadata coverage |
| CLI and web workflows | `tests/test_cli_params.py`, `tests/test_web_*.py` | Parameter and session/orbit flow coverage |
| Launch/deployment foundation | `tests/test_mission_framework.py`, `examples/mission_deployment_demo.py` | Domain models, constraint gates, first-order launch budget and deployed `KeplerOrbit` integration |
| Learning-task snapshot | `fl_space/study/`, `tests/test_learning_task_framework.py`, `tests/test_web_learning_task.py` | Revisioned server snapshot for data/model/nodes/infrastructure/scheduler visualization |
| SpaceX catalog boundary | `fl_space/integrations/spacex_api.py`, `docs/knowledge/spacex_public_data.md` | OMM/payload mapping and explicitly assumed optical-link profile; no operational contact telemetry |
| Local cluster benchmark | `fl_space/local_cluster/`, `tests/test_local_cluster.py`, `/api/local-cluster/run` | Independent cluster construction and `local_cluster_profile_v1`; no global-session side effects |
| Quality and contact-model audit | `docs/CODE_QUALITY_AUDIT_2026-09-18.md` | Current Ruff/test gate, contact pipeline, resolved frame issue, remaining defects, and Starlink gap assessment |
| Coding standards | `CODING_STANDARDS.md`, `pyproject.toml` | Ruff is the only formatter/linter |
| Change provenance | `WORKLOG.md` | Historical implementation and validation notes |

## Satellite-Ground Contact Boundary

The current runtime computes a sampled geometric visibility matrix. The Kepler path compares the
ground-station central angle with the exact spherical elevation-mask relation and instantaneous
Kepler radius; the optional Skyfield path samples topocentric elevation after a simplified
GMST-based ECI/TEME-to-ECEF rotation for generated SGP4 satellites. `ContactMatrix` stores either
the first visible station or all visible stations. `CommunicationScheduler` merges consecutive
visible slots and can find a continuous single-station transfer interval with a per-round
one-terminal reservation guard.

This is still not a full link-availability or traffic-delivery model. Frequency, bandwidth, rate,
terminal count, link acquisition, weather, beam capacity, interference, handover, and routing are
declared in the mission/study layers but are not all coupled to contact generation. FL transfer
time must now fit one continuous geometric window and station reservations prevent same-round
overlap on one station, but the rate is still supplied by the selected time model. Treat current
results as relative algorithm experiments using sampled geometric opportunity, not as Starlink
network capacity or service availability.

## Research Assets

Local literature and analysis live in `文献/`. Primary current anchors include:

- FedLEO: `文献/[22]_Zhai2024_FedLEO_中文翻译.md` and `fl_space/fedleo/`.
- Ground-assisted and scheduling baselines: `文献/[20]_Razmi2022_Ground-Assisted_FL_LEO_中文翻译.md` and `文献/[21]_Razmi2022_Scheduling_FL_LEO_中文翻译.md`.
- Communication-efficient decentralized direction: `文献/[23]_DFedSat2025_通信高效鲁棒去中心化FL_中文分析.md` and `文献/README_通信压缩模块文献指南.md`.
- Broader direction and project choices: `太空联邦学习研究分析报告.md` and `docs/基础太空联邦算法改进与开源采用建议_20260813.md`.

## FedLEO Boundary

`fl_space/fedleo/conformance.py` is authoritative for the FedLEO implementation claim. At this map revision, the repository provides a lightweight discrete simulation with non-IID satellite datasets, ISL-neighbor offloading, greedy iterative offloading, weighted intra/inter-plane aggregation, and metrics.

It deliberately approximates discrete ratio search, divergence, ring aggregation, delay, and topology. It does not implement the paper's KKT communication-power optimization, multi-hop streaming contention/channel gain, dynamic cross-seam contact constraints, or full continuous P1-P4 optimization. Do not state otherwise without changing that metadata and adding validation.

## Near-Term Direction

1. Connect study/mission link parameters to the simulator and FL time model.
2. Replace the bounded one-terminal reservation guard with continuous-window capacity accounting,
   including transfer splitting when explicitly enabled.
3. Connect `local_cluster_profile_v1` to a real dataset/model training callback so the current
   analytical workload can be compared with measured accuracy and convergence, without merging
   local and global run state.
4. Implement communication compression as a bounded experiment, beginning with a measurable byte/time model and a no-compression control. The literature guide proposes integration points in `fl_space/fedleo/aggregator.py`, `fl_space/isl/base.py`, `fl_space/fl/fedbuff.py`, `fl_space/fl/scheduler.py`, and `fl_space/fl/time_model.py`.
5. Improve dynamic topology and contact realism before making stronger decentralized-FL claims.
6. Use multi-seed controlled experiments to measure accuracy, convergence time, transmitted bytes, delay/staleness, fairness, and resource cost together.
7. Retain clear separation between simulator fidelity and operational satellite feasibility.
8. Extend the mission layer with explicit vehicle staging, payload allocation, and research/visual
   asset plugins without embedding game rules in orbit propagation.

## Quality Gate

The verified local gate is:

```powershell
$env:PYTHONPATH = ""
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.\venv\Scripts\python.exe -m ruff check .
.\venv\Scripts\python.exe -m ruff format --check .
.\venv\Scripts\python.exe -m pytest -q
```

The 2026-09-20 run reported zero Ruff findings, 231 formatted files, and 59 passing tests.

## Maintenance Rules

- Update this map after changing a module boundary, implementation classification, experiment entry point, validated result, or research direction.
- Verify every code-status statement against source/tests and every research-status statement against a local paper or result artifact.
- Put chronological detail in `WORKLOG.md`; keep this document concise and current.
- Record command, config, seed, revision, metrics, and output path for material experiments.
- For mission work, also record units, fidelity label, external data source, and assumptions.
