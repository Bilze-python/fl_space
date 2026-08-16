# SpaceFL Project Map

Last verified: 2026-08-16

## Purpose

SpaceFL is a modular research framework for federated learning in space and LEO satellite-network settings. This document is the entry point for agents and contributors. It records current code boundaries and evidence, not aspirational features.

## Architecture

```text
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
| FL core | Algorithms and timing | `fl_space/fl/` | FedAvg, FedProx, FedBuff, scheduler, time model, runner, validation |
| FedLEO | Research extension | `fl_space/fedleo/` | Decentralized offloading planner, aggregation, metrics, experiment, conformance metadata |
| Integrations | External adapters | `fl_space/integrations/flower/` | Flower adapter |
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
| Coding standards | `CODING_STANDARDS.md`, `pyproject.toml` | Ruff is the only formatter/linter |
| Change provenance | `WORKLOG.md` | Historical implementation and validation notes |

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

1. Implement communication compression as a bounded experiment, beginning with a measurable byte/time model and a no-compression control. The literature guide proposes integration points in `fl_space/fedleo/aggregator.py`, `fl_space/isl/base.py`, `fl_space/fl/fedbuff.py`, `fl_space/fl/scheduler.py`, and `fl_space/fl/time_model.py`.
2. Improve dynamic topology and contact realism before making stronger decentralized-FL claims.
3. Use multi-seed controlled experiments to measure accuracy, convergence time, transmitted bytes, delay/staleness, fairness, and resource cost together.
4. Retain clear separation between simulator fidelity and operational satellite feasibility.

## Maintenance Rules

- Update this map after changing a module boundary, implementation classification, experiment entry point, validated result, or research direction.
- Verify every code-status statement against source/tests and every research-status statement against a local paper or result artifact.
- Put chronological detail in `WORKLOG.md`; keep this document concise and current.
- Record command, config, seed, revision, metrics, and output path for material experiments.
