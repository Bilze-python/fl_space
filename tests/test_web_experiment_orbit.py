import json
from pathlib import Path

from web import server


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_archive_parameters_and_metrics_are_mapped_to_orbit(monkeypatch, tmp_path):
    archive = tmp_path / "experiments" / "validation_results" / "archive-1"
    _write_json(
        archive / "validation_summary.json",
        {
            "config": {
                "num_planes": 2,
                "sats_per_plane": 3,
                "num_rounds": 2,
                "timeslot_duration_sec": 30,
                "dataset": "mnist",
                "seed": 17,
            },
            "offload_on": {"final_accuracy": 0.8, "total_offloaded": 12},
            "offload_off": {"final_accuracy": 0.7},
            "deltas_on_minus_off": {"final_accuracy": 0.1},
            "passed": True,
        },
    )
    _write_json(
        archive / "offload_on" / "fedleo_offload.json",
        {
            "history": [
                {"round": 0, "accuracy": 0.4, "total_offloaded_samples": 0},
                {"round": 1, "accuracy": 0.8, "total_offloaded_samples": 12},
            ]
        },
    )
    _write_json(archive / "offload_off" / "fedleo_no_offload.json", {"history": []})
    _write_json(
        tmp_path / ".fls_session.json",
        {
            "tune": {},
            "mount": {
                "sats": 5,
                "stations": 4,
                "altitude": 610,
                "inclination": 70,
                "timeslot_min": 2,
                "isl": "disabled",
                "isl_buffer": 0,
            },
        },
    )

    captured = {}

    def fake_orbit(**kwargs):
        captured.update(kwargs)
        return {
            "satellites": kwargs["sats"],
            "ground_stations": [{}] * kwargs["gs"],
            "timeslots": [{"ts": 0}, {"ts": 1}],
        }

    monkeypatch.setattr(server, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(server, "SESSION_FILE", tmp_path / ".fls_session.json")
    monkeypatch.setattr(server, "_default_session", lambda: {"tune": {}, "mount": {}})
    monkeypatch.setattr(server, "build_orbit_data", fake_orbit)

    result = server.build_experiment_orbit_data("archive-1")

    assert captured == {
        "sim_hours": 24.0,
        "sats": 6,
        "gs": 4,
        "altitude_km": 610.0,
        "inclination_deg": 70.0,
        "timeslot_min": 10.0,
        "isl_enabled": False,
        "isl_buffer": 0.0,
        "seed": 17,
    }
    assert result["source"] == "experiment_archive"
    assert result["experiment"]["id"] == "archive-1"
    assert result["timeslots"][1]["experiment"] == {
        "round": 1,
        "accuracy": 0.8,
        "total_offloaded_samples": 12,
    }
    assert result["parameter_sources"]["satellites"] == "experiment.config"
    assert result["parameter_sources"]["altitude"] == "session.mount"
    assert result["parameter_sources"]["timeslot"] == "projection"
    assert result["projection"] == {
        "days": 1.0,
        "step_min": 10.0,
        "frames": 2,
        "profile": "standard",
    }
