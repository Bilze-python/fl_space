from __future__ import annotations

import pytest
from fastapi import HTTPException

from web import server


def test_session_update_validates_and_normalizes_satellite_profiles() -> None:
    current = server._default_session()
    result = server._validated_session_update(
        current,
        {
            "mount": {"sats": 3},
            "tune": {
                "protocol_mode": "paper_approx",
                "selection_strategy": "earliest_return",
                "satellite_data_profiles": {
                    "1": {
                        "preferred_classes": [7, 2, 7],
                        "preference_probability": 0.9,
                        "max_samples": 240,
                    }
                },
            },
        },
    )
    assert result["tune"]["satellite_data_profiles"] == {
        "1": {
            "preferred_classes": [2, 7],
            "preference_probability": 0.9,
            "max_samples": 240,
        }
    }


def test_session_update_rejects_profile_outside_constellation() -> None:
    with pytest.raises(HTTPException) as error:
        server._validated_session_update(
            server._default_session(),
            {
                "mount": {"sats": 2},
                "tune": {
                    "satellite_data_profiles": {
                        "2": {"preferred_classes": [0], "preference_probability": 0.8}
                    }
                },
            },
        )
    assert error.value.status_code == 422


def test_session_profiles_round_trip_through_storage(tmp_path) -> None:
    session_path = tmp_path / "session.json"
    saved = server._validated_session_update(
        server._default_session(),
        {
            "mount": {"sats": 4},
            "tune": {
                "satellite_data_profiles": {
                    "3": {
                        "preferred_classes": [1, 9],
                        "preference_probability": 0.85,
                        "max_samples": 180,
                    }
                }
            },
        },
    )
    server._write_json(session_path, saved)
    loaded = server._read_json(session_path, {})
    assert loaded["mount"]["sats"] == 4
    assert loaded["tune"]["satellite_data_profiles"]["3"]["max_samples"] == 180
