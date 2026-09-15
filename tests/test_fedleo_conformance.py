from fl_space.fedleo.conformance import (
    format_implementation_profile,
    get_implementation_profile,
)


def test_external_fedavg_backend_is_not_reported_as_integrated() -> None:
    profile = get_implementation_profile()
    assert profile["implementation_level"] == "lightweight_discrete_simulation"
    assert profile["external_backend"]["integrated"] is False
    assert "centralized" in profile["external_backend"]["reason"]


def test_profile_copy_does_not_mutate_module_metadata() -> None:
    profile = get_implementation_profile()
    profile["implemented"].clear()
    assert get_implementation_profile()["implemented"]


def test_terminal_profile_exposes_approximations() -> None:
    output = format_implementation_profile()
    assert "近似实现" in output
    assert "Ring-Allreduce" in output
    assert "已接入: 否" in output
