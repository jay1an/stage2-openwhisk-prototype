import numpy as np

from runner.stage2_forecastor.entry_cold_estimator import (
    available_from_poolstate,
    estimate_entry_cold_rate,
    p_entry_cold,
)


def test_p_entry_cold_bounds_and_high_available():
    mean = np.array([0.2, 1.0, 4.0, 8.0])
    for dist, alpha in [("poisson", 0.0), ("nb", 0.35)]:
        values = p_entry_cold(mean, np.array([0.0, 1.0, 3.0, 100.0]), dist=dist, alpha=alpha)
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)
        assert values[-1] < 1e-8
        assert p_entry_cold(3.0, 0.0, dist=dist, alpha=alpha) == 1.0


def test_p_entry_cold_monotonic_in_available():
    mean = np.full(6, 3.5)
    availabilities = [0, 1, 2, 3, 5, 10]
    for dist, alpha in [("poisson", 0.0), ("nb", 0.5)]:
        rates = [
            estimate_entry_cold_rate(mean, np.full_like(mean, available), dist=dist, alpha=alpha)
            for available in availabilities
        ]
        assert rates == sorted(rates, reverse=True)


def test_nb_has_heavier_tail_than_poisson_for_same_mean_and_available():
    mean = np.array([5.0])
    available = np.array([7.0])
    poisson_rate = estimate_entry_cold_rate(mean, available, dist="poisson", alpha=0.0)
    nb_rate = estimate_entry_cold_rate(mean, available, dist="nb", alpha=0.6)
    assert nb_rate > poisson_rate


def test_available_from_poolstate_uses_free_plus_warming_and_normalizes_action():
    rows = [
        {
            "action": "/guest/wf_civic_detect_object_3072",
            "memoryMB": 3072,
            "free": 2,
            "busy": 99,
            "warming": 3,
        },
        {
            "action": "wf_civic_detect_object_1280",
            "memoryMB": 1280,
            "free": 4,
            "busy": 1,
            "warming": 0,
        },
    ]
    available = available_from_poolstate(rows)
    assert available[("/guest/wf_civic_detect_object_3072", 3072)] == 5
    assert available[("wf_civic_detect_object_3072", 3072)] == 5
    assert available[("wf_civic_detect_object_1280", 1280)] == 4
