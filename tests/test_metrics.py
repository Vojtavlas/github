import pytest

from reasonflow import mean_speedup, speedup


def test_speedup_valid():
    assert speedup(100.0, 50.0) == 2.0
    assert speedup(100.0, 100.0) == 1.0
    assert speedup(50.0, 100.0) == 0.5


def test_speedup_zero_optimized_raises():
    with pytest.raises(ValueError):
        speedup(100.0, 0.0)


def test_speedup_negative_optimized_raises():
    with pytest.raises(ValueError):
        speedup(100.0, -10.0)


def test_speedup_non_positive_baseline_raises():
    with pytest.raises(ValueError):
        speedup(0.0, 50.0)
    with pytest.raises(ValueError):
        speedup(-10.0, 50.0)


def test_mean_speedup_valid():
    measurements = [(100.0, 50.0), (200.0, 100.0)]
    # (100 + 200) / (50 + 100) = 300 / 150 = 2.0
    assert mean_speedup(measurements) == 2.0


def test_mean_speedup_zero_optimized_raises():
    with pytest.raises(ValueError):
        mean_speedup([(100.0, 0.0)])


def test_mean_speedup_negative_optimized_raises():
    with pytest.raises(ValueError):
        mean_speedup([(100.0, -10.0)])


def test_mean_speedup_non_positive_baseline_raises():
    with pytest.raises(ValueError):
        mean_speedup([(0.0, 50.0)])
    with pytest.raises(ValueError):
        mean_speedup([(-10.0, 50.0)])
