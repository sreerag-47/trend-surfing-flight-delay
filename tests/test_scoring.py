import numpy as np
import pytest
from trendsurf.scoring import (
    euclidean_distance,
    rmsd_score,
    compute_pca_outlier_score,
    compute_knn_outlier_score,
    score_trend
)
from trendsurf.trend import Trend
import pandas as pd


def test_euclidean_distance():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([1.0, 2.0, 3.0])
    assert euclidean_distance(v1, v2) == 0.0

    v3 = np.array([4.0, 6.0, 3.0])
    # sqrt(3^2 + 4^2 + 0^2) = 5.0
    assert euclidean_distance(v1, v3) == 5.0


def test_score_trend_identical_and_outlier():
    dates = pd.date_range("2008-01-01", periods=3)
    t1 = Trend(
        label="T1",
        raw_values=np.array([1.0, 2.0, 3.0]),
        normalized_values=np.array([0.0, 0.0, 0.0]),
        dates=dates,
        observation_count=10,
        missing_ratio=0.0,
        mean_val=2.0,
        std_val=1.0,
        min_val=1.0,
        max_val=3.0
    )
    t2 = Trend(
        label="T2",
        raw_values=np.array([1.0, 2.0, 3.0]),
        normalized_values=np.array([0.0, 0.0, 0.0]),
        dates=dates,
        observation_count=10,
        missing_ratio=0.0,
        mean_val=2.0,
        std_val=1.0,
        min_val=1.0,
        max_val=3.0
    )
    # Identical trends have zero distance
    sc_ident = score_trend(t1, t2, method="euclidean")
    assert np.isclose(sc_ident, 0.0)

    t3 = Trend(
        label="T3",
        raw_values=np.array([10.0, 20.0, 30.0]),
        normalized_values=np.array([3.0, 4.0, 0.0]),
        dates=dates,
        observation_count=10,
        missing_ratio=0.0,
        mean_val=20.0,
        std_val=8.0,
        min_val=10.0,
        max_val=30.0
    )
    sc_diff = score_trend(t3, t1, method="euclidean")
    assert sc_diff == 5.0
