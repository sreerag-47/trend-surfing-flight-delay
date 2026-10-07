import pandas as pd
import numpy as np
from trendsurf.trend import compute_global_trend, compute_group_trend, render_ascii_sparkline


def test_compute_global_and_group_trend():
    dates = pd.date_range("2008-01-01", periods=4, freq="D")
    df = pd.DataFrame({
        "Date": [dates[0], dates[1], dates[2], dates[3], dates[0]],
        "ArrDelay": [10.0, 20.0, 30.0, 40.0, 50.0],
        "Carrier": ["WN", "WN", "AA", "AA", "WN"]
    })

    gt = compute_global_trend(df, "Date", "ArrDelay", dates, imputation="interpolate")
    assert gt.label == "ALL (Global)"
    assert len(gt.raw_values) == 4
    assert gt.observation_count == 5

    sub_df = df[df["Carrier"] == "AA"]
    group_tr = compute_group_trend(sub_df, "Carrier=AA", "Date", "ArrDelay", dates, reference_raw=gt.raw_values)
    assert group_tr is not None
    assert group_tr.observation_count == 2
    assert len(group_tr.raw_values) == 4


def test_render_ascii_sparkline():
    vec = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    spark = render_ascii_sparkline(vec, length=9)
    assert len(spark) == 9
    assert isinstance(spark, str)
