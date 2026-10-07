import pandas as pd
import numpy as np
from trendsurf.surfers.surfer1 import run_trendsurfer1
from trendsurf.trend import compute_global_trend


def test_trendsurfer1_execution():
    dates = pd.date_range("2008-01-01", periods=10, freq="D")
    data = []
    carriers = ["WN", "AA", "UA"]
    origins = ["ORD", "ATL", "DFW"]

    for d in dates:
        for c in carriers:
            for o in origins:
                # Add outlier spike for WN at ORD
                delay = 100.0 if (c == "WN" and o == "ORD" and d.day > 5) else 15.0
                data.append({"Date": d, "ArrDelay": delay, "UniqueCarrier": c, "Origin": o})

    df = pd.DataFrame(data)
    gt = compute_global_trend(df, "Date", "ArrDelay", dates)

    res = run_trendsurfer1(
        df=df,
        global_trend=gt,
        dimension_cols=["UniqueCarrier", "Origin"],
        date_col="Date",
        measure_col="ArrDelay",
        temporal_index=dates,
        target_depth=2,
        min_support=5
    )

    assert res.algorithm == "TrendSurfer I"
    assert len(res.path) == 3  # depth 0, 1, 2
    assert res.nodes_opened > 0
    assert res.execution_time >= 0.0
    assert "UniqueCarrier" in res.final_node.dimensions
    assert "Origin" in res.final_node.dimensions
