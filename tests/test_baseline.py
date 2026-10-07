import pandas as pd
import numpy as np
from trendsurf.baseline import run_exhaustive_search, compare_surfer_with_baseline
from trendsurf.surfers.surfer2 import run_trendsurfer2
from trendsurf.trend import compute_global_trend


def test_exhaustive_baseline_and_comparison():
    dates = pd.date_range("2008-01-01", periods=6, freq="D")
    data = []
    carriers = ["WN", "AA"]
    origins = ["ORD", "ATL"]

    for d in dates:
        for c in carriers:
            for o in origins:
                data.append({"Date": d, "ArrDelay": 20.0, "UniqueCarrier": c, "Origin": o})

    df = pd.DataFrame(data)
    gt = compute_global_trend(df, "Date", "ArrDelay", dates)

    base_res = run_exhaustive_search(
        df=df,
        global_trend=gt,
        dimension_cols=["UniqueCarrier", "Origin"],
        date_col="Date",
        measure_col="ArrDelay",
        temporal_index=dates,
        target_depth=2,
        min_support=2
    )
    assert base_res.total_nodes_evaluated > 0

    surf_res = run_trendsurfer2(
        df=df,
        global_trend=gt,
        dimension_cols=["UniqueCarrier", "Origin"],
        date_col="Date",
        measure_col="ArrDelay",
        temporal_index=dates,
        target_depth=2,
        min_support=2
    )

    comparison = compare_surfer_with_baseline(surf_res, base_res)
    assert "search_reduction_pct" in comparison
    assert "speedup_factor" in comparison
