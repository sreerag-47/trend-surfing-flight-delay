import pandas as pd
import numpy as np
from trendsurf.preprocessing import prepare_dataframe, impute_trend_series, normalize_trend_vector


def test_prepare_dataframe():
    df = pd.DataFrame({
        "Year": [2008, 2008, 2008, 2008],
        "Month": [1, 1, 1, 1],
        "DayofMonth": [1, 2, 3, 4],
        "ArrDelay": [10.0, np.nan, 25.0, 40.0],
        "UniqueCarrier": ["WN", "WN", "AA", "AA"],
        "Origin": ["IAD", "IAD", "ATL", "ATL"]
    })
    cleaned_df, t_index, stats = prepare_dataframe(
        df=df,
        temporal_cols=["Year", "Month", "DayofMonth"],
        measure_col="ArrDelay",
        dimension_cols=["UniqueCarrier", "Origin"]
    )
    assert len(cleaned_df) == 3  # 1 NaN dropped
    assert "Date" in cleaned_df.columns
    assert len(t_index) == 3
    assert stats["valid_rows"] == 3


def test_impute_trend_series():
    dates = pd.date_range("2008-01-01", periods=5, freq="D")
    # Series missing index 2008-01-03
    series = pd.Series([10.0, 20.0, 40.0, 50.0], index=[dates[0], dates[1], dates[3], dates[4]])
    imputed = impute_trend_series(series, dates, strategy="interpolate")
    assert len(imputed) == 5
    assert not np.isnan(imputed).any()
    assert np.isclose(imputed[2], 30.0)  # linearly interpolated between 20 and 40


def test_normalize_trend_vector():
    vec = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    norm = normalize_trend_vector(vec, method="zscore")
    assert np.isclose(np.mean(norm), 0.0)
    assert np.isclose(np.std(norm), 1.0)

    minmax = normalize_trend_vector(vec, method="minmax")
    assert np.isclose(minmax[0], 0.0)
    assert np.isclose(minmax[-1], 1.0)
