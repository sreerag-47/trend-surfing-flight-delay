"""
Data Preprocessing Module
Handles date creation, filtering, temporal indexing, imputation, and vector normalization.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


def prepare_dataframe(
    df: pd.DataFrame,
    temporal_cols: List[str],
    measure_col: str,
    dimension_cols: List[str],
    date_col_name: str = "Date",
    granularity: str = "daily"
) -> Tuple[pd.DataFrame, pd.DatetimeIndex, Dict[str, Any]]:
    """
    Cleans and prepares the DataFrame for trend surfing:
    1. Constructs the datetime representation from temporal columns.
    2. Drops unusable rows (missing dates or missing measures).
    3. Trims to only necessary columns.
    4. Computes the complete temporal index.
    
    Returns (cleaned_df, temporal_index, stats_summary).
    """
    initial_rows = len(df)
    cols_to_keep = list(set(temporal_cols + [measure_col] + dimension_cols))
    sub_df = df[cols_to_keep].copy()

    # Drop missing measure values
    sub_df = sub_df.dropna(subset=[measure_col])

    # Construct datetime column
    if len(temporal_cols) == 3:
        # Expected [Year, Month, Day/DayofMonth]
        try:
            year_col, month_col, day_col = temporal_cols
            dates = pd.to_datetime(
                {
                    "year": sub_df[year_col].astype(int),
                    "month": sub_df[month_col].astype(int),
                    "day": sub_df[day_col].astype(int),
                },
                errors="coerce"
            )
            sub_df[date_col_name] = dates
        except Exception:
            # Fallback to string concatenation
            date_str = (
                sub_df[temporal_cols[0]].astype(str) + "-" +
                sub_df[temporal_cols[1]].astype(str).str.zfill(2) + "-" +
                sub_df[temporal_cols[2]].astype(str).str.zfill(2)
            )
            sub_df[date_col_name] = pd.to_datetime(date_str, errors="coerce")
    elif len(temporal_cols) == 1:
        sub_df[date_col_name] = pd.to_datetime(sub_df[temporal_cols[0]], errors="coerce")
    else:
        # Multi-column fallback
        date_str = sub_df[temporal_cols].astype(str).agg("-".join, axis=1)
        sub_df[date_col_name] = pd.to_datetime(date_str, errors="coerce")

    # Drop invalid dates
    sub_df = sub_df.dropna(subset=[date_col_name])

    # Apply granularity if needed
    if granularity == "weekly":
        sub_df[date_col_name] = sub_df[date_col_name].dt.to_period("W").dt.to_timestamp()
    elif granularity == "monthly":
        sub_df[date_col_name] = sub_df[date_col_name].dt.to_period("M").dt.to_timestamp()
    else:
        # Default daily: normalize to midnight
        sub_df[date_col_name] = sub_df[date_col_name].dt.floor("D")

    # Ensure measure is float
    sub_df[measure_col] = sub_df[measure_col].astype(float)

    # Cast dimensions to string for consistent grouping
    for dim in dimension_cols:
        sub_df[dim] = sub_df[dim].fillna("Unknown").astype(str)

    # Final columns to keep
    final_cols = [date_col_name, measure_col] + dimension_cols
    cleaned_df = sub_df[final_cols].copy()

    # Complete temporal index
    unique_dates = pd.DatetimeIndex(np.sort(cleaned_df[date_col_name].unique()))

    stats = {
        "initial_rows": initial_rows,
        "valid_rows": len(cleaned_df),
        "dropped_rows": initial_rows - len(cleaned_df),
        "retention_rate": round((len(cleaned_df) / initial_rows) * 100, 2) if initial_rows > 0 else 0.0,
        "time_points": len(unique_dates),
        "start_date": str(unique_dates[0].date()) if len(unique_dates) > 0 else "",
        "end_date": str(unique_dates[-1].date()) if len(unique_dates) > 0 else "",
        "granularity": granularity,
    }

    return cleaned_df, unique_dates, stats


def impute_trend_series(
    series: pd.Series,
    temporal_index: pd.DatetimeIndex,
    strategy: str = "interpolate",
    fill_value: float = 0.0,
    reference_trend: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Aligns a time series to the full temporal index and imputes missing dates.
    
    Strategies:
    - 'interpolate': linear interpolation, with ffill/bfill for edges
    - 'ffill_bfill': forward fill then backward fill
    - 'reference': fill missing points using values from a reference trend
    - 'constant': fill with fill_value (e.g., 0.0 or series mean)
    """
    reindexed = series.reindex(temporal_index)

    if strategy == "interpolate":
        # Linear interpolation + bfill + ffill
        imputed = reindexed.interpolate(method="time").bfill().ffill()
        if imputed.isnull().any():
            imputed = imputed.fillna(fill_value)
        return imputed.to_numpy(dtype=float)

    elif strategy == "ffill_bfill":
        imputed = reindexed.ffill().bfill().fillna(fill_value)
        return imputed.to_numpy(dtype=float)

    elif strategy == "reference" and reference_trend is not None and len(reference_trend) == len(temporal_index):
        arr = reindexed.to_numpy(dtype=float)
        mask = np.isnan(arr)
        arr[mask] = reference_trend[mask]
        return arr

    elif strategy == "constant":
        return reindexed.fillna(fill_value).to_numpy(dtype=float)

    else:
        # Default fallback: series mean or 0
        fallback = series.mean() if not np.isnan(series.mean()) else fill_value
        return reindexed.fillna(fallback).to_numpy(dtype=float)


def normalize_trend_vector(vector: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalizes a trend vector to isolate shape deviations from absolute scale.
    
    Methods:
    - 'zscore': (v - mean) / std (shapes centered at 0, unit variance)
    - 'center': v - mean (preserves amplitude scale, removes global offset)
    - 'minmax': (v - min) / (max - min + eps)
    - 'none': raw values
    """
    if method == "none":
        return vector.copy()

    arr = np.asarray(vector, dtype=float)
    if len(arr) == 0:
        return arr

    if method == "zscore":
        std = np.std(arr)
        if std < 1e-8:
            return np.zeros_like(arr)
        return (arr - np.mean(arr)) / std

    elif method == "center":
        return arr - np.mean(arr)

    elif method == "minmax":
        min_val = np.min(arr)
        max_val = np.max(arr)
        denom = max_val - min_val
        if denom < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / denom

    else:
        return arr.copy()
