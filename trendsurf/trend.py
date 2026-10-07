"""
Trend Representation & Visualization Module
Defines the Trend object and functions to calculate global and group temporal trends.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from trendsurf.preprocessing import impute_trend_series, normalize_trend_vector


@dataclass
class Trend:
    """
    Represents a temporal trend for an entire dataset or a specific subgroup.
    """
    label: str
    raw_values: np.ndarray
    normalized_values: np.ndarray
    dates: pd.DatetimeIndex
    observation_count: int
    missing_ratio: float
    mean_val: float
    std_val: float
    min_val: float
    max_val: float
    normalization_method: str = "zscore"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "observation_count": self.observation_count,
            "missing_ratio": round(self.missing_ratio, 4),
            "mean_val": round(float(self.mean_val), 3),
            "std_val": round(float(self.std_val), 3),
            "min_val": round(float(self.min_val), 3),
            "max_val": round(float(self.max_val), 3),
            "length": len(self.raw_values),
            "normalization_method": self.normalization_method,
        }


def compute_global_trend(
    df: pd.DataFrame,
    date_col: str,
    measure_col: str,
    temporal_index: pd.DatetimeIndex,
    imputation: str = "interpolate",
    normalization: str = "zscore"
) -> Trend:
    """
    Computes the depth-0 (all data) global temporal trend.
    """
    grouped = df.groupby(date_col)[measure_col].mean()
    total_dates = len(temporal_index)
    missing_dates = total_dates - len(grouped)
    missing_ratio = (missing_dates / total_dates) if total_dates > 0 else 0.0

    raw_imputed = impute_trend_series(grouped, temporal_index, strategy=imputation)
    norm_values = normalize_trend_vector(raw_imputed, method=normalization)

    return Trend(
        label="ALL (Global)",
        raw_values=raw_imputed,
        normalized_values=norm_values,
        dates=temporal_index,
        observation_count=len(df),
        missing_ratio=missing_ratio,
        mean_val=float(np.mean(raw_imputed)),
        std_val=float(np.std(raw_imputed)),
        min_val=float(np.min(raw_imputed)),
        max_val=float(np.max(raw_imputed)),
        normalization_method=normalization,
    )


def compute_group_trend(
    sub_df: pd.DataFrame,
    label: str,
    date_col: str,
    measure_col: str,
    temporal_index: pd.DatetimeIndex,
    imputation: str = "interpolate",
    normalization: str = "zscore",
    reference_raw: Optional[np.ndarray] = None
) -> Optional[Trend]:
    """
    Computes the temporal trend for a specific subgroup DataFrame.
    Returns None if sub_df is empty.
    """
    if sub_df.empty:
        return None

    obs_count = len(sub_df)
    grouped = sub_df.groupby(date_col)[measure_col].mean()
    total_dates = len(temporal_index)
    missing_dates = total_dates - len(grouped)
    missing_ratio = (missing_dates / total_dates) if total_dates > 0 else 0.0

    raw_imputed = impute_trend_series(
        grouped,
        temporal_index,
        strategy=imputation,
        reference_trend=reference_raw,
        fill_value=float(sub_df[measure_col].mean() if not np.isnan(sub_df[measure_col].mean()) else 0.0)
    )
    norm_values = normalize_trend_vector(raw_imputed, method=normalization)

    return Trend(
        label=label,
        raw_values=raw_imputed,
        normalized_values=norm_values,
        dates=temporal_index,
        observation_count=obs_count,
        missing_ratio=missing_ratio,
        mean_val=float(np.mean(raw_imputed)),
        std_val=float(np.std(raw_imputed)),
        min_val=float(np.min(raw_imputed)),
        max_val=float(np.max(raw_imputed)),
        normalization_method=normalization,
    )


def render_ascii_sparkline(vector: np.ndarray, length: int = 24) -> str:
    """
    Creates an ASCII sparkline representation of a numerical vector.
    """
    if len(vector) == 0:
        return ""
    ticks = ["_", ".", "-", "~", "=", "+", "*", "#"]
    
    # Resample vector to fixed length
    if len(vector) > length:
        indices = np.linspace(0, len(vector) - 1, length, dtype=int)
        sampled = vector[indices]
    else:
        sampled = vector

    v_min, v_max = np.min(sampled), np.max(sampled)
    if v_max == v_min:
        return ticks[3] * len(sampled)

    norm = (sampled - v_min) / (v_max - v_min)
    indices = np.clip((norm * (len(ticks) - 1)).astype(int), 0, len(ticks) - 1)
    return "".join(ticks[idx] for idx in indices)


def plot_trend_terminal(
    trend: Trend,
    reference_trend: Optional[Trend] = None,
    title: Optional[str] = None,
    width: int = 70,
    height: int = 15
) -> str:
    """
    Uses plotext to generate an ANSI terminal chart comparing the trend with reference.
    Falls back to textual summary if plotext fails.
    """
    try:
        import plotext as plt
        plt.clf()
        plt.theme("clear")
        plt.plotsize(width, height)
        plt.title(title or f"Temporal Trend: {trend.label}")

        x_vals = [str(d.date()) for d in trend.dates]

        if reference_trend is not None:
            plt.plot(reference_trend.raw_values, label=f"Ref: {reference_trend.label}", color="cyan")
        
        plt.plot(trend.raw_values, label=f"Subgroup: {trend.label}", color="red")
        
        plt.xlabel("Time Index")
        plt.ylabel("Mean Measure")
        chart_str = plt.build()
        plt.clf()
        return chart_str
    except Exception as e:
        spark = render_ascii_sparkline(trend.raw_values)
        return f"Trend {trend.label} [Sparkline: {spark}] (Mean: {trend.mean_val:.2f}, Min: {trend.min_val:.2f}, Max: {trend.max_val:.2f})"
