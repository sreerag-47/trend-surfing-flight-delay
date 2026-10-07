"""
Dimension Explorer & Top Unusual Trends Module
Supports exploring single dimensions and retrieving top-K anomalous subgroups.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from trendsurf.trend import Trend, compute_group_trend, render_ascii_sparkline
from trendsurf.scoring import score_trend
from trendsurf.datacube import CubeNode, filter_subgroup


def explore_dimension(
    df: pd.DataFrame,
    dimension: str,
    global_trend: Trend,
    date_col: str,
    measure_col: str,
    temporal_index: pd.DatetimeIndex,
    scoring_method: str = "euclidean",
    normalization: str = "zscore",
    imputation: str = "interpolate",
    min_support: int = 20,
    top_n: int = 20
) -> List[Dict[str, Any]]:
    """
    Evaluates all distinct values of a single dimension and ranks them by outlier score.
    """
    results = []
    val_counts = df[dimension].value_counts()
    valid_vals = val_counts[val_counts >= min_support]

    for val, count in valid_vals.items():
        sub_df = df[df[dimension].astype(str) == str(val)]
        label = f"{dimension}={val}"

        trend = compute_group_trend(
            sub_df=sub_df,
            label=label,
            date_col=date_col,
            measure_col=measure_col,
            temporal_index=temporal_index,
            imputation=imputation,
            normalization=normalization,
            reference_raw=global_trend.raw_values
        )

        if trend is None:
            continue

        score = score_trend(
            target_trend=trend,
            reference_trend=global_trend,
            method=scoring_method,
            use_normalized=True
        )

        results.append({
            "dimension": dimension,
            "value": str(val),
            "score": round(score, 4),
            "observations": count,
            "mean_measure": round(trend.mean_val, 2),
            "std_measure": round(trend.std_val, 2),
            "sparkline": render_ascii_sparkline(trend.raw_values, length=16),
            "trend": trend,
        })

    # Sort descending by outlier score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def find_top_k_unusual_trends(
    df: pd.DataFrame,
    dimension_cols: List[str],
    global_trend: Trend,
    date_col: str,
    measure_col: str,
    temporal_index: pd.DatetimeIndex,
    k: int = 10,
    scoring_method: str = "euclidean",
    normalization: str = "zscore",
    imputation: str = "interpolate",
    min_support: int = 30,
    max_values_per_dim: int = 15
) -> List[Dict[str, Any]]:
    """
    Finds top-K most unusual trends across single and pairwise dimension combinations.
    """
    candidates = []

    # 1-D trends
    for dim in dimension_cols:
        dim_results = explore_dimension(
            df=df,
            dimension=dim,
            global_trend=global_trend,
            date_col=date_col,
            measure_col=measure_col,
            temporal_index=temporal_index,
            scoring_method=scoring_method,
            normalization=normalization,
            imputation=imputation,
            min_support=min_support,
            top_n=max_values_per_dim
        )
        for r in dim_results:
            candidates.append({
                "depth": 1,
                "label": f"{r['dimension']} = {r['value']}",
                "dimensions": {r['dimension']: r['value']},
                "score": r["score"],
                "observations": r["observations"],
                "mean_measure": r["mean_measure"],
                "sparkline": r["sparkline"],
                "trend": r["trend"]
            })

    # 2-D trends (pairwise)
    if len(dimension_cols) >= 2:
        for i in range(len(dimension_cols)):
            for j in range(i + 1, len(dimension_cols)):
                dim1, dim2 = dimension_cols[i], dimension_cols[j]
                vc1 = df[dim1].value_counts()
                top_v1 = vc1[vc1 >= min_support].index[:8]

                for v1 in top_v1:
                    sub1 = df[df[dim1].astype(str) == str(v1)]
                    vc2 = sub1[dim2].value_counts()
                    top_v2 = vc2[vc2 >= min_support].index[:8]

                    for v2 in top_v2:
                        sub2 = sub1[sub1[dim2].astype(str) == str(v2)]
                        if len(sub2) < min_support:
                            continue

                        lbl = f"{dim1}={v1} & {dim2}={v2}"
                        tr = compute_group_trend(
                            sub_df=sub2,
                            label=lbl,
                            date_col=date_col,
                            measure_col=measure_col,
                            temporal_index=temporal_index,
                            imputation=imputation,
                            normalization=normalization,
                            reference_raw=global_trend.raw_values
                        )
                        if tr is None:
                            continue

                        sc = score_trend(
                            target_trend=tr,
                            reference_trend=global_trend,
                            method=scoring_method,
                            use_normalized=True
                        )

                        candidates.append({
                            "depth": 2,
                            "label": f"{dim1}={v1} & {dim2}={v2}",
                            "dimensions": {dim1: str(v1), dim2: str(v2)},
                            "score": round(sc, 4),
                            "observations": len(sub2),
                            "mean_measure": round(tr.mean_val, 2),
                            "sparkline": render_ascii_sparkline(tr.raw_values, length=16),
                            "trend": tr
                        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:k]
