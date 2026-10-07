"""
TrendSurfer I Algorithm Implementation
Greedy local outlier child selection heuristic.
"""

import time
from typing import List, Optional
import pandas as pd

from trendsurf.datacube import CubeNode, generate_child_nodes, filter_subgroup
from trendsurf.trend import Trend
from trendsurf.scoring import score_trend
from trendsurf.surfers import SurferResult


def run_trendsurfer1(
    df: pd.DataFrame,
    global_trend: Trend,
    dimension_cols: List[str],
    date_col: str,
    measure_col: str,
    temporal_index: pd.DatetimeIndex,
    target_depth: int = 3,
    scoring_method: str = "euclidean",
    normalization: str = "zscore",
    imputation: str = "interpolate",
    min_support: int = 20,
    top_values_per_dim: Optional[int] = 50
) -> SurferResult:
    """
    Executes TrendSurfer I heuristic search.
    At each depth, evaluates candidate children against the current node (local outlier),
    and picks the highest local outlier score.
    """
    start_time = time.perf_counter()

    root_node = CubeNode(
        depth=0,
        dimensions={},
        trend=global_trend,
        local_score=0.0,
        global_score=0.0,
        observation_count=len(df),
        parent_id=None
    )

    path: List[CubeNode] = [root_node]
    current_node = root_node
    current_df = df
    nodes_opened = 1

    max_possible_depth = min(target_depth, len(dimension_cols))

    for depth in range(1, max_possible_depth + 1):
        children = generate_child_nodes(
            parent_node=current_node,
            sub_df=current_df,
            dimension_cols=dimension_cols,
            date_col=date_col,
            measure_col=measure_col,
            temporal_index=temporal_index,
            imputation=imputation,
            normalization=normalization,
            min_support=min_support,
            top_values_per_dim=top_values_per_dim
        )

        if not children:
            break

        nodes_opened += len(children)

        # Prepare sibling vectors if PCA/kNN/CBLOF is requested
        sibling_vectors = [c.trend.normalized_values for c in children if c.trend]

        best_child: Optional[CubeNode] = None
        best_score = -1.0

        for child in children:
            if child.trend is None:
                continue

            # TrendSurfer I: local outlier score compared to immediate parent trend
            local_sc = score_trend(
                target_trend=child.trend,
                reference_trend=current_node.trend,
                method=scoring_method,
                sibling_vectors=sibling_vectors,
                use_normalized=True
            )
            child.local_score = local_sc

            # Record global distance as well
            global_sc = score_trend(
                target_trend=child.trend,
                reference_trend=global_trend,
                method="euclidean",
                use_normalized=True
            )
            child.global_score = global_sc

            if local_sc > best_score:
                best_score = local_sc
                best_child = child

        if best_child is None:
            break

        path.append(best_child)
        current_node = best_child
        current_df = filter_subgroup(df, current_node.dimensions)

    elapsed_time = time.perf_counter() - start_time

    return SurferResult(
        algorithm="TrendSurfer I",
        target_depth=target_depth,
        path=path,
        nodes_opened=nodes_opened,
        execution_time=elapsed_time,
        scoring_method=scoring_method,
        normalization_method=normalization,
        global_trend=global_trend,
        measure_col=measure_col,
        dimension_cols=dimension_cols
    )
