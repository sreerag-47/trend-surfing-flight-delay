"""
Exhaustive Baseline Search & Comparative Evaluation Module
Implements exhaustive search across dimension combinations and comparison metrics.
"""

import time
import itertools
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from trendsurf.datacube import CubeNode, filter_subgroup
from trendsurf.trend import Trend, compute_group_trend
from trendsurf.scoring import score_trend
from trendsurf.surfers import SurferResult


@dataclass
class BaselineResult:
    target_depth: int
    total_nodes_evaluated: int
    execution_time: float
    top_node: CubeNode
    all_evaluated_nodes: List[CubeNode]
    measure_col: str
    dimension_cols: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_depth": self.target_depth,
            "total_nodes_evaluated": self.total_nodes_evaluated,
            "execution_time_seconds": round(self.execution_time, 4),
            "top_node": {
                "node_id": self.top_node.node_id,
                "dimensions": self.top_node.dimensions,
                "global_score": round(self.top_node.global_score, 4),
                "observations": self.top_node.observation_count,
            },
            "top_5_nodes": [
                {
                    "node_id": n.node_id,
                    "dimensions": n.dimensions,
                    "global_score": round(n.global_score, 4),
                    "depth": n.depth,
                    "observations": n.observation_count,
                }
                for n in sorted(self.all_evaluated_nodes, key=lambda x: x.global_score, reverse=True)[:5]
            ]
        }


def run_exhaustive_search(
    df: pd.DataFrame,
    global_trend: Trend,
    dimension_cols: List[str],
    date_col: str,
    measure_col: str,
    temporal_index: pd.DatetimeIndex,
    target_depth: int = 2,
    scoring_method: str = "euclidean",
    normalization: str = "zscore",
    imputation: str = "interpolate",
    min_support: int = 20,
    max_values_per_dim: int = 20
) -> BaselineResult:
    """
    Exhaustively searches all dimension combinations up to target_depth.
    Designed for small datasets or restricted dimension subspaces to establish ground truth.
    """
    start_time = time.perf_counter()
    evaluated_nodes: List[CubeNode] = []

    # Get valid values for each dimension
    dim_valid_values: Dict[str, List[str]] = {}
    for dim in dimension_cols:
        vc = df[dim].value_counts()
        valid = vc[vc >= min_support].index[:max_values_per_dim].tolist()
        dim_valid_values[dim] = [str(v) for v in valid]

    # Evaluate depth 1 up to target_depth
    for d in range(1, target_depth + 1):
        for dim_combo in itertools.combinations(dimension_cols, d):
            # Cartesian product of valid values for this combination of dimensions
            val_pools = [dim_valid_values[dim] for dim in dim_combo]
            for val_combo in itertools.product(*val_pools):
                dim_dict = {dim_combo[i]: val_combo[i] for i in range(d)}
                sub_df = filter_subgroup(df, dim_dict)
                if len(sub_df) < min_support:
                    continue

                label = " & ".join(f"{k}={v}" for k, v in dim_dict.items())
                trend = compute_group_trend(
                    sub_df=sub_df,
                    label=label,
                    date_col=date_col,
                    measure_col=measure_col,
                    temporal_index=temporal_index,
                    imputation=imputation,
                    normalization=normalization
                )

                if trend is None:
                    continue

                global_sc = score_trend(
                    target_trend=trend,
                    reference_trend=global_trend,
                    method=scoring_method,
                    use_normalized=True
                )

                node = CubeNode(
                    depth=d,
                    dimensions=dim_dict,
                    trend=trend,
                    global_score=global_sc,
                    observation_count=len(sub_df)
                )
                evaluated_nodes.append(node)

    elapsed_time = time.perf_counter() - start_time

    if not evaluated_nodes:
        # Fallback dummy root if no candidates met support
        top_node = CubeNode(depth=0, dimensions={}, trend=global_trend, global_score=0.0)
    else:
        top_node = max(evaluated_nodes, key=lambda n: n.global_score)

    return BaselineResult(
        target_depth=target_depth,
        total_nodes_evaluated=len(evaluated_nodes),
        execution_time=elapsed_time,
        top_node=top_node,
        all_evaluated_nodes=evaluated_nodes,
        measure_col=measure_col,
        dimension_cols=dimension_cols
    )


def compare_surfer_with_baseline(
    surfer_res: SurferResult,
    baseline_res: BaselineResult
) -> Dict[str, Any]:
    """
    Computes comparative metrics between a heuristic surfer and exhaustive baseline.
    """
    ex_nodes = baseline_res.total_nodes_evaluated
    sf_nodes = surfer_res.nodes_opened

    search_reduction_pct = ((1.0 - (sf_nodes / ex_nodes)) * 100.0) if ex_nodes > 0 else 0.0
    speedup = (baseline_res.execution_time / surfer_res.execution_time) if surfer_res.execution_time > 0 else 1.0

    sf_final_score = surfer_res.final_node.global_score
    ex_best_score = baseline_res.top_node.global_score

    # Calculate percentile / score ratio
    score_ratio = (sf_final_score / ex_best_score) if ex_best_score > 0 else 1.0

    return {
        "algorithm": surfer_res.algorithm,
        "surfer_nodes_opened": sf_nodes,
        "exhaustive_nodes_evaluated": ex_nodes,
        "search_reduction_pct": round(search_reduction_pct, 2),
        "surfer_time_sec": round(surfer_res.execution_time, 4),
        "exhaustive_time_sec": round(baseline_res.execution_time, 4),
        "speedup_factor": round(speedup, 2),
        "surfer_score": round(sf_final_score, 4),
        "exhaustive_best_score": round(ex_best_score, 4),
        "score_ratio": round(score_ratio, 4),
        "discovered_dimensions": surfer_res.final_node.dimensions,
        "exhaustive_best_dimensions": baseline_res.top_node.dimensions,
    }
