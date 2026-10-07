"""
Data Cube & Search Tree Module
Defines CubeNode and child node generation logic across multidimensional combinations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
import pandas as pd
import numpy as np

from trendsurf.trend import Trend, compute_group_trend


@dataclass
class CubeNode:
    """
    Represents a node in the multidimensional data cube search space.
    """
    depth: int
    dimensions: Dict[str, str] = field(default_factory=dict)
    trend: Optional[Trend] = None
    local_score: float = 0.0
    global_score: float = 0.0
    observation_count: int = 0
    parent_id: Optional[str] = None

    @property
    def node_id(self) -> str:
        if not self.dimensions:
            return "ALL (Global)"
        # Canonical sorted string
        return " & ".join(f"{k}={v}" for k, v in sorted(self.dimensions.items()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "dimensions": self.dimensions,
            "observation_count": self.observation_count,
            "local_score": round(self.local_score, 4),
            "global_score": round(self.global_score, 4),
            "trend_summary": self.trend.to_dict() if self.trend else None,
        }


def filter_subgroup(df: pd.DataFrame, dimensions: Dict[str, str]) -> pd.DataFrame:
    """
    Filters dataframe for a specific set of dimension key-value pairs.
    """
    if not dimensions:
        return df
    mask = np.ones(len(df), dtype=bool)
    for dim, val in dimensions.items():
        mask &= (df[dim].astype(str) == str(val)).to_numpy()
    return df[mask]


def generate_child_nodes(
    parent_node: CubeNode,
    sub_df: pd.DataFrame,
    dimension_cols: List[str],
    date_col: str,
    measure_col: str,
    temporal_index: pd.DatetimeIndex,
    imputation: str = "interpolate",
    normalization: str = "zscore",
    min_support: int = 20,
    top_values_per_dim: Optional[int] = 50
) -> List[CubeNode]:
    """
    Generates all valid immediate child nodes for the given parent node.
    Each child branches on one remaining unassigned dimension.
    """
    assigned_dims = set(parent_node.dimensions.keys())
    remaining_dims = [d for d in dimension_cols if d not in assigned_dims]

    children: List[CubeNode] = []
    parent_raw = parent_node.trend.raw_values if parent_node.trend else None

    for dim in remaining_dims:
        # Get value counts in the current subgroup
        val_counts = sub_df[dim].value_counts()
        # Filter by min_support
        valid_vals = val_counts[val_counts >= min_support]
        if top_values_per_dim and len(valid_vals) > top_values_per_dim:
            valid_vals = valid_vals.iloc[:top_values_per_dim]

        for val, count in valid_vals.items():
            child_dims = dict(parent_node.dimensions)
            child_dims[dim] = str(val)

            # Filter data for this child
            child_df = sub_df[sub_df[dim].astype(str) == str(val)]
            child_label = f"{dim}={val}"

            child_trend = compute_group_trend(
                sub_df=child_df,
                label=child_label,
                date_col=date_col,
                measure_col=measure_col,
                temporal_index=temporal_index,
                imputation=imputation,
                normalization=normalization,
                reference_raw=parent_raw
            )

            if child_trend is not None:
                child_node = CubeNode(
                    depth=parent_node.depth + 1,
                    dimensions=child_dims,
                    trend=child_trend,
                    observation_count=len(child_df),
                    parent_id=parent_node.node_id
                )
                children.append(child_node)

    return children
