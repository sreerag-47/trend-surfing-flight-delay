"""
Trend Surfer algorithms package.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from trendsurf.datacube import CubeNode
from trendsurf.trend import Trend


@dataclass
class SurferResult:
    algorithm: str
    target_depth: int
    path: List[CubeNode]
    nodes_opened: int
    execution_time: float
    scoring_method: str
    normalization_method: str
    global_trend: Trend
    measure_col: str
    dimension_cols: List[str]

    @property
    def final_node(self) -> CubeNode:
        return self.path[-1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "target_depth": self.target_depth,
            "measure": self.measure_col,
            "dimensions": self.dimension_cols,
            "scoring_method": self.scoring_method,
            "normalization_method": self.normalization_method,
            "nodes_opened": self.nodes_opened,
            "execution_time_seconds": round(self.execution_time, 4),
            "selected_path": [
                {
                    "depth": node.depth,
                    "node_id": node.node_id,
                    "dimensions": node.dimensions,
                    "local_score": round(node.local_score, 4),
                    "global_score": round(node.global_score, 4),
                    "observations": node.observation_count,
                }
                for node in self.path
            ],
            "final_node": {
                "node_id": self.final_node.node_id,
                "dimensions": self.final_node.dimensions,
                "local_score": round(self.final_node.local_score, 4),
                "global_score": round(self.final_node.global_score, 4),
                "observations": self.final_node.observation_count,
            }
        }
