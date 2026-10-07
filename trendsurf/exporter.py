"""
Results Exporter Module
Exports search results and comparison summaries to JSON and CSV.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Union, Optional
import pandas as pd

from trendsurf.surfers import SurferResult
from trendsurf.baseline import BaselineResult


def export_result(
    result: Union[SurferResult, BaselineResult, Dict[str, Any]],
    output_dir: str = "results/exports",
    prefix: Optional[str] = None
) -> str:
    """
    Saves analysis results to a timestamped JSON file.
    Returns the file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if isinstance(result, SurferResult):
        data = result.to_dict()
        algo_name = result.algorithm.lower().replace(" ", "")
        default_prefix = algo_name
    elif isinstance(result, BaselineResult):
        data = result.to_dict()
        default_prefix = "baseline"
    else:
        data = result
        default_prefix = "experiment"

    pref = prefix or default_prefix
    filename = f"{pref}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    data["exported_at"] = datetime.now().isoformat()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return filepath


def export_comparison_table(
    comparison_data: Dict[str, Any],
    output_path: str = "results/exports/comparison_summary.csv"
) -> str:
    """
    Appends or creates a CSV comparison record.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    row_df = pd.DataFrame([comparison_data])
    row_df["timestamp"] = datetime.now().isoformat()

    if os.path.exists(output_path):
        row_df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(output_path, index=False)

    return output_path
