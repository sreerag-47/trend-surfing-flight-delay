"""
Dataset Loader & Validation Module
Handles loading CSV / ZIP archives, automatic column detection,
and configuration validation.
"""

import os
import zipfile
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class DatasetConfig:
    filepath: str
    temporal_cols: List[str]
    measure_col: str
    dimension_cols: List[str]
    date_col_name: str = "Date"
    sample_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filepath": self.filepath,
            "temporal_cols": self.temporal_cols,
            "measure_col": self.measure_col,
            "dimension_cols": self.dimension_cols,
            "date_col_name": self.date_col_name,
            "sample_size": self.sample_size,
        }


def load_dataset(filepath: str, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    """
    Loads dataset from a CSV file or a ZIP archive containing a CSV file.
    Returns (DataFrame, resolved_filename).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(filepath, nrows=nrows, low_memory=False)
        return df, os.path.basename(filepath)

    elif ext == ".zip":
        with zipfile.ZipFile(filepath, "r") as zf:
            csv_files = [f for f in zf.namelist() if f.lower().endswith(".csv") and not f.startswith("__MACOSX")]
            if not csv_files:
                raise ValueError(f"No CSV file found inside archive: {filepath}")
            target_csv = csv_files[0]
            with zf.open(target_csv) as f:
                df = pd.read_csv(f, nrows=nrows, low_memory=False)
            return df, target_csv
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Only .csv and .zip files are supported.")


def detect_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes dataframe and suggests suitable temporal columns, numeric measures,
    and categorical grouping dimensions.
    """
    cols = df.columns.tolist()
    col_lower = {c: c.lower() for c in cols}

    # 1. Temporal column detection
    temporal_candidates: List[List[str]] = []
    # Check composite: Year, Month, Day/DayofMonth
    year_col = next((c for c in cols if col_lower[c] in ["year", "yr"]), None)
    month_col = next((c for c in cols if col_lower[c] in ["month", "mo"]), None)
    day_col = next((c for c in cols if col_lower[c] in ["dayofmonth", "day", "dom", "day_of_month"]), None)

    if year_col and month_col and day_col:
        temporal_candidates.append([year_col, month_col, day_col])

    # Check single date / timestamp columns
    for c in cols:
        cl = col_lower[c]
        if any(term in cl for term in ["date", "timestamp", "datetime", "flightdate"]) and [c] not in temporal_candidates:
            temporal_candidates.append([c])

    default_temporal = temporal_candidates[0] if temporal_candidates else ([cols[0]] if cols else [])

    # 2. Numeric measure detection
    known_measure_names = [
        "arrdelay", "depdelay", "delay", "distance", "airtime", "actualelapsedtime",
        "sales", "revenue", "amount", "profit", "value", "price", "count", "metric"
    ]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    
    preferred_measures = ["arrdelay", "delay", "depdelay", "distance", "sales", "revenue", "amount"]
    def measure_rank(col):
        cl = col_lower[col]
        if cl == "arrdelay":
            return -10
        if cl in preferred_measures:
            return preferred_measures.index(cl)
        if any(km in cl for km in preferred_measures):
            return 10
        if cl in known_measure_names:
            return 20
        if cl not in ["unnamed: 0", "id", "flightnum", "year", "month", "dayofmonth", "dayofweek"]:
            return 30
        return 100

    measure_candidates = sorted(numeric_cols, key=measure_rank)
    # Filter out pure IDs or year/month if other options exist
    usable_measures = [
        c for c in measure_candidates 
        if col_lower[c] not in ["unnamed: 0", "id", "year", "month", "dayofmonth"]
    ]
    default_measure = usable_measures[0] if usable_measures else (numeric_cols[0] if numeric_cols else "")

    # 3. Categorical grouping dimensions detection
    known_dim_names = [
        "uniquecarrier", "carrier", "airline", "origin", "dest", "destination",
        "dayofweek", "region", "city", "state", "category", "product", "segment",
        "tailnum", "flightnum", "cancellationcode"
    ]
    dim_candidates = []
    for c in cols:
        cl = col_lower[c]
        # Skip index column
        if cl in ["unnamed: 0", "index", "id"]:
            continue
        # If string / categorical or low cardinality numeric
        nunique = df[c].nunique(dropna=True)
        is_obj = pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)
        is_discrete = pd.api.types.is_integer_dtype(df[c]) and (nunique <= 50)

        if is_obj or is_discrete or cl in known_dim_names:
            if c != default_measure and c not in (default_temporal if isinstance(default_temporal, list) else [default_temporal]):
                dim_candidates.append(c)

    # Sort dimension candidates to put primary ones first
    dim_candidates = sorted(
        dim_candidates,
        key=lambda c: 0 if col_lower[c] in known_dim_names else 1
    )

    # Preferred default dimensions for flight delays if present
    preferred_defaults = ["UniqueCarrier", "Origin", "Dest", "DayOfWeek"]
    default_dims = [c for c in preferred_defaults if c in cols]
    if not default_dims:
        default_dims = dim_candidates[:4]

    return {
        "all_columns": cols,
        "temporal_candidates": temporal_candidates,
        "default_temporal": default_temporal,
        "measure_candidates": usable_measures,
        "default_measure": default_measure,
        "dimension_candidates": dim_candidates,
        "default_dimensions": default_dims,
    }


def validate_configuration(
    df: pd.DataFrame,
    temporal_cols: List[str],
    measure_col: str,
    dimension_cols: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Validates dataset against user configuration.
    Returns (is_valid, error_message).
    """
    if df.empty:
        return False, "Dataset is empty."

    if not temporal_cols:
        return False, "No temporal columns specified."
    for col in temporal_cols:
        if col not in df.columns:
            return False, f"Temporal column '{col}' does not exist in dataset."

    if not measure_col:
        return False, "No numeric dependent measure specified."
    if measure_col not in df.columns:
        return False, f"Measure column '{measure_col}' does not exist in dataset."
    if not pd.api.types.is_numeric_dtype(df[measure_col]):
        return False, f"Measure column '{measure_col}' is not numeric."
    if df[measure_col].dropna().empty:
        return False, f"Measure column '{measure_col}' contains no valid numeric values."

    if not dimension_cols:
        return False, "At least one grouping dimension must be selected."
    for col in dimension_cols:
        if col not in df.columns:
            return False, f"Grouping dimension '{col}' does not exist in dataset."

    return True, None


def get_dataset_summary(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    """
    Generates a structured overview of the dataset.
    """
    missing_counts = df.isnull().sum()
    missing_dict = {col: int(cnt) for col, cnt in missing_counts.items() if cnt > 0}

    col_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "filename": filename,
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "column_types": col_types,
        "missing_counts": missing_dict,
        "sample_preview": df.head(3).to_dict(orient="records"),
    }
