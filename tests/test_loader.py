import os
import pytest
import pandas as pd
from trendsurf.loader import load_dataset, detect_columns, validate_configuration, get_dataset_summary


def test_load_csv(tmp_path):
    csv_file = tmp_path / "test_flights.csv"
    csv_file.write_text("Year,Month,DayofMonth,ArrDelay,UniqueCarrier,Origin\n2008,1,1,10.5,AA,JFK\n2008,1,2,20.0,AA,JFK\n")
    df, filename = load_dataset(str(csv_file))
    assert len(df) == 2
    assert filename == "test_flights.csv"


def test_detect_columns():
    df = pd.DataFrame({
        "Year": [2008, 2008],
        "Month": [1, 1],
        "DayofMonth": [1, 2],
        "ArrDelay": [15.0, 30.0],
        "UniqueCarrier": ["WN", "AA"],
        "Origin": ["ORD", "ATL"],
        "Dest": ["DFW", "LAX"],
        "DayOfWeek": [1, 2]
    })
    detected = detect_columns(df)
    assert detected["default_temporal"] == ["Year", "Month", "DayofMonth"]
    assert detected["default_measure"] == "ArrDelay"
    assert "UniqueCarrier" in detected["default_dimensions"]


def test_validate_configuration():
    df = pd.DataFrame({
        "Year": [2008],
        "Month": [1],
        "DayofMonth": [1],
        "ArrDelay": [15.0],
        "UniqueCarrier": ["WN"]
    })
    valid, msg = validate_configuration(df, ["Year", "Month", "DayofMonth"], "ArrDelay", ["UniqueCarrier"])
    assert valid is True
    assert msg is None

    # Missing column
    invalid, err = validate_configuration(df, ["Year"], "NonExistentMeasure", ["UniqueCarrier"])
    assert invalid is False
    assert "NonExistentMeasure" in err
