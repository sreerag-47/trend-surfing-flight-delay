# Multidimensional Trend Discovery Using Trend Surfing

## 1. Project Overview
This project implements the **Trend Surfing** methodology for discovering unusual temporal trends inside subgroups of a multidimensional dataset without exhaustively exploring exponential combinations of dimensions.

Based on the research paper on discovering outlier trends in multidimensional data cubes, the system implements:
- **TrendSurfer I**: Greedy local outlier child selection.
- **TrendSurfer II**: Global trend deviation heuristic, comparing candidate trends against the depth-0 baseline.
- **Exhaustive Baseline**: Exhaustive search over all dimension combinations (for comparison on smaller cubes / samples).
- **Multiple Outlier Metrics**: Euclidean distance (MVP default), PCA Reconstruction Error, kNN Distance, and CBLOF.
- **Flexible Data Upload & Detection**: Supports CSV and ZIP files, automatic temporal/measure/dimension detection, and interactive configuration.
- **Terminal UI & Visualization**: Interactive Rich CLI menus and Plotext terminal trend charts.
- **Interactive Web Dashboard**: Streamlit app for exploratory visual discovery.

---

## 2. Architecture & Modules
- `trendsurf/loader.py`: Dataset loading (CSV/ZIP), column introspection, and validation.
- `trendsurf/preprocessing.py`: Date parsing, missing value handling, temporal indexing, normalization.
- `trendsurf/trend.py`: Time-series representation, trend extraction, vector normalization, plotting.
- `trendsurf/datacube.py`: Data cube nodes, dimension hierarchies, candidate generation.
- `trendsurf/scoring.py`: Distance metrics and outlier scorers (Euclidean, PCA, kNN, CBLOF).
- `trendsurf/surfers/surfer1.py`: TrendSurfer I heuristic search algorithm.
- `trendsurf/surfers/surfer2.py`: TrendSurfer II heuristic search algorithm.
- `trendsurf/baseline.py`: Exhaustive search baseline and comparison metrics (nodes opened, reduction %, runtime).
- `trendsurf/explorer.py`: Single dimension exploration and top-k unusual trends extraction.
- `trendsurf/exporter.py`: JSON and CSV export of discovery results.
- `trendsurf/terminal_ui.py`: Rich terminal interface and menu system.
- `main.py`: Unified CLI entry point supporting interactive mode and subcommands (`summary`, `surf`, `compare`, `top`, `baseline`, `load`).

---

## 3. Evaluation & Metrics
- **Search Efficiency**: Number of nodes opened, elapsed wall-clock execution time.
- **Outlier Quality**: Local outlier score, distance from global depth-0 trend.
- **Search Space Reduction**: `(1 - Nodes_Surfer / Nodes_Exhaustive) * 100%`.
