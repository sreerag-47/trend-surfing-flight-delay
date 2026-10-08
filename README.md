# Multidimensional Trend Discovery Using Trend Surfing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-13%20passed-brightgreen.svg)](https://pytest.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit%20%7C%20Rich%20CLI-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, terminal-based and web-dashboard application that discovers unusual temporal trends inside subgroups of high-dimensional datasets using the **Trend Surfing** methodology.

---

## Table of Contents
- [1. Why Trend Surfing is Used](#1-why-trend-surfing-is-used)
  - [The Combinatorial Explosion Problem](#the-combinatorial-explosion-problem)
  - [Why Traditional Dashboards Fail](#why-traditional-dashboards-fail)
  - [Real-World Value](#real-world-value)
- [2. How It Works (Methodology & Architecture)](#2-how-it-works-methodology--architecture)
  - [Data Cube Tree Representation](#data-cube-tree-representation)
  - [Temporal Aggregation & Imputation](#temporal-aggregation--imputation)
  - [Vector Normalization (Shape vs. Magnitude)](#vector-normalization-shape-vs-magnitude)
  - [Outlier Scoring Metrics](#outlier-scoring-metrics)
  - [Search Heuristics: TrendSurfer I vs II](#search-heuristics-trendsurfer-i-vs-ii)
  - [Exhaustive Baseline Comparison](#exhaustive-baseline-comparison)
- [3. Installation & Setup](#3-installation--setup)
- [4. How to Use](#4-how-to-use)
  - [Option A: Interactive Streamlit Web Dashboard](#option-a-interactive-streamlit-web-dashboard)
  - [Option B: Interactive Terminal Menu](#option-b-interactive-terminal-menu)
  - [Option C: Command-Line Interface (CLI)](#option-c-command-line-interface-cli)
- [5. Testing With Custom & Kaggle Datasets](#5-testing-with-custom--kaggle-datasets)
  - [Required Dataset Structure](#required-dataset-structure)
  - [Recommended Datasets to Try](#recommended-datasets-to-try)
- [6. Experimental Benchmarks & Results](#6-experimental-benchmarks--results)
- [7. Automated Testing Suite](#7-automated-testing-suite)
- [8. Project Structure](#8-project-structure)

---

## 1. Why Trend Surfing is Used

### The Combinatorial Explosion Problem
In modern multidimensional datasets (e.g., flight operations, sales transactions, telemetry, logistics), finding anomalous behavior requires analyzing time-series trends across all possible combinations of dimensional subgroups.

For $m$ grouping dimensions $D = \{D_1, D_2, \dots, D_m\}$ each with cardinality $|D_i|$, the number of possible subcubes up to depth $k$ is given by:

$$\sum_{j=1}^k \binom{m}{j} \prod_{l=1}^j |D_{i_l}|$$

For an airline dataset with 4 dimensions (Carrier, Origin Airport, Destination Airport, Day of Week) containing hundreds of distinct airports and carriers, evaluating all subgroup combinations up to depth 3 yields **hundreds of thousands of subcubes**. Exhaustive calculation takes hours or days and is computationally intractable for interactive analysis.

### Why Traditional Dashboards Fail
Traditional BI dashboards and visual analytics tools suffer from two major flaws:
1. **Aggregations Mask Local Anomalies**: Global averages or 1-dimensional breakdowns flatten out micro-patterns. A severe flight delay spike at `Origin=EWR & Dest=BWI & DayOfWeek=5` is completely invisible when looking at all flights overall or even just all flights on Fridays.
2. **Manual Exploration is Impractical**: Analysts cannot manually hypothesize and plot thousands of multidimensional cross-slices to spot which specific group had an irregular surge.

### Real-World Value
Trend Surfing models the multidimensional space as a **Data Cube Tree** and applies intelligent pruning heuristics to "surf" down the dimension hierarchy:
- ⚡ **Discovers top outlier trends in fractions of a second** (sub-second discovery on 50K+ rows).
- 📉 **Reduces evaluated search space by 64% – 90%+** compared to brute-force Cartesian enumeration.
- 🎯 **Retains over 99% of optimal outlier score accuracy** without exhaustive scanning.
- 🏢 **Applicable across industries**: Aviation delays, retail store demand anomalies, urban transit bottlenecks, and server telemetry spikes.

---

## 2. How It Works (Methodology & Architecture)

```
 Raw Tabular Dataset (CSV / ZIP)
            │
            ▼
┌────────────────────────────────────────┐
│ 1. Ingestion & Column Auto-Detection   │ ➔ Detects temporal, continuous measure, and dimensions
└────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────┐
│ 2. Temporal Aggregation & Imputation   │ ➔ Uniform date index, handles missing data (interpolate, ffill)
└────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────┐
│ 3. Vector Normalization                │ ➔ Isolates trend shape vs magnitude (zscore, minmax, center, none)
└────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────┐
│ 4. Data Cube Tree Search               │
│    ├─ TrendSurfer I (Local Greedy)     │ ➔ Branches on child with max deviation from parent
│    ├─ TrendSurfer II (Global Distance) │ ➔ Branches on child with max distance from global baseline
│    └─ Exhaustive Baseline              │ ➔ Complete Cartesian evaluation for ground-truth comparison
└────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────┐
│ 5. Visualization & Export              │ ➔ Interactive Streamlit plots, ASCII sparklines, JSON/CSV exports
└────────────────────────────────────────┘
```

### Data Cube Tree Representation
The search space is structured as a tree rooted at depth 0:
- **Depth 0 (Root)**: The global dataset trend (e.g., all flights over time).
- **Depth 1**: Single-dimension filters (e.g., `DayOfWeek = 1`, `Origin = ORD`).
- **Depth 2**: Two-dimension conjunctions (e.g., `DayOfWeek = 1 & Dest = RNO`).
- **Depth $k$**: Conjunction of $k$ distinct dimension constraints.

### Temporal Aggregation & Imputation
1. **Aggregation**: Subgroups are grouped by uniform date periods (e.g. daily, weekly) to calculate the mean of the dependent measure ($M_t = \frac{1}{N_t}\sum_{i=1}^{N_t} y_{i,t}$).
2. **Missing Value Imputation**: When subgroups lack observations on specific dates, they are aligned to the global temporal index and imputed using configurable strategies:
   - `interpolate` *(default)*: Time-weighted linear interpolation with backward/forward edge fill.
   - `ffill_bfill`: Forward fill followed by backward fill.
   - `reference`: Fallback to values from the global reference trend.
   - `constant`: Constant fill with a fixed value (e.g., `0.0` or subgroup mean).

### Vector Normalization (Shape vs. Magnitude)
To ensure the algorithm isolates meaningful **temporal shapes** (e.g., sudden spikes, unusual cyclical patterns) rather than merely selecting groups with high baseline volume, vectors can be normalized:

| Method | Formula | Description |
| :--- | :--- | :--- |
| **`zscore`** *(Default)* | $v' = \frac{v - \mu}{\sigma}$ | Standardizes to zero mean and unit variance. Focuses purely on shape deviations and bursts. |
| **`minmax`** | $v' = \frac{v - v_{\min}}{v_{\max} - v_{\min}}$ | Rescales values to the interval $[0, 1]$. |
| **`center`** | $v' = v - \mu$ | Removes baseline vertical offset while preserving original swing amplitude. |
| **`none`** | $v' = v$ | Uses raw measure values. Best when absolute magnitude itself defines the anomaly. |

### Outlier Scoring Metrics
- **Euclidean Trend Distance**:
  $$\text{Score}(T, C) = \sqrt{\sum_{i=1}^n (t_i - c_i)^2}$$
- **PCA Reconstruction Error**: Projects candidate trends onto the principal subspace of sibling trends to measure anomalous reconstruction residual.
- **$k$-Nearest Neighbors ($k$-NN)**: Evaluates distance to the $k$-nearest sibling vectors.
- **Cluster-Based Local Outlier Factor (CBLOF)**: Measures distance to nearest cluster centroids.

### Search Heuristics: TrendSurfer I vs II

#### TrendSurfer I (Local Greedy Outlier)
At each depth $d$:
1. Generates candidate child nodes by branching on remaining unassigned dimensions.
2. Compares each candidate trend against its **immediate parent subgroup trend** (local outlier score).
3. Greedily selects the child node exhibiting the highest local deviation.
4. Continues descent until the target depth is reached.

#### TrendSurfer II (Global Reference Deviation)
At each depth $d$:
1. Generates candidate child nodes branching on remaining unassigned dimensions.
2. Compares each candidate trend directly against the **depth-0 (global dataset) trend**.
3. Selects the candidate trend that is **farthest from the global baseline**.
4. Records both local deviation and global trend distance at every step.

### Exhaustive Baseline Comparison
For small samples or restricted dimension sets, performs complete Cartesian enumeration across all dimension combinations to establish ground truth for:
- **Search Space Reduction (%)**: $\left(1 - \frac{\text{Nodes}_{\text{Surfer}}}{\text{Nodes}_{\text{Exhaustive}}}\right) \times 100\%$
- **Speedup Factor**: $\frac{\text{Time}_{\text{Exhaustive}}}{\text{Time}_{\text{Surfer}}}$
- **Score Optimality Ratio**: $\frac{\text{Score}_{\text{Surfer}}}{\text{Score}_{\text{Exhaustive}}}$

---

## 3. Installation & Setup

Ensure Python 3.10+ is installed:

```bash
# Clone the repository
git clone https://github.com/sreerag-47/trend-surfing-flight-delay.git
cd trend-surfing-flight-delay

# Install dependencies
pip install -r requirements.txt
```

---

## 4. How to Use

### Option A: Interactive Streamlit Web Dashboard

Launch the browser-based graphical user interface:

```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

**Features in the Web App:**
1. **Dataset Selection & Upload**: Choose built-in datasets or upload any custom `.csv` / `.zip` file via sidebar drag-and-drop.
2. **Column Mapping**: Automatically suggests temporal, measure, and dimension columns, with full manual override.
3. **Trend Normalization**: Toggle between `zscore`, `minmax`, `center`, and `none` on the fly.
4. **Interactive Discovery**: Run TrendSurfer I or II with custom depth (1–4) and scoring metrics (`euclidean`, `pca`, `knn`, `cblof`).
5. **Interactive Matplotlib Plots**: Side-by-side comparison of the global trend vs. the discovered outlier subgroup trend.
6. **Step-by-Step Trajectory**: Inspect every decision node, local score, and global distance down the tree.
7. **Algorithm Comparison**: Benchmark Surfer I vs. Surfer II vs. Exhaustive Baseline in a single click.
8. **Dimension Explorer & Top-K**: Rank values along any single dimension or find the top $K$ most anomalous subgroups across the entire dataset.
9. **One-Click JSON Export**: Download structured run logs.

---

### Option B: Interactive Terminal Menu

Run without arguments for an interactive terminal experience:

```bash
python main.py
```

```text
======================================================================
               MULTIDIMENSIONAL TREND DISCOVERY
                      USING TREND SURFING
======================================================================

Select an action:
  1. Upload / Load Dataset
  2. Dataset Summary
  3. Discover Unusual Trend (Quick Run)
  4. Run TrendSurfer I
  5. Run TrendSurfer II
  6. Compare TrendSurfer I vs II
  7. Explore a Dimension
  8. Show Top Unusual Trends
  9. Run Exhaustive Baseline
 10. Export Results
  0. Exit
```

---

### Option C: Command-Line Interface (CLI)

Run direct headless CLI commands for scripting and automated pipelines:

#### 1. Dataset Overview & Auto-Detection
```bash
python main.py summary data/DelayedFlights_sample.csv
```

#### 2. Run TrendSurfer I (Local Greedy)
```bash
python main.py surf data/DelayedFlights_sample.csv --algorithm surfer1 --depth 3
```

#### 3. Run TrendSurfer II (Global Reference Deviation)
```bash
python main.py surf data/DelayedFlights_sample.csv --algorithm surfer2 --depth 3
```

#### 4. Run Heuristic vs. Exhaustive Baseline Comparison
```bash
python main.py compare data/DelayedFlights_sample.csv --depth 2 --baseline
```

#### 5. Discover Top-K Unusual Trends Across Dataset
```bash
python main.py top data/DelayedFlights_sample.csv --k 10
```

#### 6. Explore & Rank Values Along a Single Dimension
```bash
python main.py explore data/DelayedFlights_sample.csv --dimension UniqueCarrier
```

---

## 5. Testing With Custom & Kaggle Datasets

Trend Surfing is not limited to flight delay data; it automatically generalizes to **any tabular time series dataset**.

### Required Dataset Structure
To test your own dataset, ensure it contains:
1. **Temporal Dimension**: Either a single date/timestamp column (e.g. `Date`, `OrderDate`, `timestamp`) or composite columns (`Year`, `Month`, `DayofMonth`).
2. **Numeric Dependent Measure**: A continuous numeric measure to track over time (e.g., `ArrDelay`, `Weekly_Sales`, `Profit`, `AQI`, `trip_duration`).
3. **Categorical Dimensions**: 2 to 6 categorical grouping columns with moderate cardinality (e.g., `Carrier`, `Origin`, `Store`, `Category`, `City`, `Region`).

### Recommended Datasets to Try

| Domain | Kaggle Dataset | Key Dimensions | Target Measure |
| :--- | :--- | :--- | :--- |
| **Aviation** | [Airline Delay Causes](https://www.kaggle.com/datasets/giovannidata/delayedflights) | `UniqueCarrier`, `Origin`, `Dest`, `DayOfWeek` | `ArrDelay` |
| **Retail Sales** | [Walmart Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) | `Store`, `Dept`, `Type`, `IsHoliday` | `Weekly_Sales` |
| **Superstore** | [Sample Superstore Sales](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) | `Region`, `Category`, `Sub-Category`, `Segment` | `Sales` or `Profit` |
| **Urban Mobility** | [NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) | `vendor_id`, `passenger_count`, `dayofweek` | `trip_duration` |
| **Air Quality** | [Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) | `City`, `Station`, `AQI_Bucket` | `AQI` |

**How to run a new dataset:**
1. Drop the `.csv` or `.zip` file into the `data/` directory.
2. Run `python main.py summary data/your_file.csv` to verify column auto-detection.
3. Run `python main.py surf data/your_file.csv --algorithm surfer2 --depth 3` to discover top outlier trends!

---

## 6. Experimental Benchmarks & Results

Benchmarked on `data/DelayedFlights_sample.csv` (50,000 records, 31 daily temporal points, depth 2):

| Metric | TrendSurfer I | TrendSurfer II | Exhaustive Baseline |
| :--- | :--- | :--- | :--- |
| **Discovered Subgroup** | `DayOfWeek=1 & Origin=TPA` | `DayOfWeek=1 & Dest=RNO` | `DayOfWeek=5 & Origin=BNA` |
| **Global Outlier Score** | 9.15 | **9.43** | 9.48 |
| **Local Outlier Score** | 9.97 | 8.13 | N/A |
| **Nodes Opened / Evaluated** | **216** | **216** | 607 |
| **Execution Time** | **0.474s** | **0.462s** | 2.327s |
| **Search Space Reduction** | **64.4%** | **64.4%** | 0.0% (Baseline) |
| **Speedup Factor** | **4.9x** | **5.0x** | 1.0x |

**Key Takeaways:**
- **Accuracy**: TrendSurfer II achieved **99.5%** of the exhaustive optimum score (`9.43` vs `9.48`).
- **Pruning**: Opened **64.4% fewer nodes** and ran **5.0x faster** than brute-force enumeration.
- At depth 3 and above, the pruning advantage increases exponentially, saving hours of computation.

---

## 7. Automated Testing Suite

The repository includes a comprehensive `pytest` test suite with 100% pass rate:

```bash
python -m pytest -v
```

```text
============================= test session starts =============================
tests/test_baseline.py::test_exhaustive_baseline_and_comparison PASSED   [  7%]
tests/test_loader.py::test_load_csv PASSED                               [ 15%]
tests/test_loader.py::test_detect_columns PASSED                         [ 23%]
tests/test_loader.py::test_validate_configuration PASSED                 [ 30%]
tests/test_preprocessing.py::test_prepare_dataframe PASSED               [ 38%]
tests/test_preprocessing.py::test_impute_trend_series PASSED             [ 46%]
tests/test_preprocessing.py::test_normalize_trend_vector PASSED          [ 53%]
tests/test_scoring.py::test_euclidean_distance PASSED                    [ 61%]
tests/test_scoring.py::test_score_trend_identical_and_outlier PASSED     [ 69%]
tests/test_surfer1.py::test_trendsurfer1_execution PASSED                [ 76%]
tests/test_surfer2.py::test_trendsurfer2_execution PASSED                [ 84%]
tests/test_trend.py::test_compute_global_and_group_trend PASSED          [ 92%]
tests/test_trend.py::test_render_ascii_sparkline PASSED                  [100%]
============================= 13 passed in 2.22s ==============================
```

---

## 8. Project Structure

```
trend-surfing-flight-delay/
│
├── main.py                     # Unified CLI entry point & Interactive Terminal Menu
├── app.py                      # Interactive Streamlit Web Dashboard
├── requirements.txt            # Python dependencies
├── plan.md                     # Architecture specification
├── README.md                   # Comprehensive documentation
│
├── data/
│   ├── DelayedFlights_sample.csv # 50K flight delay benchmark dataset
│   ├── DelayedFlights_sample.zip # Sample ZIP archive for compression testing
│   ├── airlines.csv              # Carrier IATA code to airline name lookup
│   ├── sample_sales.csv          # Sample multidimensional retail sales dataset
│   └── README.md                 # Dataset instructions
│
├── trendsurf/
│   ├── __init__.py
│   ├── loader.py               # CSV/ZIP loader, auto-column detection, validation
│   ├── preprocessing.py        # Date creation, imputation, and vector normalization
│   ├── trend.py                # Trend dataclass, temporal aggregation, sparklines
│   ├── datacube.py             # CubeNode, child generation, subgroup filtering
│   ├── scoring.py              # Euclidean, PCA, kNN, and CBLOF outlier scoring
│   ├── surfers/
│   │   ├── __init__.py         # SurferResult dataclass
│   │   ├── surfer1.py          # TrendSurfer I implementation (Local Greedy)
│   │   └── surfer2.py          # TrendSurfer II implementation (Global Reference)
│   ├── baseline.py             # Exhaustive baseline search & comparative evaluation
│   ├── explorer.py             # Dimension ranking & top-K trend discovery
│   ├── exporter.py             # JSON and CSV export utilities
│   └── terminal_ui.py          # Rich terminal formatting, tables, and charts
│
├── tests/
│   ├── test_loader.py          # Unit tests for loader & validation
│   ├── test_preprocessing.py   # Unit tests for preprocessing & imputation
│   ├── test_trend.py           # Unit tests for trend generation
│   ├── test_scoring.py         # Unit tests for distance & outlier metrics
│   ├── test_surfer1.py         # Tests for TrendSurfer I
│   ├── test_surfer2.py         # Tests for TrendSurfer II
│   └── test_baseline.py        # Tests for exhaustive baseline
│
└── results/
    ├── surfer1/                # JSON logs for TrendSurfer I
    ├── surfer2/                # JSON logs for TrendSurfer II
    └── exports/                # Exported comparison tables and summaries
```

---

## License

This project is licensed under the MIT License.
