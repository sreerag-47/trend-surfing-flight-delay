# Multidimensional Trend Discovery Using Trend Surfing

An intelligent, terminal-based and web-dashboard application that discovers unusual temporal trends inside subgroups of multidimensional datasets using the **Trend Surfing** methodology.

---

## 1. Problem Statement & Research Background

In modern multidimensional datasets (e.g., commercial flight operations, sales, telemetry), analyzing temporal trends across all possible combinations of dimensional subgroups suffers from **combinatorial explosion**. 

For $m$ grouping dimensions $D = \{D_1, D_2, \dots, D_m\}$ each with cardinality $|D_i|$, the number of possible subcubes up to depth $k$ is given by:
$$\sum_{j=1}^k \binom{m}{j} \prod_{l=1}^j |D_{i_l}|$$

Exhaustively generating and evaluating time series for all subcubes is computationally intractable. 

**Trend Surfing** addresses this by modeling the search space as a data cube tree and using intelligent heuristics to descend through the dimension hierarchy, pruning unpromising branches and discovering top outlier trends in seconds.

---

## 2. Core Methodologies

### 2.1 Trend Representation
- **Temporal Aggregation**: Data is grouped by a time index (e.g., Date constructed from `Year + Month + DayofMonth`) to calculate the mean of the dependent measure (e.g., `ArrDelay`).
- **Missing Value Handling**: Incomplete time series are aligned with the complete dataset temporal index and imputed using configurable strategies (`interpolate`, `ffill_bfill`, `reference_trend`, `constant`).
- **Vector Normalization**: Configurable normalization (`zscore`, `minmax`, `center`, `none`) allows algorithms to focus on temporal trend shapes and deviations rather than merely absolute baseline magnitude.

### 2.2 Outlier Scoring Metrics
- **Euclidean Trend Distance**:
  $$\text{Score}(T, C) = \sqrt{\sum_{i=1}^n (t_i - c_i)^2}$$
- **PCA Reconstruction Error**: Projects candidate trends onto the principal subspace of sibling trends to measure anomalous reconstruction residual.
- **k-Nearest Neighbors (k-NN)**: Evaluates distance to $k$-nearest sibling vectors.
- **Cluster-Based Local Outlier Factor (CBLOF)**: Measures distance to nearest cluster centers.

### 2.3 Heuristic Search Algorithms

#### TrendSurfer I (Local Greedy Outlier)
At each depth $d$:
1. Generate candidate child nodes branching on remaining unassigned dimensions.
2. Evaluate each child's trend against its **immediate parent subgroup trend** (local outlier score).
3. Greedily select the child node exhibiting the highest local deviation.
4. Continue descent until target depth is reached.

#### TrendSurfer II (Global Reference Deviation)
At each depth $d$:
1. Generate candidate child nodes branching on remaining unassigned dimensions.
2. Compare each candidate trend directly against the **depth-0 (global overall) trend**.
3. Select the candidate trend that is **farthest from the global trend**.
4. Record both the local deviation score and global trend distance at every step.

#### Exhaustive Search Baseline
For small samples or restricted dimension sets, performs complete Cartesian enumeration across all dimension combinations to establish ground truth for:
- **Search Space Reduction (%)**:
  $$\text{Reduction} = \left(1 - \frac{\text{Nodes}_{\text{Surfer}}}{\text{Nodes}_{\text{Exhaustive}}}\right) \times 100\%$$
- **Speedup Factor**: $\frac{\text{Time}_{\text{Exhaustive}}}{\text{Time}_{\text{Surfer}}}$
- **Score Ratio**: $\frac{\text{Score}_{\text{Surfer}}}{\text{Score}_{\text{Exhaustive}}}$

---

## 3. Project Structure

```
TrendSurf/
│
├── main.py                     # Unified CLI entry point & Interactive Terminal Menu
├── app.py                      # Interactive Streamlit Web Dashboard
├── requirements.txt            # Python dependencies
├── plan.md                     # Architecture & implementation plan
├── README.md                   # Complete documentation
│
├── data/
│   ├── DelayedFlights.csv      # Complete flight delay dataset (247 MB)
│   ├── DelayedFlights_sample.csv # 50K sample for instant benchmarking
│   └── DelayedFlights_sample.zip # Sample ZIP archive testing upload
│
├── trendsurf/
│   ├── __init__.py
│   ├── loader.py               # CSV/ZIP loader, auto-column detection, validation
│   ├── preprocessing.py        # Date parsing, filtering, imputation, normalization
│   ├── trend.py                # Trend dataclass, temporal aggregation, sparklines, plotting
│   ├── datacube.py             # CubeNode, child generation, subgroup filtering
│   ├── scoring.py              # Euclidean, PCA, kNN, and CBLOF outlier scoring
│   ├── surfers/
│   │   ├── __init__.py         # SurferResult dataclass
│   │   ├── surfer1.py          # TrendSurfer I implementation
│   │   └── surfer2.py          # TrendSurfer II implementation
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

## 4. Installation & Setup

Ensure Python 3.10+ is installed.

```bash
# Clone or navigate to the repository
cd "c:\Users\soora\da project"

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Usage & Execution

### 5.1 Interactive Terminal Menu
Run without arguments to launch the interactive terminal interface:
```bash
python main.py
```

Menu options:
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

### 5.2 Direct Command-Line Interface (CLI)

#### Dataset Summary
```bash
python main.py summary data/DelayedFlights_sample.csv
```

#### Run TrendSurfer I
```bash
python main.py surf data/DelayedFlights_sample.csv --algorithm surfer1 --depth 3
```

#### Run TrendSurfer II
```bash
python main.py surf data/DelayedFlights_sample.csv --algorithm surfer2 --depth 3
```

#### Comparative Evaluation (Surfer I vs Surfer II vs Exhaustive Baseline)
```bash
python main.py compare data/DelayedFlights_sample.csv --depth 2 --baseline
```

#### Top Unusual Trends
```bash
python main.py top data/DelayedFlights_sample.csv --k 10
```

#### Explore a Single Dimension
```bash
python main.py explore data/DelayedFlights_sample.csv --dimension UniqueCarrier
```

#### Run on Full Dataset
```bash
python main.py surf data/DelayedFlights.csv --algorithm surfer2 --depth 3 --sample 100000
```

### 5.3 Interactive Streamlit Dashboard
Launch the web interface for visual interactive exploration:
```bash
streamlit run app.py
```

---

## 6. Experimental Results & Verification

Benchmarked on `data/DelayedFlights_sample.csv` (50,000 records, 31 daily temporal points):

| Metric | TrendSurfer I | TrendSurfer II | Exhaustive Baseline |
| :--- | :--- | :--- | :--- |
| **Discovered Subgroup** | `DayOfWeek=1 & Origin=TPA` | `DayOfWeek=1 & Dest=RNO` | `DayOfWeek=5 & Origin=BNA` |
| **Global Outlier Score** | 9.15 | **9.43** | 9.48 |
| **Local Outlier Score** | 9.97 | 8.13 | N/A |
| **Nodes Opened / Evaluated** | **216** | **216** | 607 |
| **Execution Time** | **0.583s** | **0.587s** | 4.956s |
| **Search Space Reduction** | **64.4%** | **64.4%** | 0.0% (Baseline) |
| **Speedup Factor** | **8.5x** | **8.4x** | 1.0x |

**Key Finding**: TrendSurfer II achieved **99.5%** of the exhaustive optimum score (`9.43` vs `9.48`) while opening **64.4% fewer nodes** and running **8.4x faster**.

---

## 7. Automated Testing
Run the test suite using pytest:
```bash
python -m pytest -v
```
All 13 unit tests pass covering loading, preprocessing, imputation, scoring, TrendSurfer I, TrendSurfer II, and baseline search.
