"""
Streamlit Web Dashboard for Multidimensional Trend Surfing
Run with: streamlit run app.py
"""

import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from trendsurf.loader import (
    load_dataset,
    detect_columns,
    validate_configuration,
    get_dataset_summary,
    DatasetConfig
)
from trendsurf.preprocessing import prepare_dataframe
from trendsurf.trend import compute_global_trend, compute_group_trend
from trendsurf.surfers.surfer1 import run_trendsurfer1
from trendsurf.surfers.surfer2 import run_trendsurfer2
from trendsurf.baseline import run_exhaustive_search, compare_surfer_with_baseline
from trendsurf.explorer import explore_dimension, find_top_k_unusual_trends
from trendsurf.exporter import export_result

st.set_page_config(
    page_title="Trend Surfing Dashboard",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Multidimensional Trend Discovery Using Trend Surfing")
st.markdown(
    "Discover unusual temporal trends across subgroups in multidimensional datasets "
    "using **TrendSurfer I & II** heuristics without exhaustive search combinatorial explosion."
)

# Sidebar: Dataset Selection & Controls
st.sidebar.header("📁 Dataset Configuration")

data_dir = "data"
available_files = []
if os.path.exists(data_dir):
    available_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith((".csv", ".zip"))
    ]

upload_mode = st.sidebar.radio("Dataset Source", ["Select Existing", "Upload File"])

target_file = None
uploaded_df = None

if upload_mode == "Select Existing":
    if available_files:
        target_file = st.sidebar.selectbox("Choose Dataset", available_files, index=0)
    else:
        st.sidebar.warning("No files found in data/ directory.")
else:
    uploaded = st.sidebar.file_uploader("Upload CSV or ZIP file", type=["csv", "zip"])
    if uploaded is not None:
        temp_path = os.path.join("data", uploaded.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        target_file = temp_path

sample_limit = st.sidebar.number_input(
    "Row Sample Limit (0 for full dataset)",
    min_value=0,
    max_value=2000000,
    value=50000,
    step=10000,
    help="Using a sample (e.g. 50,000) provides instantaneous results for interactive exploration."
)
nrows = sample_limit if sample_limit > 0 else None

if target_file and os.path.exists(target_file):
    @st.cache_data(show_spinner=False)
    def cached_load(path, n):
        return load_dataset(path, nrows=n)

    with st.spinner("Loading dataset..."):
        raw_df, filename = cached_load(target_file, nrows)

    col_meta = detect_columns(raw_df)

    st.sidebar.subheader("📐 Column Mapping")
    all_cols = raw_df.columns.tolist()

    temp_cols = st.sidebar.multiselect(
        "Temporal Column(s)",
        options=all_cols,
        default=[c for c in col_meta["default_temporal"] if c in all_cols]
    )

    measure_options = col_meta["measure_candidates"] or all_cols
    default_m = col_meta["default_measure"] if col_meta["default_measure"] in measure_options else measure_options[0]
    measure_col = st.sidebar.selectbox(
        "Dependent Measure",
        options=measure_options,
        index=measure_options.index(default_m) if default_m in measure_options else 0
    )

    dim_options = [c for c in all_cols if c != measure_col and c not in temp_cols]
    default_dims = [c for c in col_meta["default_dimensions"] if c in dim_options]
    dimension_cols = st.sidebar.multiselect(
        "Grouping Dimensions",
        options=dim_options,
        default=default_dims if default_dims else dim_options[:4]
    )

    # Validate
    is_valid, err_msg = validate_configuration(raw_df, temp_cols, measure_col, dimension_cols)
    if not is_valid:
        st.error(f"Configuration Error: {err_msg}")
        st.stop()

    # Preprocessing
    @st.cache_data(show_spinner=False)
    def cached_prepare(df_in, t_cols, m_col, d_cols):
        return prepare_dataframe(df_in, t_cols, m_col, d_cols)

    clean_df, temporal_index, prep_stats = cached_prepare(raw_df, temp_cols, measure_col, dimension_cols)
    global_trend = compute_global_trend(clean_df, "Date", measure_col, temporal_index)

    # Main Tabs
    tab_discovery, tab_compare, tab_explorer, tab_summary = st.tabs([
        "🔍 Trend Surfing",
        "⚖️ Surfer I vs II vs Baseline",
        "📊 Dimension Explorer",
        "📋 Dataset Overview"
    ])

    with tab_discovery:
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            algorithm = st.selectbox("Algorithm", ["TrendSurfer II", "TrendSurfer I"])
        with col_ctrl2:
            depth = st.slider("Search Depth", min_value=1, max_value=min(4, len(dimension_cols)), value=3)
        with col_ctrl3:
            scoring_method = st.selectbox("Scoring Metric", ["euclidean", "pca", "knn", "cblof"])

        if st.button("🚀 Run Trend Surfing", type="primary"):
            t0 = time.perf_counter()
            if algorithm == "TrendSurfer I":
                result = run_trendsurfer1(
                    df=clean_df,
                    global_trend=global_trend,
                    dimension_cols=dimension_cols,
                    date_col="Date",
                    measure_col=measure_col,
                    temporal_index=temporal_index,
                    target_depth=depth,
                    scoring_method=scoring_method
                )
            else:
                result = run_trendsurfer2(
                    df=clean_df,
                    global_trend=global_trend,
                    dimension_cols=dimension_cols,
                    date_col="Date",
                    measure_col=measure_col,
                    temporal_index=temporal_index,
                    target_depth=depth,
                    scoring_method=scoring_method
                )
            elapsed = time.perf_counter() - t0

            final = result.final_node

            # Metric Cards
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Selected Subgroup", final.node_id)
            m2.metric("Local Outlier Score", f"{final.local_score:.2f}")
            m3.metric("Global Distance", f"{final.global_score:.2f}")
            m4.metric("Nodes Opened", f"{result.nodes_opened}")
            m5.metric("Runtime", f"{result.execution_time:.3f}s")

            # Chart
            st.subheader("📈 Temporal Trend Comparison")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(global_trend.dates, global_trend.raw_values, label=f"Global Trend ({global_trend.label})", color="#00bcd4", linewidth=2, linestyle="--")
            if final.trend:
                ax.plot(final.trend.dates, final.trend.raw_values, label=f"Discovered Subgroup ({final.node_id})", color="#e91e63", linewidth=2.5)
            ax.set_title(f"Temporal Trend Comparison (Measure: {measure_col})", fontsize=12, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel(f"Mean {measure_col}")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig)

            # Path Details Table
            st.subheader("🪜 Step-by-Step Trajectory")
            path_rows = []
            for n in result.path:
                path_rows.append({
                    "Depth": n.depth,
                    "Subgroup": n.node_id,
                    "Local Score": round(n.local_score, 3),
                    "Global Distance": round(n.global_score, 3),
                    "Observations": n.observation_count
                })
            st.dataframe(pd.DataFrame(path_rows), use_container_width=True)

            # JSON export
            json_file = export_result(result)
            st.download_button(
                "📥 Download Result JSON",
                data=pd.Series(result.to_dict()).to_json(indent=2),
                file_name=f"{algorithm.lower().replace(' ', '')}_result.json",
                mime="application/json"
            )

    with tab_compare:
        st.subheader("⚖️ Algorithm Comparison & Exhaustive Baseline")
        st.markdown(
            "Compare the search efficiency, execution runtime, and outlier discovery quality "
            "between heuristic TrendSurfers and exhaustive enumeration."
        )

        cmp_depth = st.slider("Comparison Depth", 1, min(3, len(dimension_cols)), 2)
        include_base = st.checkbox("Include Exhaustive Baseline", value=True)

        if st.button("Run Comparison"):
            with st.spinner("Computing comparative trajectories..."):
                r1 = run_trendsurfer1(clean_df, global_trend, dimension_cols, "Date", measure_col, temporal_index, cmp_depth)
                r2 = run_trendsurfer2(clean_df, global_trend, dimension_cols, "Date", measure_col, temporal_index, cmp_depth)
                b_res = None
                if include_base:
                    b_res = run_exhaustive_search(clean_df, global_trend, dimension_cols, "Date", measure_col, temporal_index, cmp_depth, max_values_per_dim=15)

            cmp_table = {
                "Metric": [
                    "Discovered Subgroup",
                    "Global Trend Distance",
                    "Local Outlier Score",
                    "Nodes Evaluated",
                    "Runtime (seconds)",
                    "Search Reduction (%)"
                ],
                "TrendSurfer I": [
                    r1.final_node.node_id,
                    f"{r1.final_node.global_score:.2f}",
                    f"{r1.final_node.local_score:.2f}",
                    f"{r1.nodes_opened}",
                    f"{r1.execution_time:.3f}s",
                    f"{((1.0 - r1.nodes_opened / b_res.total_nodes_evaluated)*100):.1f}%" if b_res and b_res.total_nodes_evaluated > 0 else "N/A"
                ],
                "TrendSurfer II": [
                    r2.final_node.node_id,
                    f"{r2.final_node.global_score:.2f}",
                    f"{r2.final_node.local_score:.2f}",
                    f"{r2.nodes_opened}",
                    f"{r2.execution_time:.3f}s",
                    f"{((1.0 - r2.nodes_opened / b_res.total_nodes_evaluated)*100):.1f}%" if b_res and b_res.total_nodes_evaluated > 0 else "N/A"
                ]
            }

            if b_res:
                cmp_table["Exhaustive Baseline"] = [
                    b_res.top_node.node_id,
                    f"{b_res.top_node.global_score:.2f}",
                    "N/A",
                    f"{b_res.total_nodes_evaluated}",
                    f"{b_res.execution_time:.3f}s",
                    "0.0% (Baseline)"
                ]

            st.dataframe(pd.DataFrame(cmp_table), use_container_width=True)

    with tab_explorer:
        st.subheader("📊 Top Unusual Trends & Dimension Ranking")
        col_exp1, col_exp2 = st.columns([1, 2])
        with col_exp1:
            explore_dim = st.selectbox("Explore Dimension", dimension_cols)
            top_k_dim = st.slider("Top N Values", 5, 30, 10)
        with col_exp2:
            st.write("")
            st.write("")
            if st.button("Explore"):
                dim_res = explore_dimension(clean_df, explore_dim, global_trend, "Date", measure_col, temporal_index, top_n=top_k_dim)
                dim_df = pd.DataFrame([{
                    "Rank": i + 1,
                    "Value": r["value"],
                    "Outlier Score": r["score"],
                    "Mean Measure": r["mean_measure"],
                    "Std Measure": r["std_measure"],
                    "Count": r["observations"],
                    "Sparkline": r["sparkline"]
                } for i, r in enumerate(dim_res)])
                st.dataframe(dim_df, use_container_width=True)

        st.divider()
        st.subheader("🏆 Top-K Discovered Unusual Trends Across Cube")
        k_count = st.slider("K Unusual Trends", 5, 20, 10)
        if st.button("Find Top-K Trends"):
            top_k = find_top_k_unusual_trends(clean_df, dimension_cols, global_trend, "Date", measure_col, temporal_index, k=k_count)
            top_df = pd.DataFrame([{
                "Rank": i + 1,
                "Depth": r["depth"],
                "Subgroup": r["label"],
                "Outlier Score": r["score"],
                "Mean Measure": r["mean_measure"],
                "Records": r["observations"],
                "Sparkline": r["sparkline"]
            } for i, r in enumerate(top_k)])
            st.dataframe(top_df, use_container_width=True)

    with tab_summary:
        st.subheader("📋 Dataset Overview")
        summary = get_dataset_summary(raw_df, filename)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Rows", f"{summary['rows']:,}")
        s2.metric("Total Columns", f"{summary['columns']}")
        s3.metric("Memory Usage", f"{summary['memory_mb']} MB")
        s4.metric("Valid Cleaned Rows", f"{len(clean_df):,}")

        st.write("Column Types & Missing Counts:")
        meta_table = pd.DataFrame({
            "Data Type": summary["column_types"],
            "Missing Count": [summary["missing_counts"].get(c, 0) for c in raw_df.columns]
        })
        st.dataframe(meta_table, use_container_width=True)

else:
    st.info("👈 Please select or upload a dataset in the sidebar to begin.")
