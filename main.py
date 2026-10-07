"""
Main Entry Point for Trend Surfing Application
Supports both interactive terminal UI and direct command-line execution.
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
from typing import Optional, Dict, Any, List
import pandas as pd
from rich.prompt import Prompt, Confirm

from trendsurf.loader import (
    load_dataset,
    detect_columns,
    validate_configuration,
    get_dataset_summary,
    DatasetConfig
)
from trendsurf.preprocessing import prepare_dataframe
from trendsurf.trend import compute_global_trend
from trendsurf.surfers.surfer1 import run_trendsurfer1
from trendsurf.surfers.surfer2 import run_trendsurfer2
from trendsurf.surfers import SurferResult
from trendsurf.baseline import run_exhaustive_search, compare_surfer_with_baseline, BaselineResult
from trendsurf.explorer import explore_dimension, find_top_k_unusual_trends
from trendsurf.exporter import export_result, export_comparison_table
from trendsurf.terminal_ui import (
    console,
    print_banner,
    display_dataset_summary,
    display_configuration,
    display_surfer_result,
    display_comparison_table,
    display_top_trends,
    display_dimension_exploration
)


class SessionState:
    """Maintains currently active dataset session in memory."""
    def __init__(self):
        self.config: Optional[DatasetConfig] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None
        self.temporal_index: Optional[pd.DatetimeIndex] = None
        self.global_trend = None
        self.last_surfer1_res: Optional[SurferResult] = None
        self.last_surfer2_res: Optional[SurferResult] = None
        self.last_baseline_res: Optional[BaselineResult] = None
        self.last_exported_file: Optional[str] = None


session = SessionState()


def setup_dataset(filepath: str, sample_size: Optional[int] = None, auto_confirm: bool = False) -> bool:
    """Loads dataset, detects columns, validates configuration, and prepares trends."""
    if not os.path.exists(filepath):
        console.print(f"[bold red]ERROR: File does not exist:[/bold red] {filepath}")
        return False

    with console.status(f"[bold green]Loading {filepath}...[/bold green]"):
        try:
            df, filename = load_dataset(filepath, nrows=sample_size)
        except Exception as e:
            console.print(f"[bold red]ERROR reading dataset:[/bold red] {e}")
            return False

    summary = get_dataset_summary(df, filename)
    display_dataset_summary(summary)

    detected = detect_columns(df)

    temporal_cols = detected["default_temporal"]
    measure_col = detected["default_measure"]
    dimension_cols = detected["default_dimensions"]

    config = DatasetConfig(
        filepath=filepath,
        temporal_cols=temporal_cols,
        measure_col=measure_col,
        dimension_cols=dimension_cols,
        sample_size=sample_size
    )

    if not auto_confirm:
        display_configuration(config)
        console.print("--------------------------------------------------")
        console.print("1. Confirm")
        console.print("2. Change configuration")
        console.print("--------------------------------------------------")
        choice = Prompt.ask("Choose option", choices=["1", "2"], default="1")

        if choice == "2":
            console.print("\n[bold cyan]Available columns:[/bold cyan]")
            for i, c in enumerate(df.columns, 1):
                console.print(f"  {i}. {c}")

            t_input = Prompt.ask("\nEnter temporal columns (comma-separated)", default=", ".join(temporal_cols))
            temporal_cols = [c.strip() for c in t_input.split(",") if c.strip()]

            m_input = Prompt.ask("Enter dependent measure", default=measure_col)
            measure_col = m_input.strip()

            d_input = Prompt.ask("Enter grouping dimensions (comma-separated)", default=", ".join(dimension_cols))
            dimension_cols = [c.strip() for c in d_input.split(",") if c.strip()]

            config = DatasetConfig(
                filepath=filepath,
                temporal_cols=temporal_cols,
                measure_col=measure_col,
                dimension_cols=dimension_cols,
                sample_size=sample_size
            )

    is_valid, err_msg = validate_configuration(df, temporal_cols, measure_col, dimension_cols)
    if not is_valid:
        console.print(f"[bold red]ERROR: {err_msg}[/bold red]")
        return False

    with console.status("[bold green]Preprocessing data & building global trend...[/bold green]"):
        clean_df, t_index, stats = prepare_dataframe(
            df=df,
            temporal_cols=temporal_cols,
            measure_col=measure_col,
            dimension_cols=dimension_cols
        )
        gt = compute_global_trend(
            df=clean_df,
            date_col="Date",
            measure_col=measure_col,
            temporal_index=t_index
        )

    session.config = config
    session.raw_df = df
    session.clean_df = clean_df
    session.temporal_index = t_index
    session.global_trend = gt

    console.print(f"\n[bold green][OK] Session initialized successfully![/bold green]")
    console.print(f"Usable rows: [bold]{len(clean_df):,}[/bold] | Time points: [bold]{len(t_index)}[/bold] | Global Mean {measure_col}: [bold cyan]{gt.mean_val:.2f}[/bold cyan]\n")
    return True


def ensure_session(default_path: str = "data/DelayedFlights_sample.csv") -> bool:
    """Ensures a dataset is loaded into the session."""
    if session.clean_df is not None:
        return True

    console.print("[yellow]No dataset loaded in current session.[/yellow]")
    target_path = default_path if os.path.exists(default_path) else "data/DelayedFlights.csv"
    if not os.path.exists(target_path):
        target_path = Prompt.ask("Enter dataset path")
    return setup_dataset(target_path)


def interactive_menu():
    """Runs the main interactive terminal menu."""
    while True:
        print_banner()
        if session.config:
            console.print(f"[bold green]Current dataset:[/bold green] {os.path.basename(session.config.filepath)} "
                          f"| [bold cyan]Measure:[/bold cyan] {session.config.measure_col} "
                          f"| [bold yellow]Dims:[/bold yellow] {', '.join(session.config.dimension_cols)}")
        else:
            console.print("[dim]No dataset loaded yet. Choose option 1 to load.[/dim]")

        console.print("\n[bold]Select an action:[/bold]")
        console.print("  1. Upload / Load Dataset")
        console.print("  2. Dataset Summary")
        console.print("  3. Discover Unusual Trend (Quick Run)")
        console.print("  4. Run TrendSurfer I")
        console.print("  5. Run TrendSurfer II")
        console.print("  6. Compare TrendSurfer I vs II")
        console.print("  7. Explore a Dimension")
        console.print("  8. Show Top Unusual Trends")
        console.print("  9. Run Exhaustive Baseline")
        console.print(" 10. Export Results")
        console.print("  0. Exit\n")

        choice = Prompt.ask("Enter choice", default="1")

        if choice == "0":
            console.print("[bold yellow]Exiting TrendSurfer. Goodbye![/bold yellow]")
            break

        elif choice == "1":
            default_candidate = "data/DelayedFlights_sample.csv" if os.path.exists("data/DelayedFlights_sample.csv") else "data/DelayedFlights.csv"
            path = Prompt.ask("Enter dataset path", default=default_candidate)
            setup_dataset(path)

        elif choice == "2":
            if ensure_session():
                summary = get_dataset_summary(session.raw_df, os.path.basename(session.config.filepath))
                display_dataset_summary(summary)
                display_configuration(session.config)

        elif choice == "3":
            if ensure_session():
                depth = int(Prompt.ask("Enter search depth", default="3"))
                with console.status("[bold green]Surfing data cube (TrendSurfer II)...[/bold green]"):
                    res = run_trendsurfer2(
                        df=session.clean_df,
                        global_trend=session.global_trend,
                        dimension_cols=session.config.dimension_cols,
                        date_col="Date",
                        measure_col=session.config.measure_col,
                        temporal_index=session.temporal_index,
                        target_depth=depth
                    )
                session.last_surfer2_res = res
                display_surfer_result(res)

        elif choice == "4":
            if ensure_session():
                depth = int(Prompt.ask("Enter search depth", default="3"))
                scoring = Prompt.ask("Outlier scoring metric", choices=["euclidean", "pca", "knn", "cblof"], default="euclidean")
                with console.status(f"[bold green]Running TrendSurfer I (depth={depth}, {scoring})...[/bold green]"):
                    res = run_trendsurfer1(
                        df=session.clean_df,
                        global_trend=session.global_trend,
                        dimension_cols=session.config.dimension_cols,
                        date_col="Date",
                        measure_col=session.config.measure_col,
                        temporal_index=session.temporal_index,
                        target_depth=depth,
                        scoring_method=scoring
                    )
                session.last_surfer1_res = res
                display_surfer_result(res)

        elif choice == "5":
            if ensure_session():
                depth = int(Prompt.ask("Enter search depth", default="3"))
                scoring = Prompt.ask("Outlier scoring metric", choices=["euclidean", "pca", "knn", "cblof"], default="euclidean")
                with console.status(f"[bold green]Running TrendSurfer II (depth={depth}, {scoring})...[/bold green]"):
                    res = run_trendsurfer2(
                        df=session.clean_df,
                        global_trend=session.global_trend,
                        dimension_cols=session.config.dimension_cols,
                        date_col="Date",
                        measure_col=session.config.measure_col,
                        temporal_index=session.temporal_index,
                        target_depth=depth,
                        scoring_method=scoring
                    )
                session.last_surfer2_res = res
                display_surfer_result(res)

        elif choice == "6":
            if ensure_session():
                depth = int(Prompt.ask("Enter comparison depth", default="3"))
                with console.status("[bold green]Executing comparative evaluation...[/bold green]"):
                    r1 = run_trendsurfer1(
                        df=session.clean_df,
                        global_trend=session.global_trend,
                        dimension_cols=session.config.dimension_cols,
                        date_col="Date",
                        measure_col=session.config.measure_col,
                        temporal_index=session.temporal_index,
                        target_depth=depth
                    )
                    r2 = run_trendsurfer2(
                        df=session.clean_df,
                        global_trend=session.global_trend,
                        dimension_cols=session.config.dimension_cols,
                        date_col="Date",
                        measure_col=session.config.measure_col,
                        temporal_index=session.temporal_index,
                        target_depth=depth
                    )
                session.last_surfer1_res = r1
                session.last_surfer2_res = r2
                display_comparison_table(r1, r2, session.last_baseline_res)

        elif choice == "7":
            if ensure_session():
                console.print(f"Available dimensions: {', '.join(session.config.dimension_cols)}")
                dim = Prompt.ask("Select dimension to explore", default=session.config.dimension_cols[0])
                if dim not in session.config.dimension_cols:
                    console.print(f"[red]Invalid dimension: {dim}[/red]")
                    continue
                with console.status(f"[bold green]Exploring dimension '{dim}'...[/bold green]"):
                    results = explore_dimension(
                        df=session.clean_df,
                        dimension=dim,
                        global_trend=session.global_trend,
                        date_col="Date",
                        measure_col=session.config.measure_col,
                        temporal_index=session.temporal_index
                    )
                display_dimension_exploration(results, dim)

        elif choice == "8":
            if ensure_session():
                k_val = int(Prompt.ask("Enter K (number of top trends)", default="10"))
                with console.status(f"[bold green]Discovering top {k_val} unusual trends...[/bold green]"):
                    top_trends = find_top_k_unusual_trends(
                        df=session.clean_df,
                        dimension_cols=session.config.dimension_cols,
                        global_trend=session.global_trend,
                        date_col="Date",
                        measure_col=session.config.measure_col,
                        temporal_index=session.temporal_index,
                        k=k_val
                    )
                display_top_trends(top_trends)

        elif choice == "9":
            if ensure_session():
                depth = int(Prompt.ask("Enter baseline target depth (recommended 1 or 2 for speed)", default="2"))
                with console.status(f"[bold green]Running exhaustive search (depth={depth})...[/bold green]"):
                    b_res = run_exhaustive_search(
                        df=session.clean_df,
                        global_trend=session.global_trend,
                        dimension_cols=session.config.dimension_cols,
                        date_col="Date",
                        measure_col=session.config.measure_col,
                        temporal_index=session.temporal_index,
                        target_depth=depth,
                        max_values_per_dim=10
                    )
                session.last_baseline_res = b_res
                console.print(f"\n[bold green]Exhaustive search complete![/bold green]")
                console.print(f"Total nodes evaluated: [bold]{b_res.total_nodes_evaluated}[/bold] in {b_res.execution_time:.3f}s")
                console.print(f"Top discovered node  : [bold yellow]{b_res.top_node.node_id}[/bold yellow] (Score: {b_res.top_node.global_score:.2f})")

        elif choice == "10":
            if session.last_surfer1_res or session.last_surfer2_res or session.last_baseline_res:
                out_file = None
                if session.last_surfer2_res:
                    out_file = export_result(session.last_surfer2_res, output_dir="results/surfer2")
                elif session.last_surfer1_res:
                    out_file = export_result(session.last_surfer1_res, output_dir="results/surfer1")
                elif session.last_baseline_res:
                    out_file = export_result(session.last_baseline_res, output_dir="results/exports")
                console.print(f"[bold green]Exported results to:[/bold green] {out_file}")
            else:
                console.print("[yellow]No run results available yet to export. Run TrendSurfer I or II first.[/yellow]")

        Prompt.ask("\n[dim]Press Enter to continue...[/dim]", default="")


def main():
    parser = argparse.ArgumentParser(description="Trend Surfing: Multidimensional Trend Discovery")
    subparsers = parser.add_subparsers(dest="command")

    # Command: summary
    p_summary = subparsers.add_parser("summary", help="Display dataset overview and detected columns")
    p_summary.add_argument("filepath", nargs="?", default="data/DelayedFlights_sample.csv")
    p_summary.add_argument("--sample", type=int, default=None)

    # Command: load
    p_load = subparsers.add_parser("load", help="Load and validate dataset")
    p_load.add_argument("filepath", nargs="?", default="data/DelayedFlights_sample.csv")

    # Command: surf
    p_surf = subparsers.add_parser("surf", help="Run TrendSurfer heuristic search")
    p_surf.add_argument("filepath", nargs="?", default="data/DelayedFlights_sample.csv")
    p_surf.add_argument("--algorithm", choices=["surfer1", "surfer2"], default="surfer2")
    p_surf.add_argument("--depth", type=int, default=3)
    p_surf.add_argument("--scoring", default="euclidean", choices=["euclidean", "pca", "knn", "cblof"])
    p_surf.add_argument("--sample", type=int, default=None)

    # Command: compare
    p_compare = subparsers.add_parser("compare", help="Compare TrendSurfer I vs II")
    p_compare.add_argument("filepath", nargs="?", default="data/DelayedFlights_sample.csv")
    p_compare.add_argument("--depth", type=int, default=3)
    p_compare.add_argument("--baseline", action="store_true", help="Include exhaustive baseline")

    # Command: top
    p_top = subparsers.add_parser("top", help="Find top-K unusual trends")
    p_top.add_argument("filepath", nargs="?", default="data/DelayedFlights_sample.csv")
    p_top.add_argument("--k", type=int, default=10)

    # Command: baseline
    p_base = subparsers.add_parser("baseline", help="Run exhaustive search baseline")
    p_base.add_argument("filepath", nargs="?", default="data/DelayedFlights_sample.csv")
    p_base.add_argument("--depth", type=int, default=2)

    # Command: explore
    p_exp = subparsers.add_parser("explore", help="Explore a single dimension")
    p_exp.add_argument("filepath", nargs="?", default="data/DelayedFlights_sample.csv")
    p_exp.add_argument("--dimension", required=True)

    args = parser.parse_args()

    # If no arguments provided, launch interactive menu
    if not args.command:
        interactive_menu()
        return

    # Handle direct CLI commands
    filepath = getattr(args, "filepath", "data/DelayedFlights_sample.csv")
    sample_size = getattr(args, "sample", None)

    if not setup_dataset(filepath, sample_size=sample_size, auto_confirm=True):
        sys.exit(1)

    if args.command in ["summary", "load"]:
        # Setup already showed summary and config
        return

    elif args.command == "surf":
        if args.algorithm == "surfer1":
            res = run_trendsurfer1(
                df=session.clean_df,
                global_trend=session.global_trend,
                dimension_cols=session.config.dimension_cols,
                date_col="Date",
                measure_col=session.config.measure_col,
                temporal_index=session.temporal_index,
                target_depth=args.depth,
                scoring_method=args.scoring
            )
        else:
            res = run_trendsurfer2(
                df=session.clean_df,
                global_trend=session.global_trend,
                dimension_cols=session.config.dimension_cols,
                date_col="Date",
                measure_col=session.config.measure_col,
                temporal_index=session.temporal_index,
                target_depth=args.depth,
                scoring_method=args.scoring
            )
        display_surfer_result(res)
        export_file = export_result(res)
        console.print(f"[dim]Results saved to: {export_file}[/dim]")

    elif args.command == "compare":
        r1 = run_trendsurfer1(
            df=session.clean_df,
            global_trend=session.global_trend,
            dimension_cols=session.config.dimension_cols,
            date_col="Date",
            measure_col=session.config.measure_col,
            temporal_index=session.temporal_index,
            target_depth=args.depth
        )
        r2 = run_trendsurfer2(
            df=session.clean_df,
            global_trend=session.global_trend,
            dimension_cols=session.config.dimension_cols,
            date_col="Date",
            measure_col=session.config.measure_col,
            temporal_index=session.temporal_index,
            target_depth=args.depth
        )
        b_res = None
        if args.baseline:
            b_res = run_exhaustive_search(
                df=session.clean_df,
                global_trend=session.global_trend,
                dimension_cols=session.config.dimension_cols,
                date_col="Date",
                measure_col=session.config.measure_col,
                temporal_index=session.temporal_index,
                target_depth=min(args.depth, 2)
            )
        display_comparison_table(r1, r2, b_res)

    elif args.command == "top":
        trends = find_top_k_unusual_trends(
            df=session.clean_df,
            dimension_cols=session.config.dimension_cols,
            global_trend=session.global_trend,
            date_col="Date",
            measure_col=session.config.measure_col,
            temporal_index=session.temporal_index,
            k=args.k
        )
        display_top_trends(trends)

    elif args.command == "baseline":
        b_res = run_exhaustive_search(
            df=session.clean_df,
            global_trend=session.global_trend,
            dimension_cols=session.config.dimension_cols,
            date_col="Date",
            measure_col=session.config.measure_col,
            temporal_index=session.temporal_index,
            target_depth=args.depth
        )
        console.print(f"\n[bold green]Exhaustive Baseline Results:[/bold green]")
        console.print(f"Nodes evaluated: [bold]{b_res.total_nodes_evaluated}[/bold]")
        console.print(f"Top discovered node: [bold yellow]{b_res.top_node.node_id}[/bold yellow] (Score: {b_res.top_node.global_score:.2f})")
        console.print(f"Runtime: {b_res.execution_time:.3f}s")
        export_file = export_result(b_res)
        console.print(f"[dim]Results saved to: {export_file}[/dim]")

    elif args.command == "explore":
        results = explore_dimension(
            df=session.clean_df,
            dimension=args.dimension,
            global_trend=session.global_trend,
            date_col="Date",
            measure_col=session.config.measure_col,
            temporal_index=session.temporal_index
        )
        display_dimension_exploration(results, args.dimension)


if __name__ == "__main__":
    main()
