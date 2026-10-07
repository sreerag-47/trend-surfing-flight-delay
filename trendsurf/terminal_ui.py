"""
Terminal User Interface Module
Implements interactive Rich-based UI, menus, and beautiful terminal rendering.
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

from typing import List, Dict, Any, Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from trendsurf.loader import DatasetConfig, get_dataset_summary
from trendsurf.surfers import SurferResult
from trendsurf.baseline import BaselineResult
from trendsurf.trend import Trend, render_ascii_sparkline, plot_trend_terminal

console = Console()


def print_banner():
    banner = """
======================================================================
               MULTIDIMENSIONAL TREND DISCOVERY
                     USING TREND SURFING
======================================================================
    """
    console.print(Panel(banner.strip(), style="bold cyan", box=box.DOUBLE))


def display_dataset_summary(summary: Dict[str, Any]):
    console.print(f"\n[bold green]Dataset loaded successfully.[/bold green]\n")
    console.print(f"File       : [bold yellow]{summary['filename']}[/bold yellow]")
    console.print(f"Rows       : [bold]{summary['rows']:,}[/bold]")
    console.print(f"Columns    : [bold]{summary['columns']}[/bold]")
    console.print(f"Memory     : {summary['memory_mb']} MB")

    # Column table
    table = Table(title="Dataset Columns", box=box.SIMPLE_HEAVY)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Column Name", style="bold")
    table.add_column("Type", style="magenta")
    table.add_column("Missing Values", style="red")

    col_types = summary["column_types"]
    missing = summary["missing_counts"]

    for idx, (col, dtype) in enumerate(col_types.items(), start=1):
        miss_str = f"{missing[col]:,}" if col in missing else "0"
        table.add_row(str(idx), col, dtype, miss_str)

    console.print(table)


def display_configuration(config: DatasetConfig):
    text = (
        f"[bold]Dataset:[/bold] {os.path.basename(config.filepath)}\n"
        f"[bold]Temporal columns:[/bold] {', '.join(config.temporal_cols)}\n"
        f"[bold]Dependent measure:[/bold] [cyan]{config.measure_col}[/cyan]\n"
        f"[bold]Grouping dimensions:[/bold] {', '.join(config.dimension_cols)}"
    )
    console.print(Panel(text, title="[bold cyan]DATASET CONFIGURATION[/bold cyan]", box=box.ROUNDED))


def display_surfer_result(result: SurferResult, show_chart: bool = True):
    console.print(f"\n[bold magenta]==================================================[/bold magenta]")
    console.print(f"[bold magenta]              TREND SURFING RESULT                [/bold magenta]")
    console.print(f"[bold magenta]==================================================[/bold magenta]\n")
    console.print(f"Algorithm   : [bold yellow]{result.algorithm}[/bold yellow]")
    console.print(f"Measure     : [bold cyan]{result.measure_col}[/bold cyan]")
    console.print(f"Depth       : [bold]{result.target_depth}[/bold]")
    console.print(f"Scoring     : {result.scoring_method}")
    console.print(f"Nodes opened: [bold green]{result.nodes_opened}[/bold green]")
    console.print(f"Time        : [bold]{result.execution_time:.3f} seconds[/bold]\n")

    # Path Table
    table = Table(title=f"Discovered Path: {result.algorithm}", box=box.ROUNDED)
    table.add_column("Depth", justify="center", style="cyan")
    table.add_column("Dimension(s)", style="bold")
    table.add_column("Value(s)", style="yellow")
    table.add_column("Local Outlier Score", justify="right", style="magenta")
    table.add_column("Global Trend Distance", justify="right", style="green")
    table.add_column("Records", justify="right", style="white")

    for node in result.path:
        if node.depth == 0:
            table.add_row(
                "0",
                "ALL (Global)",
                "Entire Dataset",
                "0.00",
                "0.00",
                f"{node.observation_count:,}"
            )
        else:
            dim_names = ", ".join(node.dimensions.keys())
            dim_vals = ", ".join(str(v) for v in node.dimensions.values())
            table.add_row(
                str(node.depth),
                dim_names,
                dim_vals,
                f"{node.local_score:.2f}",
                f"{node.global_score:.2f}",
                f"{node.observation_count:,}"
            )

    console.print(table)

    final = result.final_node
    final_text = (
        f"[bold]Selected Group:[/bold] {final.node_id}\n"
        f"[bold]Local Outlier Score:[/bold] {final.local_score:.2f}\n"
        f"[bold]Global Trend Distance:[/bold] {final.global_score:.2f}\n"
        f"[bold]Sample Size:[/bold] {final.observation_count:,} observations\n"
        f"[bold]Nodes Opened:[/bold] {result.nodes_opened}\n"
        f"[bold]Execution Time:[/bold] {result.execution_time:.3f} seconds"
    )
    console.print(Panel(final_text, title="[bold green]FINAL DISCOVERED TREND[/bold green]", box=box.ROUNDED))

    if show_chart and final.trend:
        console.print(Panel(
            plot_trend_terminal(final.trend, result.global_trend, title=f"Trend Comparison: {final.node_id}"),
            title="[bold yellow]Terminal Trend Plot[/bold yellow]",
            box=box.MINIMAL
        ))


def display_comparison_table(res1: SurferResult, res2: SurferResult, baseline: Optional[BaselineResult] = None):
    table = Table(title="Heuristic & Baseline Comparison", box=box.DOUBLE)
    table.add_column("Metric", style="bold")
    table.add_column(res1.algorithm, style="cyan")
    table.add_column(res2.algorithm, style="magenta")
    if baseline:
        table.add_column("Exhaustive Baseline", style="green")

    row_path1 = res1.final_node.node_id
    row_path2 = res2.final_node.node_id
    table.add_row("Discovered Path", row_path1, row_path2, baseline.top_node.node_id if baseline else "N/A")
    table.add_row("Global Outlier Score", f"{res1.final_node.global_score:.2f}", f"{res2.final_node.global_score:.2f}", f"{baseline.top_node.global_score:.2f}" if baseline else "N/A")
    table.add_row("Local Outlier Score", f"{res1.final_node.local_score:.2f}", f"{res2.final_node.local_score:.2f}", "N/A")
    table.add_row("Nodes Opened / Evaluated", str(res1.nodes_opened), str(res2.nodes_opened), str(baseline.total_nodes_evaluated) if baseline else "N/A")
    table.add_row("Execution Time (sec)", f"{res1.execution_time:.3f}s", f"{res2.execution_time:.3f}s", f"{baseline.execution_time:.3f}s" if baseline else "N/A")

    if baseline and baseline.total_nodes_evaluated > 0:
        red1 = (1.0 - res1.nodes_opened / baseline.total_nodes_evaluated) * 100
        red2 = (1.0 - res2.nodes_opened / baseline.total_nodes_evaluated) * 100
        table.add_row("Search Reduction (%)", f"{red1:.1f}%", f"{red2:.1f}%", "0.0% (Baseline)")

    console.print(table)


def display_top_trends(trends: List[Dict[str, Any]]):
    table = Table(title="Top Discovered Unusual Trends", box=box.ROUNDED)
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Depth", justify="center")
    table.add_column("Subgroup / Dimensions", style="bold yellow")
    table.add_column("Outlier Score", justify="right", style="magenta")
    table.add_column("Mean Delay", justify="right", style="green")
    table.add_column("Count", justify="right")
    table.add_column("Trend Shape", style="cyan")

    for idx, item in enumerate(trends, start=1):
        table.add_row(
            str(idx),
            str(item["depth"]),
            item["label"],
            f"{item['score']:.2f}",
            f"{item['mean_measure']:.1f}",
            f"{item['observations']:,}",
            item["sparkline"]
        )

    console.print(table)


def display_dimension_exploration(results: List[Dict[str, Any]], dim_name: str):
    table = Table(title=f"Dimension Outlier Ranking: {dim_name}", box=box.ROUNDED)
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Value", style="bold yellow")
    table.add_column("Outlier Score", justify="right", style="magenta")
    table.add_column("Mean Delay", justify="right", style="green")
    table.add_column("Std Delay", justify="right")
    table.add_column("Count", justify="right")
    table.add_column("Sparkline", style="cyan")

    for idx, r in enumerate(results, start=1):
        table.add_row(
            str(idx),
            r["value"],
            f"{r['score']:.2f}",
            f"{r['mean_measure']:.1f}",
            f"{r['std_measure']:.1f}",
            f"{r['observations']:,}",
            r["sparkline"]
        )

    console.print(table)
