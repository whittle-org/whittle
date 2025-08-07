#!/usr/bin/env python3
"""
GPU Utilization Analysis Script for Distributed Training Profiling Results

This script processes JSON profiling files from distributed training runs and creates
visualizations showing GPU utilization, tokens per second, and other metrics across
different node/GPU configurations.

Example Usage:
    python visualize_tps_profiler.py --profiling-dir /path/to/profiling/directory --output-dir /path/to/output/directory

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
    }
)

COLORS = {
    "primary_blue": "#2E86AB",
    "secondary_blue": "#A23B72",
    "accent_coral": "#F18F01",
    "light_blue": "#87CEEB",
    "dark_blue": "#1E5F74",
    "success_green": "#28A745",
    "warning_orange": "#FD7E14",
    "error_red": "#DC3545",
}

BAR_WIDTH = 0.2


class GPUUtilizationAnalyzer:
    """Analyzer for GPU utilization data from distributed training profiling."""

    def __init__(self, profiling_dir: str):
        self.profiling_dir = Path(profiling_dir)
        self.data = {}
        self.aggregated_data = {}

    def load_profiling_data(self) -> dict[str, list[dict]]:
        """Load all JSON profiling files from the directory."""
        json_files = list(self.profiling_dir.glob("*.json"))

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.profiling_dir}")

        print(f"Found {len(json_files)} profiling files")

        # Group files by configuration (nodes, gpus, strategy, timestamp)
        grouped_data = defaultdict(list)

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Extract configuration key
                config_key = self._extract_config_key(data)
                grouped_data[config_key].append(data)

            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")

        self.data = dict(grouped_data)
        print(f"Loaded data for {len(self.data)} different configurations")
        return self.data

    def _extract_config_key(self, data: dict) -> str:
        """Extract configuration key from profiling data."""
        hw_config = data.get("hardware_config", {})
        num_nodes = hw_config.get("num_nodes", 0)
        num_gpus = hw_config.get("num_gpus_total", 0)
        strategy = hw_config.get("parallel_strategy", "unknown")

        return f"{num_nodes}nodes_{num_gpus}gpus_{strategy}"

    def aggregate_data(self) -> dict[str, dict]:
        """Aggregate data across ranks for each configuration."""
        self.aggregated_data = {}

        for config_key, rank_data_list in self.data.items():
            if not rank_data_list:
                continue

            # Initialize aggregation containers
            agg_data = {
                "config": rank_data_list[0]["hardware_config"],
                "num_ranks": len(rank_data_list),
                "tokens_per_second": [],
                "gpu_utilization": [],
                "memory_utilization": [],
                "peak_memory_usage": [],
                "total_tokens": 0,
                "total_runtime": 0,
                "individual_ranks": [],
            }

            for rank_data in rank_data_list:
                perf = rank_data.get("local_performance", {})
                hw_metrics = rank_data.get("local_hardware_metrics", {})

                # Collect per-rank metrics
                tps = perf.get("average_tokens_per_second", 0)
                gpu_util = hw_metrics.get("average_gpu_utilization", 0)
                mem_util = hw_metrics.get("average_memory_utilization", 0)
                peak_mem = hw_metrics.get("peak_memory_usage_gb", 0)

                agg_data["tokens_per_second"].append(tps)
                agg_data["gpu_utilization"].append(gpu_util)
                agg_data["memory_utilization"].append(mem_util)
                agg_data["peak_memory_usage"].append(peak_mem)

                agg_data["total_tokens"] += perf.get("tokens_processed", 0)
                agg_data["total_runtime"] = max(
                    agg_data["total_runtime"], perf.get("total_runtime_seconds", 0)
                )

                agg_data["individual_ranks"].append(
                    {
                        "rank": rank_data.get("rank_info", {}).get("rank", -1),
                        "tokens_per_second": tps,
                        "gpu_utilization": gpu_util,
                        "memory_utilization": mem_util,
                    }
                )

            # Calculate aggregate statistics
            agg_data["avg_tokens_per_second"] = np.mean(agg_data["tokens_per_second"])
            agg_data["std_tokens_per_second"] = np.std(agg_data["tokens_per_second"])
            agg_data["total_global_tps"] = sum(agg_data["tokens_per_second"])

            agg_data["avg_gpu_utilization"] = np.mean(agg_data["gpu_utilization"])
            agg_data["std_gpu_utilization"] = np.std(agg_data["gpu_utilization"])

            agg_data["avg_memory_utilization"] = np.mean(agg_data["memory_utilization"])
            agg_data["total_peak_memory"] = sum(agg_data["peak_memory_usage"])

            self.aggregated_data[config_key] = agg_data

        return self.aggregated_data

    def create_summary_plots(self, output_dir: str = None):
        """Create comprehensive summary plots."""
        if not self.aggregated_data:
            raise ValueError("No aggregated data available. Run aggregate_data() first.")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Create multiple plots
        self._plot_tokens_per_second_comparison(output_dir)
        self._plot_gpu_utilization_comparison(output_dir)
        self._plot_scaling_efficiency(output_dir)
        self._plot_detailed_metrics_heatmap(output_dir)

        # Create summary table
        self._create_summary_table(output_dir)

    def _plot_tokens_per_second_comparison(self, output_dir: str = None):
        """Plot tokens per second comparison across configurations."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            "Tokens Per Second Performance Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Prepare data
        configs = []
        avg_tps = []
        total_tps = []
        std_tps = []

        for config_key, data in self.aggregated_data.items():
            # Create readable config label
            num_nodes = data["config"]["num_nodes"]
            num_gpus = data["config"]["num_gpus_total"]
            config_label = f"{num_nodes}/{num_gpus}"

            configs.append(config_label)
            avg_tps.append(data["avg_tokens_per_second"])
            total_tps.append(data["total_global_tps"])
            std_tps.append(data["std_tokens_per_second"])

        # Sort by total GPUs for better visualization
        sorted_data = sorted(
            zip(configs, avg_tps, total_tps, std_tps),
            key=lambda x: int(x[0].split("/")[1]),
        )
        configs, avg_tps, total_tps, std_tps = zip(*sorted_data)

        x_pos = np.arange(len(configs))

        # Plot 1: Average TPS per GPU
        bars1 = ax1.bar(
            x_pos,
            avg_tps,
            BAR_WIDTH,
            yerr=std_tps,
            capsize=4,
            color=COLORS["primary_blue"],
            edgecolor=COLORS["dark_blue"],
            linewidth=1.2,
            alpha=0.8,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )

        ax1.set_xlabel("Configuration (Nodes/GPUs)", fontweight="bold")
        ax1.set_ylabel("Average Tokens/Second per GPU", fontweight="bold")
        ax1.set_title("Average Tokens per Second per GPU", fontweight="bold", pad=20)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(configs, rotation=0, ha="center")
        ax1.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)

        # Add value labels on bars
        for bar, val, std in zip(bars1, avg_tps, std_tps):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + height * 0.02,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

        # Plot 2: Total Global TPS
        bars2 = ax2.bar(
            x_pos,
            total_tps,
            BAR_WIDTH,
            color=COLORS["secondary_blue"],
            edgecolor=COLORS["dark_blue"],
            linewidth=1.2,
            alpha=0.8,
        )

        ax2.set_xlabel("Configuration (Nodes/GPUs)", fontweight="bold")
        ax2.set_ylabel("Total Global Tokens/Second", fontweight="bold")
        ax2.set_title("Total Global Throughput", fontweight="bold", pad=20)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(configs, rotation=0, ha="center")
        ax2.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars2, total_tps):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.02,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

        plt.tight_layout()

        if output_dir:
            plt.savefig(
                f"{output_dir}/tokens_per_second_comparison.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.show()

    def _plot_gpu_utilization_comparison(self, output_dir: str = None):
        """Plot GPU utilization comparison."""
        fig, ax = plt.subplots(figsize=(12, 7))

        configs = []
        gpu_utils = []
        gpu_stds = []

        for config_key, data in self.aggregated_data.items():
            num_nodes = data["config"]["num_nodes"]
            num_gpus = data["config"]["num_gpus_total"]
            config_label = f"{num_nodes}/{num_gpus}"

            configs.append(config_label)
            gpu_utils.append(data["avg_gpu_utilization"])
            gpu_stds.append(data["std_gpu_utilization"])

        # Sort by total GPUs
        sorted_data = sorted(
            zip(configs, gpu_utils, gpu_stds), key=lambda x: int(x[0].split("/")[1])
        )
        configs, gpu_utils, gpu_stds = zip(*sorted_data)

        x_pos = np.arange(len(configs))

        bars = ax.bar(
            x_pos,
            gpu_utils,
            BAR_WIDTH,
            yerr=gpu_stds,
            capsize=4,
            color=COLORS["primary_blue"],
            edgecolor=COLORS["dark_blue"],
            linewidth=1.2,
            alpha=0.8,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )

        ax.set_xlabel("Configuration (Nodes/GPUs)", fontweight="bold")
        ax.set_ylabel("Average GPU Utilization (%)", fontweight="bold")
        ax.set_title("GPU Utilization Across Configurations", fontweight="bold", pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=0, ha="center")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)

        # Add value labels
        for bar, val, std in zip(bars, gpu_utils, gpu_stds):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 1,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

        plt.tight_layout()

        if output_dir:
            plt.savefig(
                f"{output_dir}/gpu_utilization_comparison.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.show()

    def _plot_scaling_efficiency(self, output_dir: str = None):
        """Plot scaling efficiency analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            "Scaling Efficiency Analysis", fontsize=16, fontweight="bold", y=0.98
        )

        # Prepare data sorted by GPU count
        data_points = []
        for config_key, data in self.aggregated_data.items():
            num_gpus = data["config"]["num_gpus_total"]
            num_nodes = data["config"]["num_nodes"]
            total_tps = data["total_global_tps"]
            avg_tps = data["avg_tokens_per_second"]

            data_points.append((num_gpus, num_nodes, total_tps, avg_tps, config_key))

        data_points.sort(key=lambda x: x[0])  # Sort by GPU count

        if not data_points:
            print("No data points available for scaling analysis")
            return

        gpus = [dp[0] for dp in data_points]
        nodes = [dp[1] for dp in data_points]
        total_tps = [dp[2] for dp in data_points]
        avg_tps = [dp[3] for dp in data_points]

        # Plot 1: Total throughput scaling
        ax1.plot(
            gpus,
            total_tps,
            "o-",
            linewidth=3,
            markersize=8,
            color=COLORS["primary_blue"],
            markerfacecolor=COLORS["secondary_blue"],
            markeredgecolor=COLORS["dark_blue"],
            markeredgewidth=2,
            label="Actual",
        )

        if len(data_points) > 1:
            # Calculate ideal linear scaling from the first point
            base_tps_per_gpu = total_tps[0] / gpus[0]
            ideal_scaling = [gpu * base_tps_per_gpu for gpu in gpus]
            ax1.plot(
                gpus,
                ideal_scaling,
                "--",
                linewidth=2,
                alpha=0.7,
                color=COLORS["accent_coral"],
                label="Ideal Linear",
            )

        ax1.set_xlabel("Number of GPUs", fontweight="bold")
        ax1.set_ylabel("Total Throughput (tokens/sec)", fontweight="bold")
        ax1.set_title("Throughput Scaling vs GPU Count", fontweight="bold", pad=20)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

        # Plot 2: Per-GPU efficiency
        x_pos = np.arange(len(gpus))

        bars = ax2.bar(
            x_pos,
            avg_tps,
            BAR_WIDTH,
            color=COLORS["accent_coral"],
            alpha=0.8,
            edgecolor=COLORS["dark_blue"],
            linewidth=1.2,
        )

        ax2.set_xlabel("Configuration", fontweight="bold")
        ax2.set_ylabel("Tokens/Second per GPU", fontweight="bold")
        ax2.set_title("Per-GPU Efficiency", fontweight="bold", pad=20)

        # Create labels for x-axis
        config_labels = [f"{n}N/{g}G" for g, n in zip(gpus, nodes)]
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(config_labels, rotation=0, ha="center")
        ax2.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, avg_tps):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.02,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

        plt.tight_layout()

        if output_dir:
            plt.savefig(
                f"{output_dir}/scaling_efficiency.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.show()

    def _plot_detailed_metrics_heatmap(self, output_dir: str = None):
        """Create a heatmap showing various metrics across configurations."""
        # Prepare data for heatmap
        configs = []
        metrics_data = {
            "Avg TPS": [],
            "GPU Util (%)": [],
            "Mem Util (%)": [],
            "Total TPS": [],
            "Peak Mem (GB)": [],
        }

        for config_key, data in self.aggregated_data.items():
            num_nodes = data["config"]["num_nodes"]
            num_gpus = data["config"]["num_gpus_total"]
            config_label = f"{num_nodes}N/{num_gpus}G"

            configs.append(config_label)
            metrics_data["Avg TPS"].append(data["avg_tokens_per_second"])
            metrics_data["GPU Util (%)"].append(data["avg_gpu_utilization"])
            metrics_data["Mem Util (%)"].append(data["avg_memory_utilization"])
            metrics_data["Total TPS"].append(
                data["total_global_tps"] / 1000
            )  # Scale down for better visualization
            metrics_data["Peak Mem (GB)"].append(data["total_peak_memory"])

        # Create DataFrame and normalize data for heatmap
        df = pd.DataFrame(metrics_data, index=configs)

        # Normalize each column to 0-1 scale for better heatmap visualization
        df_normalized = df.copy()
        for col in df.columns:
            df_normalized[col] = (df[col] - df[col].min()) / (
                df[col].max() - df[col].min()
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            "Performance Metrics Analysis", fontsize=16, fontweight="bold", y=0.98
        )

        # Plot normalized heatmap
        sns.heatmap(
            df_normalized.T,
            annot=False,
            cmap="Blues",
            ax=ax1,
            cbar_kws={"label": "Normalized Value (0-1)"},
            linewidths=0.5,
            linecolor="white",
        )
        ax1.set_title("Normalized Metrics Heatmap", fontweight="bold", pad=20)
        ax1.set_xlabel("Configuration", fontweight="bold")
        ax1.set_ylabel("Metrics", fontweight="bold")

        # Plot raw values heatmap with annotations
        df_display = df.copy()
        df_display["Total TPS"] = (
            df_display["Total TPS"] * 1000
        )  # Convert back for display

        sns.heatmap(
            df_display.T,
            annot=True,
            fmt=".0f",
            cmap="RdYlBu_r",
            ax=ax2,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 9, "weight": "bold"},
        )
        ax2.set_title("Raw Metrics Values", fontweight="bold", pad=20)
        ax2.set_xlabel("Configuration", fontweight="bold")
        ax2.set_ylabel("Metrics", fontweight="bold")

        plt.tight_layout()

        if output_dir:
            plt.savefig(
                f"{output_dir}/metrics_heatmap.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.show()

    def _create_summary_table(self, output_dir: str = None):
        """Create and display summary table."""
        summary_data = []

        for config_key, data in self.aggregated_data.items():
            summary_data.append(
                {
                    "Configuration": f"{data['config']['num_nodes']}N/{data['config']['num_gpus_total']}G",
                    "Strategy": data["config"]["parallel_strategy"].upper(),
                    "Ranks": data["num_ranks"],
                    "Avg TPS/GPU": f"{data['avg_tokens_per_second']:.0f}",
                    "Total TPS": f"{data['total_global_tps']:.0f}",
                    "GPU Util (%)": f"{data['avg_gpu_utilization']:.1f}",
                    "Mem Util (%)": f"{data['avg_memory_utilization']:.1f}",
                    "Total Peak Mem (GB)": f"{data['total_peak_memory']:.1f}",
                    "Runtime (min)": f"{data['total_runtime'] / 60:.1f}",
                }
            )

        # Sort by total GPUs
        summary_data.sort(
            key=lambda x: int(x["Configuration"].split("/")[1].replace("G", ""))
        )

        df_summary = pd.DataFrame(summary_data)

        print("\n" + "=" * 100)
        print("DISTRIBUTED TRAINING PERFORMANCE SUMMARY")
        print("=" * 100)
        print(df_summary.to_string(index=False))
        print("=" * 100)

        if output_dir:
            df_summary.to_csv(f"{output_dir}/performance_summary.csv", index=False)
            print(f"\nSummary saved to: {output_dir}/performance_summary.csv")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze GPU utilization from distributed training profiling results"
    )
    parser.add_argument(
        "--profiling-dir",
        default="outputs/gpt/pretrain/profiling",
        help="Directory containing JSON profiling files",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gpt/pretrain/profiling/visualization",
        help="Directory to save plots and summary (optional)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.profiling_dir):
        print(f"Error: Directory {args.profiling_dir} does not exist")
        sys.exit(1)

    try:
        # Initialize analyzer
        analyzer = GPUUtilizationAnalyzer(args.profiling_dir)

        # Load and process data
        print("Loading profiling data...")
        analyzer.load_profiling_data()

        print("Aggregating data across ranks...")
        analyzer.aggregate_data()

        print("Creating visualizations...")
        analyzer.create_summary_plots(args.output_dir)

        print("\nAnalysis complete!")

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
