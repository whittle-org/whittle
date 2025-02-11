import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Optional


def setup(
    results_dir: Path,
    plot_dir: Optional[Path] = None,
    metric: str = "parameters",
    tasks: Optional[str] = None,
    task_metric: Optional[str] = None,
) -> None:
    """
    Plot the results of a sub-network search.

    Arguments:
        results_dir: The path to the directory containing the sub-network search results.
        plot_dir: Directory in which to save the plots. If not provided, saving to `results_dir/plot` by default.
        metric: The metric to use for the x-axis of the plot.
        tasks: Task names to evaluate. Example: "hellaswag,mmlu"
        task_metric: The metric to use for the y-axis of the plot. If not provided, the first metric from unique metrics is used.
    """
    results_path = results_dir / "sub_network_results.csv"
    if plot_dir is None:
        plot_dir = results_dir / "plot"
        plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path, index_col=0)

    if tasks is None:
        task_list = [t for t in df["task"].unique() if not t.startswith("_")]
    else:
        task_list = tasks.split(",")

    x_metric_df = df[df["metric"] == metric]
    x_metric_df = (
        x_metric_df.set_index("sub_network")
        .drop(columns=["task", "metric"])
        .rename(columns={"score": metric})
    )

    for task in task_list:
        task_df = df[df["task"] == task]
        if task_metric is None:
            possible = task_df["metric"].unique().tolist()
            if "acc_norm" in possible:
                task_col = "acc_norm"
            else:
                task_col = possible[0]
        else:
            task_col = task_metric

        task_df = task_df[task_df["metric"] == task_col]

        y_name = f"{task} {task_col}"
        drop_cols = (
            ["task", "metric", "pareto"]
            if "pareto" in task_df.columns
            else ["task", "metric"]
        )
        task_df = (
            task_df.set_index("sub_network")
            .drop(columns=drop_cols)
            .rename(columns={"score": y_name})
        )

        results = x_metric_df.join(task_df, how="inner")
        print(f"Plotting {len(results)} sub-networks, task: {task}, metric: {metric}")

        plt.figure()
        sns.scatterplot(
            data=results,
            x=metric,
            y=y_name,
            hue="pareto" if "pareto" in results.columns else None,
        )
        plt.title(f"{task} {task_col} vs {metric}")
        plt.savefig(plot_dir / f"{task}_{task_col}_vs_{metric}.png")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)
