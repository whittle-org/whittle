import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Any


def load_sub_network_results(eval_path: Path) -> List[dict[str, Any]]:
    metrics_path = eval_path / "metrics.json"
    results_path = eval_path / "results.json"

    metrics = json.loads(metrics_path.read_text())
    results = json.loads(results_path.read_text())
    sub_network_results = []

    # process metrics - parameters, flops, latency
    for m_key, m_value in metrics.items():
        sub_network_results.append(
            {"task": "_metrics", "metric": m_key, "score": m_value}
        )

    # process lm_eval_harness results
    tasks = results["results"]
    for task, task_metrics in tasks.items():
        for m_key, m_value in task_metrics.items():
            sub_network_results.append(
                {"task": task, "metric": m_key.replace(",none", ""), "score": m_value}
            )

    return sub_network_results


def setup(
    results_dir: Path,
    output_path: Optional[Path] = None,
    pareto_path: Optional[Path] = None,
) -> None:
    """
    Load and process evaluation results from a directory containing sub-network evaluation results.

    Arguments:
        results_dir: The path to the directory containing sub-network evaluation results.
        output_path: The path to save the processed results. If not provided, saving to `results_dir/sub_network_results.json` by default.
        pareto_path: The path to the pareto front results. If provided, the processed results will contain an indicator if the sub-network is in the pareto front.
    """

    if output_path is None:
        output_path = results_dir / "sub_network_results.csv"

    pareto_front = None
    if pareto_path is not None:
        pareto_front = json.loads(pareto_path.read_text())
        pareto_front = [str(Path(p).absolute()) for p in pareto_front]

    full_results = []
    res_iter = tqdm(results_dir.iterdir())
    for eval_path in res_iter:
        res_iter.set_description(f"Processing {eval_path.name}")
        if eval_path.is_dir():
            sub_network_name = eval_path.name
            try:
                sub_network_results = load_sub_network_results(eval_path / "eval")
                # every metric of the network is a new row
                for sub_res in sub_network_results:
                    entry = {"sub_network": sub_network_name, **sub_res}
                    if pareto_front is not None:
                        entry["pareto"] = (
                            str(eval_path.absolute() / "lit_model.pth") in pareto_front
                        )

                    full_results.append(entry)

            except FileNotFoundError:
                print(
                    f"Skipping {eval_path} as it does not contain evaluation results."
                )
                continue

    results = pd.DataFrame(full_results)
    results.to_csv(output_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)
