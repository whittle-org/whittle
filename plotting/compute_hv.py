# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import os
import pandas as pd
import numpy as np
import json

from pathlib import Path
from collections import defaultdict
from itertools import product

from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer
from pygmo import hypervolume

from syne_tune.experiments import load_experiment
from syne_tune.constants import ST_TUNER_TIME

# from syne_tune.blackbox_repository.repository import load
from nas_fine_tuning.task_data import GLUE_TASK_INFO


class HyperVolume:
    def __init__(self, ref_point):
        self.ref_point = ref_point

    def __call__(self, points):
        return hypervolume(points).compute(self.ref_point)


experiment = "weight_sharing_v9"
base_path = Path(f"/Users/kleiaaro/experiments/nas_search_result/{experiment}")
# methods = [
#     "morea",
#     "random_search",
#     "local_search",
#     "nsga2",
#     "lsbo",
#     "rsbo",
#     "ehvi",
# ]
methods = ["random_search", "morea", "ehvi", "local_search", "nsga2"]

checkpoints = ["linear_random", "sandwich", "one_shot", "standard", "random", "kd"]
epochs = [5, 10, 20, 40][:1]
models = ["bert-base-cased", "roberta-base"]
labels = checkpoints

ref_point = [2, 2]
seeds = np.arange(10)
random_sub_nets = [1, 2]
runs = np.arange(1)
# datasets = ["rte", "mrpc", "cola", "stsb", "sst2", "qnli", 'imdb', 'swag', "mnli", "qqp"][:6]
datasets = ["sst2", "qnli", "swag", "stsb", "cola", "mrpc", "imdb", "rte"]
search_spaces = [
    "layer",
    "small",
    "medium",
    "uniform",
    "meta_small_kde_1_tasks",
    "meta_small_kde_5_tasks",
][:-2]
# search_spaces = ["small"]

runtimes = {
    # "rte": np.linspace(110, 400, 100),
    "rte": np.linspace(150, 400, 100),
    "mrpc": np.linspace(150, 500, 100),
    "cola": np.linspace(205, 1000, 100),
    "stsb": np.linspace(200, 1000, 100),
    "sst2": np.linspace(150, 1000, 100),
    "imdb": np.linspace(500, 10000, 100),
    "swag": np.linspace(500, 5000, 100),
    "qnli": np.linspace(200, 2000, 100),
    # "qnli": [50 * i for i in range(4, 61)],
    # "mnli": [50 * i for i in range(4, 61)],
    # "qqp": [50 * i for i in range(4, 61)],
}
data = defaultdict(list)
hv = HyperVolume(ref_point=ref_point)
oracle_performance = dict()

print("collect all data")

for (
    dataset,
    search_space,
    model,
    epoch,
    checkpoint,
    random_sub_net,
    method,
    seed,
    run,
) in product(
    datasets,
    search_spaces,
    models,
    epochs,
    checkpoints,
    random_sub_nets,
    methods,
    seeds,
    runs,
):
    path = (
        base_path
        / search_space
        / model
        / f"epochs_{epoch}"
        / dataset
        / checkpoint
        / f"random_sub_nets_{random_sub_net}"
        / f"seed_{seed}"
        / method
        / f"run_{run}"
    )

    if not os.path.exists(path):
        print(f"{path} is missing")
        continue
    d = json.load(open(path / f"results_{dataset}.json"))
    N = len(d["params"])
    print(
        f"Results {dataset} {method} {checkpoint} {seed} {run}: N={N}, T-min={np.min(d['runtime'])} T-Max={np.max(d['runtime'])}"
    )
    for row in range(N):
        # data["runtime"].append(row)
        # if row == 0:
        #     print(dataset, d['runtime'][row])
        data["runtime"].append(d["runtime"][row])
        data["dataset"].append(dataset)
        data["method"].append(method)
        data["checkpoint"].append(checkpoint)
        data["seed"].append(seed)
        data["model"].append(model)
        data["epoch"].append(epoch)
        data["search_space"].append(search_space)
        data["run_id"].append(run)
        data["random_sub_net"].append(random_sub_net)
        data[f"objective_0"].append(d["error"][row])
        data[f"objective_1"].append(d["params"][row])

data = pd.DataFrame(data)
print("start Quantile normalization")
for dataset, df in data.groupby("dataset"):
    print(f"normalize benchmark: {dataset}")

    for i in range(2):
        qt = QuantileTransformer()
        transformed_objective = qt.fit_transform(
            df.loc[:, f"objective_{i}"].to_numpy().reshape(-1, 1)
        )
        mask = data["dataset"] == dataset
        data.loc[mask, f"objective_{i}"] = transformed_objective

print("finished data normalization")
final_results = defaultdict(list)

for dataset, df in data.groupby("dataset"):
    print(f"process dataset: {dataset}")

    # compute hypervolume
    for keys, sub_df in df.groupby(
        [
            "model",
            "search_space",
            "method",
            "epoch",
            "checkpoint",
            "random_sub_net",
            "seed",
            "run_id",
        ]
    ):
        model = keys[0]
        search_space = keys[1]
        method = keys[2]
        epoch = keys[3]
        checkpoint = keys[4]
        random_sub_net = keys[5]
        seed = keys[6]
        run_id = keys[7]

        for runtime in runtimes[dataset]:
            split = sub_df[sub_df["runtime"] <= runtime]
            points = np.empty((len(split), 2))
            for i in range(2):
                points[:, i] = split[f"objective_{i}"]

            y = hv(points)
            # runtime = list(split['runtime'])[-1]
            final_results["method"].append(method)
            final_results["model"].append(model)
            final_results["checkpoint"].append(checkpoint)
            final_results["seed"].append(seed)
            final_results["epoch"].append(epoch)
            final_results["dataset"].append(dataset)
            final_results["runtime"].append(runtime)
            final_results["hv"].append(y)
            final_results["search_space"].append(search_space)
            final_results["random_sub_net"].append(random_sub_net)
            final_results["run_id"].append(run_id)

final_results = pd.DataFrame(final_results)
final_results.to_csv(f"{experiment}.csv")
