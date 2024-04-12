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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams["text.usetex"] = True
rcParams["font.family"] = "sans"

experiment = "weight_sharing_v9"

method = "random_search"
# model = 'roberta-base'
model = "bert-base-cased"
# model = 'gpt2'
checkpoint = "one_shot"
epochs = 5
random_sub_net = 2
df = pd.read_csv(f"{experiment}.csv")
df = df.query(
    f"model == '{model}' & checkpoint == '{checkpoint}' & method == '{method}' "
    f"& epoch == {epochs} & random_sub_net == {random_sub_net}"
)
labels = {
    "small": "Small",
    # "medium": "Medium",
    # "layer": "Layer",
    # "uniform": "Large",
    "meta_small_kde_1_tasks": "meta-1",
    "meta_small_kde_5_tasks": "meta-5",

}

marker = ["o", "x", "s", "d", "p", "P", "^", "v", "<", ">"]
for dataset, df_benchmark in df.groupby("dataset"):
    plt.figure(dpi=200)
    names, vals, xs = [], [], []
    search_space_names = []
    mi = 0
    for search_space, df_search_space in df_benchmark.groupby("search_space"):
        if search_space not in labels:
            continue
        max_runtime = df_search_space.runtime.max()
        y = df_search_space[df_search_space["runtime"] == max_runtime]["hv"]
        vals.append(y)
        names.append(search_space)
        xs.append(np.random.normal(mi + 1, 0.04, y.shape[0]))
        print(search_space, dataset, len(y))
        search_space_names.append(search_space)
        mi += 1
    plt.boxplot(vals, labels=names)
    palette = [f"C{i}" for i in range(len(names))]
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c)
    plt.xticks(np.arange(1, 1 + len(names)), names)
    plt.ylabel("hypervolume", fontsize=20)
    plt.xlabel("search space", fontsize=20)
    plt.title(f"{dataset.upper()}", fontsize=20)
    plt.grid(linewidth="1", alpha=0.4)
    plt.xticks(
        np.arange(1, 1 + len(search_space_names)),
        [labels[c] for c in search_space_names],
        fontsize=15,
    )

    plt.savefig(
        f"./figures/hypervolume_search_spaces_{dataset}_{model}.pdf",
        bbox_inches="tight",
    )
    plt.show()
