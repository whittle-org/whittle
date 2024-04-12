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

# rcParams["text.usetex"] = True
rcParams["font.family"] = "sans"

experiment = "weight_sharing_v9"

method = "random_search"
search_space = "small"
model = "bert-base-cased"
# model = 'roberta-base'
epochs = 5
df = pd.read_csv(f"{experiment}.csv")
df = df.query(
    f"search_space == '{search_space}' & epoch == {epochs} "
    f"& method == '{method}' & model == '{model}'"
)
config = {
    "standard": {"label": "standard", "random_sub_nets": 1},
    "random": {"label": "random", "random_sub_nets": 1},
    "linear_random": {"label": "linear", "random_sub_nets": 1},
    "sandwich": {"label": "sandwich", "random_sub_nets": 2},
    "kd": {"label": "inplace-kd", "random_sub_nets": 2},
    "one_shot": {"label": "full", "random_sub_nets": 2},
    "ats": {"label": "ats", "random_sub_nets": 2},

}
marker = ["o", "x", "s", "d", "p", "P", "^", "v", "<", ">"]
checkpoint_names = ["standard", "random", "linear_random", "sandwich", "one_shot", "kd", 'ats']
for dataset, df_benchmark in df.groupby("dataset"):
    plt.figure(dpi=200)
    # checkpoint_names, vals, xs = [], [], []
    vals, xs = [], []
    # for mi, (checkpoint, df_checkpoint) in enumerate(df_benchmark.groupby("checkpoint")):
    for mi, checkpoint in enumerate(checkpoint_names):
        print(mi, checkpoint)
        random_sub_net = config[checkpoint]["random_sub_nets"]

        df_checkpoint = df_benchmark.query(
            f"checkpoint == '{checkpoint}' & random_sub_net == {random_sub_net}"
        )
        max_runtime = df_checkpoint.runtime.max()
        y = df_checkpoint[df_checkpoint["runtime"] == max_runtime]["hv"]
        # plt.boxplot(y, positions=[mi])
        vals.append(y)
        # checkpoint_names.append(checkpoint)
        xs.append(np.random.normal(mi + 1, 0.04, y.shape[0]))
    labels = [config[checkpoint]["label"] for checkpoint in checkpoint_names]
    plt.boxplot(vals)
    palette = [f"C{i}" for i in range(len(checkpoint_names))]
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c)
    plt.xticks(
        np.arange(1, 1 + len(checkpoint_names)),
        labels,
        fontsize=15,
    )
    plt.ylabel("hypervolume", fontsize=20)

    plt.title(f"{dataset.upper()}", fontsize=20)
    plt.xlabel("super-network training strategy", fontsize=20)
    plt.grid(linewidth="1", alpha=0.4)
    plt.savefig(
        f"./figures/hypervolume_checkpoints_{dataset}_{model}.pdf",
        bbox_inches="tight",
    )
    plt.show()
