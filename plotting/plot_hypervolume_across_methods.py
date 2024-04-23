import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams
from compute_ranks import compute_ranks


rcParams["text.usetex"] = False
rcParams["font.family"] = "sans"

experiment = "weight_sharing_v9"
df = pd.read_csv(f"{experiment}.csv")

checkpoint = "one_shot"
search_space = "small"
epochs = 5
random_sub_net = 2
df = df.query(
    f"search_space == '{search_space}' & checkpoint == '{checkpoint}' & epoch == {epochs} & random_sub_net == {random_sub_net}"
)
method_info = {
    "morea": {"label": "MO-REA", "color": "C6"},
    "random_search": {"label": "RS", "color": "C0"},
    # 'local_search_random': {'label': 'LS-R', 'color': 'C4'},
    "local_search": {"label": "LS", "color": "C1"},
    # 'local_search_lower_bound': {'label': 'LS-L', 'color': 'C5'},
    "nsga2": {"label": "NSGA-2", "color": "C3"},
    "moasha": {"label": "MO-ASHA", "color": "C6"},
    "lsbo": {"label": "LS-BO", "color": "C2"},
    "rsbo": {"label": "RS-BO", "color": "C5"},
    "ehvi": {"label": "EHVI", "color": "black"},
}
ylims = {}
ylims["bert-base-cased"] = {
    "rte": (0.2, 0.4),
    "stsb": (0.3, 0.7),
    "cola": (0.3, 0.6),
    "mrpc": (0.2, 0.6),
    "swag": (0.3, 0.6),
    "sst2": (0.2, 0.4),
    "imdb": (0.4, 0.6),
    "qnli": (0.3, 0.6),
    "mnli": (0.0, 0.7),
    "qqp": (0.0, 1.7),
}
ylims["roberta-base"] = {
    "rte": (0.6, 0.8),
    "stsb": (0.75, 1.0),
    "cola": (0.8, 0.925),
    "mrpc": (0.8, 1.0),
    "swag": (0.75, 0.91),
    "sst2": (0.85, 1.0),
    "imdb": (0.7, 0.9),
    "qnli": (0.7, 0.9),
    "mnli": (0.0, 0.7),
    "qqp": (0.0, 1.7),
}
marker = ["o", "x", "s", "d", "p", "P", "^", "v", "<", ">"]
methods = []

for model, df_model in df.groupby("model"):
    if model == "bert-base-cased":
        continue

    n_runs = 9
    n_iters = 100
    n_methods = len(df["method"].unique())
    n_tasks = len(df["dataset"].unique())

    error = np.empty((n_methods, n_tasks, n_runs, n_iters))

    for di, (dataset, df_benchmark) in enumerate(df_model.groupby("dataset")):
        plt.figure(dpi=200)
        for mi, (method, df_method) in enumerate(df_benchmark.groupby("method")):
            if method not in method_info:
                continue

            methods.append(method)
            traj = []

            for checkpoint_seed, df_seeds in df_method.groupby("seed"):
                for checkpoint_run, df_run in df_seeds.groupby("run_id"):
                    traj.append(list(df_run.sort_values(by="runtime")["hv"]))

            runtimes = df_method.runtime.unique()

            traj = 4 - np.array(traj)
            error[mi, di] = traj[:n_runs, :]
            print(dataset, model, method, traj.shape)
            mean_prediction = np.mean(traj, axis=0)
            variance_prediction = np.mean(
                (traj - mean_prediction) ** 2 + np.var(mean_prediction, axis=0), axis=0
            )
            plt.errorbar(
                runtimes,
                mean_prediction,
                yerr=variance_prediction,
                color=method_info[method]["color"],
                marker=marker[mi],
                fillstyle="none",
                label=method_info[method]["label"],
                linestyle="-",
                markersize=1,
                markeredgewidth=1.5,
            )

        # for runtime, l in list(df_method.groupby(['runtime'])['hv']):
        #     plt.scatter([runtime] * len(l), l, color='C%i' % mi, alpha=0.4, s=20)

        plt.legend()
        plt.ylabel("regret hypervolume", fontsize=20)
        plt.xlabel("runtime (seconds)", fontsize=20)
        # plt.ylim(3.8, 4)
        plt.title(f"{dataset.upper()}", fontsize=20)
        # plt.xscale('log', base=2)
        # plt.xticks(list(runtimes), list(runtimes))

        # if 'ablation' in experiment:
        #     plt.savefig(f"/Users/kleiaaro/git/SyneDocs/AaronK/auto_fine_tuning/figures/hypervolume_ablation_{benchmark}.pdf")
        # else:
        plt.grid(linewidth="1", alpha=0.4)
        print(dataset, model, checkpoint)
        plt.ylim(ylims[model][dataset])
        # plt.yscale('log')
        plt.savefig(
            f"./figures/hypervolume_search_{dataset}_{model}_{checkpoint}.pdf",
            bbox_inches="tight",
        )
        plt.show()

    ranks = compute_ranks(error)
    for i in range(ranks.shape[0]):
        method = methods[i]
        plt.plot(
            ranks[i],
            marker="o",
            label=method_info[method]["label"],
            color=method_info[method]["color"],
            linestyle="--",
        )

    plot_label = False

    plt.grid(linewidth="1", alpha=0.4)
    plt.legend(loc=1)
    plt.title(model.replace("_", "-").upper())
    plt.xlabel("time steps", fontsize=20)
    plt.ylabel("average rank", fontsize=15)
    # plt.xticks(thresholds, [f"{int(xi * 100)}%" for xi in thresholds])
    # plt.savefig(f"./figures/ranks_ws_nas_search_{model}.pdf", bbox_inches="tight")
    plt.show()
