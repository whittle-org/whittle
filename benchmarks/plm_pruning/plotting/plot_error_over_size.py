import pandas
import numpy as np
import matplotlib.pyplot as plt

from parse_data.load_distillation import load_distillation


data = pandas.read_csv("./parse_data/data_relative_to_model_size.csv")
suffix = ""
# data = pandas.read_csv('./parse_data/data_relative_to_model_size_standard_nas.csv')
# suffix = "_standard_nas"
# data = pandas.read_csv("./parse_data/data_relative_to_model_size_ws_nas.csv")
# suffix = "_ws_nas"
num_runs = 10
epochs = 5
# model = "bert-base-cased"
model = "roberta-base"
# model = 'gpt2'
data = data[data["model"] == model]


data_unpruned = pandas.read_csv("./parse_data/data_unpruned_model.csv")
data_unpruned = data_unpruned[data_unpruned["model"] == model]

plot_distillation = False

colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
marker = ["x", "o", "D", "s", "v", "^"]
ylims = {
    # "mrpc": (0.07, 0.25),
    # "stsb": (0.05, 0.5),
    # "sst2": (0.05, 0.2),
    # "imdb": (0.075, 0.25),
    "mrpc": (0.8, 0.93),
    "stsb": (0.5, 0.95),
    "sst2": (0.8, 0.95),
    "imdb": (0.75, 0.925),
    "mnli": (0.5, 0.9),
    "qqp": (0.5, 0.9),
}
labels = {
    "hp": "HP",
    "ld": "LD",
    "nas": "WS-NAS",
    "standard_nas": "S-NAS",
    "rfp": "RFP",
}
for dataset, df_data in data.groupby("dataset"):
    for runtime, df_runtime in df_data.groupby("runtime"):
        plt.figure(dpi=200)
        print(dataset, runtime)
        for i, (method, df_method) in enumerate(df_runtime.groupby("method")):
            print(method)
            mean_performance = []
            std_performance = []
            thresholds = []
            for j, (threshold, df_threshold) in enumerate(
                df_method.groupby("threshold")
            ):
                performance = []
                for seed, df_seed in df_threshold.groupby("seed"):
                    performance.append(1 - df_seed["test_error"].iloc[0])

                # if len(loss) < num_runs:
                #     print(threshold)
                #     continue
                # plt.scatter([threshold] * len(loss), loss, color=colors[i])
                mean_performance.append(np.mean(performance))
                std_performance.append(np.std(performance))
                thresholds.append(1 - threshold)

            if len(mean_performance) > 0:
                plt.errorbar(
                    thresholds,
                    mean_performance,
                    yerr=std_performance,
                    marker=marker[i],
                    markersize=8,
                    label=labels[method],
                    color=colors[i],
                    linestyle="--",
                )
            # for k in range(all_loss.shape[1]):
            #     plt.scatter(params_grid, all_loss[:, k], marker='.',
            #                  color=colors[i])

        if plot_distillation:
            (
                mean_loss,
                std_loss,
                relative_parameters,
                distillation_model,
            ) = load_distillation(dataset, model, num_runs, epochs)
            plt.errorbar(
                relative_parameters,
                1 - mean_loss,
                yerr=std_loss,
                marker="D",
                markersize=8,
                label=distillation_model,
                color="black",
                linestyle="--",
            )
        # plt.xscale('log')
        plt.grid(linewidth="1", alpha=0.4)
        plt.legend(loc=3)

        sub_data_unpruned = data_unpruned.query(f"dataset == '{dataset}'")
        y_star = 1 - np.mean(sub_data_unpruned["test_error"])
        plt.axhline(y_star, linestyle="--", color="black", alpha=0.5)
        plt.axhline(0.95 * y_star, linestyle="--", color="black", alpha=0.3)
        plt.axhline(0.90 * y_star, linestyle="--", color="black", alpha=0.1)

        # plt.xlabel("retained number of parameters", fontsize=15)
        plt.xlabel("number of parameters pruned", fontsize=15)
        # plt.ylabel("relative loss in test performance", fontsize=15)
        plt.ylabel("test performance")
        if dataset in ylims:
            plt.ylim(ylims[dataset])
        thresholds = np.linspace(0.05, 0.7, 15)
        plt.xticks(thresholds[::2], [f"{int(xi * 100)}%" for xi in thresholds][::2])
        # plt.title(f"{dataset.upper()} ({runtime} seconds)", fontsize=15)
        plt.title(f"{dataset.upper()}", fontsize=15)

        plt.savefig(
            f"./figures/parameter_efficiency_{model}_{dataset}{suffix}.pdf",
            bbox_inches="tight",
        )
        plt.show()
