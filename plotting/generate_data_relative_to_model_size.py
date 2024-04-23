import itertools
import pandas
import numpy as np

from functools import partial
from collections import defaultdict

from load_nas_data import load_nas_data
from load_ld_data import load_ld_data
from load_standard_nas_data import load_standard_nas_data
from load_rfp_data import load_rfp_data
from load_cofi_data import load_cofi_data
from load_hp_data import load_hp_data

"""
Threshold on model size: Min error of model where #params >= threshold * #params_unpruned_model after runtime

Method | Task | Model | Seed | Runtime | Threshold | Test Error | Valid Error

"""

experiment_version = "v9"
experiment = f"weight_sharing_{experiment_version}"
epochs = 5
pruning_methods = ["nas", "ld", "standard_nas", "rfp", "hp", "meta-nas", "cofi"][:-2]
# pruning_methods = ['nas', 'ld']
# models = ["bert-base-cased"]
models = ["roberta-base", "bert-base-cased"]
seeds = np.arange(10)
# datasets = ["rte", "mrpc", "cola", "stsb", "sst2", "qnli", "mnli", "qqp"][:-3]
datasets = [
    "rte",
    "mrpc",
    "cola",
    "stsb",
    "sst2",
    "qnli",
    "swag",
    "imdb",
    "mnli",
    "qqp",
][:-2]
# datasets = ['swag']

thresholds = np.linspace(0.2, 0.99, 21)

# runtimes = [3600 * i for i in range(3, 10)]
runtimes = {
    "rte": [600, 1800, 3600, 7200][-1:],
    "mrpc": [600, 1800, 3600, 7200][-1:],
    "cola": [1800, 3600, 7200][-1:],
    "stsb": [1800, 3600, 7200][-1:],
    "sst2": [3600, 7200, 3600 * 4][-1:],
    "qnli": [3600, 7200, 3600 * 4][-1:],
    "imdb": [3600, 7200, 3600 * 4][-1:],
    "swag": [3600, 7200, 3600 * 4][-1:],
    "mnli": [3600, 7200, 3600 * 4, 3600 * 16][-1:],
    "qqp": [3600, 7200, 3600 * 4, 3600 * 16][-1:],
}
search_method_weight_sharing_nas = "ehvi"
# search_method_weight_sharing_nas = 'random_search'

original_model_size = {
    "bert-base-cased": 108311810,
    "gpt2": 124441344,
    "gpt2-medium": 354825216,
    "roberta-base": 124647170,
}

data = defaultdict(list)

load_dataset_routines = {
    "nas": partial(
        load_nas_data,
        experiment=experiment,
        method=search_method_weight_sharing_nas,
        search_space="small",
        epochs=epochs,
        checkpoint="one_shot",
        random_sub_nets=2,
    ),
    "rfp": partial(load_rfp_data, epochs=epochs),
    "hp": partial(load_hp_data, epochs=epochs),
    "ld": partial(load_ld_data, epochs=epochs),
    "cofi": partial(load_cofi_data, epochs=epochs),
    "standard_nas": partial(
        load_standard_nas_data,
        # method="local_search_upper_bound",
        method="moasha",
        search_space="small",
        epochs=epochs,
    ),
    "meta-nas": partial(
        load_nas_data,
        experiment=experiment,
        method=search_method_weight_sharing_nas,
        search_space="meta_small_kde",
        epochs=epochs,
        checkpoint="one_shot",
        random_sub_nets=2,
    ),
}
for method, model, dataset, seed in itertools.product(
    pruning_methods, models, datasets, seeds
):
    (
        runtime_traj,
        valid_error_traj,
        test_error_traj,
        params_traj,
    ) = load_dataset_routines[method](dataset=dataset, seed=seed, model=model)
    if runtime_traj is None:
        continue
    # if seed == 0:
    print(
        f"{method}, {dataset}, min runtime={runtime_traj.min()}, max runtime={runtime_traj.max()}, N={len(valid_error_traj)}"
    )
    for runtime in runtimes[dataset]:
        idx = np.where(runtime_traj < runtime)[0]
        objective_0 = params_traj[idx]
        objective_1 = valid_error_traj[idx]
        test_error = test_error_traj[idx]

        for threshold in thresholds:
            smaller_than_threshold = np.where(
                objective_0 < threshold * original_model_size[model]
            )[0]
            if len(smaller_than_threshold) == 0:
                # print(dataset, method, runtime, threshold)
                continue
            best_idx = np.argmin(objective_1[smaller_than_threshold])

            data["runtime"].append(runtime)
            data["threshold"].append(threshold)
            data["test_error"].append(test_error[smaller_than_threshold][best_idx])
            data["valid_error"].append(objective_1[smaller_than_threshold][best_idx])
            data["method"].append(method)
            data["model"].append(model)
            data["dataset"].append(dataset)
            data["seed"].append(seed)

data = pandas.DataFrame(data)
data.to_csv("data_relative_to_model_size.csv")
print("Successful!")
