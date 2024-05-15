import json

import os
import time
import logging
import sys

from dataclasses import dataclass, field

import numpy as np
import torch
import datasets
import transformers

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from evaluate import load
from functools import partial

from lobotomy.search import multi_objective_search

from bert import SuperNetBertForSequenceClassification
from estimate_efficency import compute_parameters
from benchmarks.plm_pruning.data_wrapper.task_data import GLUE_TASK_INFO
from search_spaces import (
    SmallSearchSpace,
    MediumSearchSpace,
    LayerSearchSpace,
    FullSearchSpace,
)
from hf_args import DataTrainingArguments, ModelArguments, parse_model_name
from data_wrapper import Glue, IMDB, SWAG
from model_data import get_model_data


SEARCHSPACES = {
    "small": SmallSearchSpace,
    "medium": MediumSearchSpace,
    "layer": LayerSearchSpace,
    "uniform": FullSearchSpace,
    "smallpower2": partial(SmallSearchSpace, power_of_2_encoding=True),
}

logger = logging.getLogger(__name__)


@dataclass
class SearchArguments:
    """
    Arguments to define the search
    """

    if "SM_CHANNEL_MODEL" in os.environ:
        checkpoint_dir_model: str = field(
            metadata={"help": ""}, default=os.environ["SM_CHANNEL_MODEL"]
        )
    else:
        checkpoint_dir_model: str = field(
            metadata={"help": ""}, default="/home/ubuntu/seed_42/"
        )

    search_strategy: str = field(metadata={"help": ""}, default="random_search")
    search_space: str = field(metadata={"help": ""}, default="small")
    use_accelerate: bool = field(metadata={"help": ""}, default=False)
    num_samples: int = field(default=500)
    log_dir: str = field(metadata={"help": ""}, default="./tensorboard_log_dir")
    optimize_memory_footprint: bool = field(metadata={}, default=False)


def main():
    start_time = time.time()
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, SearchArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        search_args,
    ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.

    if int(training_args.seed) == -1:
        training_args.seed = np.random.randint(2**32 - 1)
    set_seed(training_args.seed)

    model_type = parse_model_name(model_args)

    st = time.time()
    # Load validation data
    if data_args.task_name in GLUE_TASK_INFO:
        data = Glue(
            training_args=training_args, model_args=model_args, data_args=data_args
        )
        metric = load("glue", data_args.task_name)
        metric_name = GLUE_TASK_INFO[data_args.task_name]["metric"]
    elif data_args.task_name == "imdb":
        data = IMDB(
            training_args=training_args, model_args=model_args, data_args=data_args
        )
        metric = load("accuracy")
        metric_name = "accuracy"
    elif data_args.task_name == "swag":
        data = SWAG(
            training_args=training_args, model_args=model_args, data_args=data_args
        )
        metric = load("accuracy")
        metric_name = "accuracy"
    # elif data_args.task_name == "imdb":
    #     data = Imdb(training_args=training_args, model_args=model_args, data_args=data_args)
    # elif data_args.task_name == "custom":
    #     data = Custom(training_args=training_args, model_args=model_args, data_args=data_args)
    _, eval_dataloader, test_dataloader = data.get_data_loaders()

    is_regression = data_args.task_name == "stsb"

    data_loading_time = time.time() - st

    st = time.time()

    if data_args.task_name in ["swag"]:
        pass
    else:
        if model_type.startswith("bert"):
            model_cls = SuperNetBertForSequenceClassification

    model = model_cls.from_pretrained(search_args.checkpoint_dir_model)
    model_data = get_model_data(model)

    attention_head_size = model_data["attention_head_size"]
    n_params_emb = model_data["n_params_emb"]
    n_params_classifier = model_data["n_params_classifier"]
    attention_size = model_data["attention_size"]
    n_params_super_net = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_loading_time = time.time() - st

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    def evaluate_masks(config, dataloader):
        head_mask, ffn_mask = search_space.config_to_mask(config)
        n_params_model = compute_parameters(
            dmodel=attention_size,
            dhead=attention_head_size,
            num_heads_per_layer=head_mask.sum(dim=1),
            num_neurons_per_layer=ffn_mask.sum(dim=1),
        )
        n_params = n_params_emb + n_params_model + n_params_classifier

        model.select_sub_network(config)
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(batch)

            logits = outputs.logits
            predictions = (
                torch.squeeze(logits) if is_regression else torch.argmax(logits, dim=-1)
            )

            metric.add_batch(predictions=predictions, references=batch["labels"])

        eval_metric = metric.compute()

        return 1 - eval_metric[metric_name], n_params / n_params_super_net

    kwargs = {"rng": np.random.RandomState(seed=training_args.seed)}
    search_space = SEARCHSPACES[search_args.search_space](model.config, **kwargs)

    if search_args.optimize_memory_footprint:
        metrics = ["error", "memory"]
    else:
        metrics = ["error", "params"]

    search_results = multi_objective_search(
        objective=evaluate_masks,
        search_space=search_space.config_space,
        objective_kwargs={"dataloader": eval_dataloader},
        num_samples=search_args.num_samples,
        search_strategy=search_args.search_strategy,
        seed=training_args.seed,
    )

    idx = search_results["is_pareto_optimal"]

    os.makedirs(training_args.output_dir, exist_ok=True)
    test_error = []
    model.eval()
    for i, config in enumerate(search_results["configs"]):
        error, n_params = evaluate_masks(config, dataloader=test_dataloader)
        test_error.append(float(error))

    results = dict()
    results["dataset"] = data_args.task_name
    results["error"] = list(search_results["costs"][:, 0])
    results["test_error"] = test_error
    results["params"] = list(search_results["costs"][:, 1])
    results["params_pareto"] = list(search_results["costs"][idx, 1])

    results["test_pareto"] = [test_error[i] for i in idx]
    results["config"] = search_results["configs"]
    results["eval_pareto"] = list(search_results["costs"][idx, 0])
    results["model_loading_time"] = model_loading_time
    results["data_loading_time"] = data_loading_time
    results["runtime"] = list(search_results["runtime"])
    results["indices"] = list(idx)
    print(results)

    fname = os.path.join(
        training_args.output_dir, f"results_{data_args.task_name}.json"
    )
    json.dump(results, open(fname, "w"))


if __name__ == "__main__":
    main()
