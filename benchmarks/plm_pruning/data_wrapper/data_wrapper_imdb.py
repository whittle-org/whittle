import logging

from datasets import load_dataset

from .data_wrapper import DataWrapper

logger = logging.getLogger(__name__)


class IMDB(DataWrapper):
    def _load_data(self):
        raw_datasets = load_dataset("imdb", cache_dir=self.model_args.cache_dir)

        def preprocess_function(examples):
            # Tokenize the texts
            result = self.tokenizer(
                examples["text"],
                padding=self.padding,
                max_length=self.max_seq_length,
                truncation=True,
            )

            return result

        with self.training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        train_dataset = raw_datasets["train"]
        test_dataset = raw_datasets["test"]

        # Split training dataset in training / validation
        split = train_dataset.train_test_split(
            train_size=0.7, seed=self.data_args.dataset_seed
        )  # fix seed, all trials have the same data split
        train_dataset = split["train"]
        valid_dataset = split["test"]

        return train_dataset, valid_dataset, test_dataset
