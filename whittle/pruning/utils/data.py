# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py
from __future__ import annotations

import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process c4 dataset
def get_c4_dataloader(n_samples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    # Generate samples from training set
    random.seed(seed)
    input_data = torch.zeros(n_samples, seqlen, dtype=torch.int)
    output_data = torch.zeros(n_samples, seqlen, dtype=torch.int)
    for k in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer.encode(traindata[i]["text"])
            if trainenc.shape[0] > seqlen:
                break
        i = random.randint(0, trainenc.shape[0] - seqlen - 1)
        j = i + seqlen
        inp = trainenc[i:j]
        tar = inp.clone()
        tar[:-1] = -100
        input_data[k] = inp
        output_data[k] = tar

    # Prepare validation dataset
    valenc = tokenizer.encode(" ".join(valdata[:1100]["text"]))
    valenc = valenc[: (256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return DataLoader(TensorDataset(input_data, output_data)), valenc
