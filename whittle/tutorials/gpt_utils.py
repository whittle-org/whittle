import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tiktoken
from syne_tune.config_space import randint

import torch
import torch.nn as nn
from torch.nn import functional as F

from whittle.sampling.random_sampler import RandomSampler
from whittle.training_strategies.sandwich import SandwichStrategy

from datasets import load_dataset


def evaluate_wikitext(model, tokenizer):
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    max_length = model.max_seq_length
    seq_len = encodings.input_ids.size(1)
    nlls = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prev_end_loc = 0
    model.to(device)
    model.eval()
    for begin_loc in tqdm(range(0, seq_len, max_length)):
        end_loc = min(begin_loc + max_length, seq_len - 1)
        _ = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids = encodings.input_ids[:, begin_loc + 1 : end_loc + 1].to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            neg_log_likelihood = torch.nn.CrossEntropyLoss()(
                outputs.view(-1, outputs.size(-1)), target_ids.view(-1)
            )
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    model.reset_super_network()
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


class SandwichStrategyGPT(SandwichStrategy):
    def __init__(self, random_samples: int = 2, **kwargs):
        super().__init__(random_samples=2, **kwargs)

    def __call__(self, model, inputs, outputs, **kwargs):
        total_loss = 0
        # update super-network
        B, T = inputs.shape
        y_supernet = model(inputs)
        y_supernet = y_supernet.view(B * T, -1)
        loss = self.loss_function(y_supernet, outputs)
        loss.backward()
        total_loss += loss.item()

        # update random sub-networks
        for i in range(self.random_samples):
            config = self.sampler.sample()
            model.select_sub_network(config)
            y_hat = model(inputs)
            y_hat = y_hat.view(B * T, -1)
            if self.kd_loss is not None:
                loss = self.kd_loss(y_hat, outputs, y_supernet)
            else:
                loss = self.loss_function(y_hat, outputs)
            loss.backward()
            model.reset_super_network()
            total_loss += loss.item()

        # smallest network
        config = self.sampler.get_smallest_sub_network()
        model.select_sub_network(config)
        y_hat = model(inputs)
        y_hat = y_hat.view(B * T, -1)
        if self.kd_loss is not None:
            loss = self.kd_loss(y_hat, outputs, y_supernet)
        else:
            loss = self.loss_function(y_hat, outputs)
        loss.backward()
        model.reset_super_network()
        total_loss += loss.item()

        return total_loss / (2 + self.random_samples)


warmup_iters = 100
lr_decay_iters = 10000
min_lr = 1e-4
learning_rate = 1e-3
decay_lr = True


def to_tokens(example):
    """Function to tokenize a string using BPE."""
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(
        example["text"]
    )  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {"ids": ids, "len": len(ids)}
    return out


# Encoder: take a string, output a list of integers
def encode(s):
    return [stoi[c] for c in s]


def plot_losses(losses, verbosity, val_losses=None):
    # plt.clf()
    plt.plot(losses, label="train")
    if val_losses is not None:
        plt.plot(val_losses, label="valid")
        plt.legend()
    plt.ylabel("Loss")
    plt.xlabel(f"Num steps (~{verbosity}x)")
    plt.xlim(0, len(losses))
    plt.show()


def get_batch(
    split: str, block_size: int = 8, batch_size: int = 4, device: str = "cuda"
):
    """Gets a randomized batch from the split of data chosen.

    Arguments
    ---------
    split : str, {"train", "valid"}
    block_size : int
        The context length for predictions, that is, a sentence length
    batch_size : int
        The batch size, that is, the number of sentences
    """
    # generate a small batch of data of inputs x and targets y
    assert split in ["train", "valid"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = train_data if split == "train" else valid_data
    # generating random indices as markers in the full text document
    # such that they are a starting point to the sentence of length
    # `block_size` that will be a data point in the batch
    ix = torch.randint(low=0, high=len(data) - block_size, size=(batch_size,))
    # extracting a sentence of length `block_size` for every
    # random starting point in `ix`
    x = torch.stack([data[i : i + block_size] for i in ix])
    # extracting a sentence of length `block_size` for every
    # random starting point in `ix` + 1 (shifted to right)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: nn.Module, eval_iters: int):
    """Function to evaluate the model on train & valid splits."""
    out = {}
    model.eval()
    for split in ["train", "valid"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            B, T = X.shape
            # evaluate loss on the batch
            logits = model(X)
            logits = logits.view(B * T, -1)
            targets = Y.view(B * T)
            loss = F.cross_entropy(logits, targets)
            # logits = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_and_evaluate_model(
    model: nn.Module,
    block_size: int,
    batch_size: int,
    optimizer: torch.optim = None,
    num_train_steps: int = 10000,
    verbosity_len: int = 1000,
    eval_iters: int = 500,
    plot_loss: bool = True,
    device: str = "cpu",
    **kwargs,
):
    model.train()
    model_config = model.config
    config = {
        "embed_dim": randint(1, model_config.n_embd),
        "depth": randint(1, model_config.n_layer),
        "num_heads": randint(1, model_config.n_head),
        "mlp_ratio": randint(1, 4),
    }
    sampler = RandomSampler(config, seed=42)
    train_strategy = SandwichStrategyGPT(
        sampler=sampler, loss_function=torch.nn.CrossEntropyLoss(), random_samples=1
    )
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs["learning_rate"])

    train_losses = [np.inf]
    valid_losses = [np.inf]

    for iter in tqdm(range(num_train_steps)):
        # sample a batch of data
        xb, yb = get_batch("train", block_size, batch_size, device)

        # evaluate loss on the batch
        xb, yb = get_batch("train", block_size, batch_size, device)
        B, T = xb.shape
        # evaluate loss on the batch
        optimizer.zero_grad(set_to_none=True)
        _ = train_strategy(model, xb, yb.view(B * T))

        # gradient update
        optimizer.step()

        # every once in a while evaluate the loss on train and val sets
        if iter % verbosity_len == 0 or iter == num_train_steps - 1:
            model.reset_super_network()
            _losses = estimate_loss(model, eval_iters)
            train_losses.append(_losses["train"])
            valid_losses.append(_losses["valid"])
            print()
            print(
                f"step {iter}: train loss {_losses['train']:.4f}, "
                f"val loss {_losses['valid']:.4f}"
            )

    if plot_loss:
        plot_losses(train_losses, verbosity_len, valid_losses)


global data, train_data, valid_data
with open("input.txt", encoding="utf-8") as f:
    text = f.read()

# Checking all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
vocab_set = "".join(chars)

# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Train and test splits
train_size = 0.9
data = torch.tensor(encode(text), dtype=torch.long)
n = int(train_size * len(data))
train_data = data[:n]
valid_data = data[n:]
