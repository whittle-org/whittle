from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")


def tokenization(example):
    return tokenizer(example["text"])


tokenized_dataset = dataset.map(tokenization, batched=True)

print(tokenized_dataset)

from itertools import chain

input_ids = tokenized_dataset["input_ids"]


def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))


input_ids = flatten_chain(input_ids)
unique, counts = np.unique(input_ids, return_counts=True)
idx = np.argsort(counts)[::-1]
print(counts[idx][0])
print(counts[idx][-1])
bins = np.arange(1, len(counts) + 1)
width = (np.max(bins) - np.min(bins)) * 1.0 / (len(bins) + 1)
plt.bar(x=bins[:2000], height=counts[idx][:2000], width=width, align="center")
plt.yscale("log")
plt.xlabel("Token ID")
plt.grid(linewidth="1", alpha=0.4)
plt.title("Distribution of 2000 most frequent tokens")
plt.savefig("distribution_tokens.pdf")
plt.show()
