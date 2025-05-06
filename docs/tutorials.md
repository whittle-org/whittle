# Tutorials

This tutorial walks you through the four main workflows supported by Whittle.
- [Tutorials](#tutorials)
  - [Pre-Training Super-Networks](#pre-training-super-networks)
  - [Searching for Sub-Networks using Multi-objective Search](#searching-for-sub-networks-using-multi-objective-search)
  - [Evaluating Sub-Networks](#evaluating-sub-networks)
  - [Converting Whittle to LitGPT Checkpoints](#converting-whittle-to-litgpt-checkpoints)

All the commands in the following sections assume that they are executed from the project root directory (e.g., `/home/username/whittle`).

## Pre-Training Super-Networks

In this example, we pre-train a Whittle super-network using a lightweight version of [Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m) on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset.
The input arguments for the super-network pre-training workflow follow the original [pretrain](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/pretrain.md) workflow of [LitGPT](https://github.com/Lightning-AI/litgpt).

There are a few different strategies for training the super-networks.
- The *standard strategy* is to update the entire super-network at each step.
- The *random strategy* updates only random sub-networks at each step, which encourages the super-network to become more prunable and resilient, making it easier to later extract strong-performing sub-models.
- Finally, the *sandwich strategy*, as proposed by Cai et al. [1], updates the largest super-network, followed by a few randomly sampled sub-networks, and then the smallest sub-network. This strategy is used by default if none are specified explicitly.

Before you run the command to pre-train a super-network, ensure that you have downloaded the tokenizer for the model
```bash
python whittle/pretrain_super_network.py EleutherAI/pythia-14m \
    --data TinyStories \
    --data.data_path ./data \
    --tokenizer_dir ~/checkpoints/EleutherAI/pythia-14m/ \
    --out_dir pretrained_super_net \
    --train.save_interval 5 \
    --train.max_tokens 1000000000 \
    --training_strategy random
```

[1] Cai, Han, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. "Once-for-all: Train one network and specialize it for efficient deployment." arXiv preprint arXiv:1908.09791 (2019).

## Searching for Sub-Networks using Multi-objective Search

After pre-training, we can search for high-performing sub-networks within a checkpoint of a pre-trained super-network. We use multi-objective search to jointly optimize efficiency (either FLOP, latency, or parameter count) and performance (either perplexity or validation loss).

```bash
python whittle/search_sub_networks.py pretrained_super_net/final/ \
    --data TextFiles \
    --data.data_path ./data \
    --search.iterations 10 \
    --out_dir sub_networks/
```

## Evaluating Sub-Networks

The following workflow allows you to evaluate checkpoints of a sub-network using the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main).

It runs standard validation to measure metrics like accuracy, loss, and resource usage, helping you choose the best sub-network for your application.

```bash
python whittle/evaluate_network.py sub_networks/sub_network_0/ \
    --out_dir evaluation/ \
    --task arc_easy
```

## Converting Whittle to LitGPT Checkpoints

We can also convert the Whittle checkpoint of a sub-network into a LitGPT-compatible checkpoint format. This allows you, for example, to easily fine-tune the model further with the LitGPT [fine-tuning workflow](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/finetune.md).

```bash
python whittle/convert_to_litgpt.py sub_networks/sub_network_0/ --out_dir litgpt_checkpoint
```