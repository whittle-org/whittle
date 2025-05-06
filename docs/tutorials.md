# Tutorials

## Pre-Training Super-Networks

In this example we pre-train a Whittle super-network using a lightweight version of [Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m) on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset.
The input arguments for super-network pre-training workflow follow the original [pretrain](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/pretrain.md) workflow of [LitGPT](https://github.com/Lightning-AI/litgpt).
During training, only random sub-networks are updated at each step, which encourages the super-network to become more prunable and resilient, making it easier to later extract strong-performing sub-models.

```bash
python pretrain_super_network.py EleutherAI/pythia-14m --data TextFiles --data.train_data_path ./data --tokenizer_dir ~/checkpoints/EleutherAI/pythia-14m/ --out_dir pretrained_super_net --train.save_interval 5 --train.max_tokens 1000000000
```

## Search for Sub-Networks using Multi-objective Search

After pre-training, we can search for high-performing sub-networks within a checkpoint of a pre-trained super-network. We use multi-objective search to jointly optimize efficiency (either FLOP, latency, or parameter count) and performance (either perplexity or validation loss).

```bash
python search_sub_networks.py pretrained_super_net/final/ --data TextFiles --data.train_data_path ./data --search.iterations 10 --out_dir sub_networks/
```

## Evaluate Sub-Networks

The following workflow allows to evaluate single checkpoints of a sub-network using the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main).

It runs standard validation to measure metrics like accuracy, loss, and resource usage,  helping you choose the best sub-network for your application.

```bash
python evaluate_network.py sub_networks/sub_network_0/ --out_dir evaluation/ --task arc_easy
```

## Convert to LitGPT checkpoints

We can also convert the whittle checkpoint of sub-network into a LitGPT-compatible checkpoint format. This allows you, for example, to easily fine-tune the model further with the LitGPT [fine-tuning workflow](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/finetune.md).

```bash
python convert_to_litgpt.py sub_networks/sub_network_0/ --out_dir litgpt_checkpoint
```
