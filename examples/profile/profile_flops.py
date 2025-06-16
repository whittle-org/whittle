from __future__ import annotations  # noqa: INP001

from litgpt import Config
from litgpt.model import GPT

from whittle.metrics.flops import compute_flops


def update_config(
    config,
    sub_network_n_embd: int,
    sub_network_intermediate_size: int,
    sub_network_num_heads: int,
    sub_network_n_layers: int,
    sub_network_query_groups=None,
    sub_network_head_size=None,
):
    config.n_embd = sub_network_n_embd
    config.intermediate_size = sub_network_intermediate_size
    config.n_head = sub_network_num_heads
    if sub_network_query_groups is not None:
        config.n_query_groups = sub_network_query_groups
    if sub_network_head_size is not None:
        config.head_size = sub_network_head_size
    return config


if __name__ == "__main__":
    # profile directly litgpt model to avoid indexing, and other inefficiencies

    config = Config()
    config.padded_vocab_size = 128
    config.n_embd = 128
    config.intermediate_size = 128 * 4
    config.n_layer = 4
    model = GPT(config)
    print(f"Full model {compute_flops(model=model)} flops")
    config = update_config(config, 64, 64 * 4, 4, 2, 4)
    model = GPT(config)
    print(f"Mini model {compute_flops(model=model)} flops")
