from __future__ import annotations

import copy
import torch
from litgpt.config import Config

from copy import deepcopy
from whittle.models.gpt import GPT
from whittle.models.gpt_flex import GPTFlex
from whittle.models.gpt.extract import extract_sub_network, set_subnet_attention_sizes


def test_extract_sub_network() -> None:
    config = Config.from_name("pythia-70m")
    config.fix_head_size = False

    super_network = GPT(config)
    sub_network_config = Config.from_name("pythia-14m")
    sub_network_config.fix_head_size = False

    super_network.eval()
    super_network.set_sub_network(
        sub_network_n_embd=sub_network_config.n_embd,
        sub_network_intermediate_size=[sub_network_config.intermediate_size]
        * sub_network_config.n_layer,
        sub_network_num_heads=[sub_network_config.n_head] * sub_network_config.n_layer,
        sub_network_n_layers=sub_network_config.n_layer,
    )

    # instantiate a new model
    sub_network = extract_sub_network(super_network, sub_network_config)
    sub_network.eval()
    input = torch.randint(0, 512, (1, 20))
    out_super_net = super_network(input).detach()
    out_sub_net = sub_network(input).detach()
    assert torch.all(
        torch.round(out_sub_net, decimals=9) == torch.round(out_super_net, decimals=9)
    )


def test_extract_sub_network_flex() -> None:
    config = Config.from_name("pythia-70m")
    config.fix_head_size = False

    super_network = GPTFlex(config)
    sub_network_config = Config.from_name("pythia-14m")
    sub_network_config.fix_head_size = False

    super_network.eval()
    super_network.set_sub_network(
        sub_network_n_embd=sub_network_config.n_embd,
        sub_network_intermediate_size=[sub_network_config.intermediate_size]
        * sub_network_config.n_layer,
        sub_network_num_heads=[sub_network_config.n_head] * sub_network_config.n_layer,
        sub_network_n_layers=sub_network_config.n_layer,
    )

    # instantiate a new model
    sub_network = extract_sub_network(super_network, sub_network_config, use_flex=True)

    sub_network.eval()
    input = torch.randint(0, 512, (1, 20))
    out_super_net = super_network(input).detach()
    out_sub_net = sub_network(input).detach()
    assert torch.all(
        torch.round(out_sub_net, decimals=9) == torch.round(out_super_net, decimals=9)
    )


def test_extract_sub_network_llamamlp() -> None:
    config = Config.from_name("micro-llama-300M")
    config.fix_head_size = False

    super_network = GPT(config)
    sub_network_config = copy.deepcopy(config)

    # simulate a smaller network
    sub_network_config.n_embd = 128
    sub_network_config.intermediate_size = 1024
    sub_network_config.n_layer = 6
    sub_network_config.n_head = 4

    super_network.eval()
    super_network.set_sub_network(
        sub_network_n_embd=sub_network_config.n_embd,
        sub_network_intermediate_size=[sub_network_config.intermediate_size]
        * sub_network_config.n_layer,
        sub_network_num_heads=[sub_network_config.n_head] * sub_network_config.n_layer,
        sub_network_n_layers=sub_network_config.n_layer,
    )

    # dynamically computed when set_sub_network is called
    sub_network_config.head_size = (
        sub_network_config.n_embd // sub_network_config.n_head
    )

    # instantiate a new model
    sub_network = extract_sub_network(super_network, sub_network_config)
    sub_network.eval()
    input = torch.randint(0, 512, (1, 20))
    out_super_net = super_network(input).detach()
    out_sub_net = sub_network(input).detach()
    assert torch.all(
        torch.round(out_sub_net, decimals=9) == torch.round(out_super_net, decimals=9)
    )


def test_extract_sub_network_intermediate_size() -> None:
    network_config = Config.from_name("pythia-14m")
    network_config.fix_head_size = False

    sizes = [network_config.intermediate_size] * network_config.n_layer
    n_changes = len(sizes[::2])
    sizes[::2] = [network_config.intermediate_size // 2] * n_changes

    super_network = GPTFlex(network_config)
    super_network.eval()
    super_network.set_sub_network(
        sub_network_n_embd=network_config.n_embd,
        sub_network_intermediate_size=sizes,
        sub_network_num_heads=[network_config.n_head] * network_config.n_layer,
        sub_network_n_layers=network_config.n_layer,
    )

    subnet_config = deepcopy(network_config)
    subnet_config.intermediate_size = sizes

    sub_network = GPTFlex(subnet_config)
    sub_network = extract_sub_network(super_network, subnet_config)
    sub_network.eval()
    input = torch.randint(0, 512, (1, 20))
    out_super_net = super_network(input).detach()
    out_sub_net = sub_network(input).detach()
    assert torch.all(
        torch.round(out_sub_net, decimals=9) == torch.round(out_super_net, decimals=9)
    )


def test_extract_sub_network_n_head() -> None:
    network_config = Config.from_name("pythia-14m")
    network_config.fix_head_size = False

    heads = [network_config.n_head] * network_config.n_layer
    n_changes = len(heads[::2])
    heads[::2] = [3] * n_changes
    heads[1] = 1

    super_network = GPTFlex(network_config)
    super_network.eval()
    super_network.set_sub_network(
        sub_network_n_embd=network_config.n_embd,
        sub_network_intermediate_size=[network_config.intermediate_size]
        * network_config.n_layer,
        sub_network_num_heads=heads,
        sub_network_n_layers=network_config.n_layer,
    )

    subnet_config = deepcopy(network_config)
    subnet_config.n_head = heads
    set_subnet_attention_sizes(super_network, subnet_config)

    sub_network = GPTFlex(subnet_config)
    sub_network = extract_sub_network(super_network, subnet_config)
    sub_network.eval()
    input = torch.randint(0, 512, (1, 20))
    out_super_net = super_network(input).detach()
    out_sub_net = sub_network(input).detach()
    assert torch.all(
        torch.round(out_sub_net, decimals=9) == torch.round(out_super_net, decimals=9)
    )
