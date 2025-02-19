from __future__ import annotations

import pathlib

import torch
from litgpt.config import Config
from litgpt.utils import save_config

from whittle.models.gpt import GPT
from whittle.models.gpt.extract import extract_current_sub_network, extract_sub_network
from whittle.modules.layernorm import LayerNorm
from whittle.modules.rmsnorm import RMSNorm


def test_extract_sub_network() -> None:
    config = Config.from_name("pythia-70m")
    config.fix_head_size = False

    super_network = GPT(config)
    sub_network_config = Config.from_name("pythia-14m")
    sub_network_config.fix_head_size = False

    # set norm weights to random
    # the default is unit/zero vector which does not test the extract
    make_norm_weights_random(super_network.transformer.ln_f)
    for i in range(sub_network_config.n_layer):
        block = super_network.transformer.h[i]
        make_norm_weights_random(block.norm_1)
        make_norm_weights_random(block.post_attention_norm)
        make_norm_weights_random(block.norm_2)
        make_norm_weights_random(block.post_mlp_norm)

    super_network.eval()
    super_network.set_sub_network(
        sub_network_n_embd=sub_network_config.n_embd,
        sub_network_intermediate_size=sub_network_config.intermediate_size,
        sub_network_num_heads=sub_network_config.n_head,
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


def test_extract_sub_network_llamamlp() -> None:
    config = Config.from_name("micro-llama-300M")
    config.fix_head_size = False

    super_network = GPT(config)

    # set norm weights to random
    # the default is unit/zero vector which does not test the extract
    make_norm_weights_random(super_network.transformer.ln_f)
    for i in range(config.n_layer):
        block = super_network.transformer.h[i]
        make_norm_weights_random(block.norm_1)
        make_norm_weights_random(block.post_attention_norm)
        make_norm_weights_random(block.norm_2)
        make_norm_weights_random(block.post_mlp_norm)

    # simulate a smaller network
    n_embd = 128
    intermediate_size = 1024
    n_layer = 6
    n_head = 12

    super_network.eval()
    super_network.set_sub_network(
        sub_network_n_embd=n_embd,
        sub_network_intermediate_size=intermediate_size,
        sub_network_num_heads=n_head,
        sub_network_n_layers=n_layer,
    )

    # instantiate a new model
    sub_network = extract_current_sub_network(super_network)
    sub_network.eval()
    input = torch.randint(0, 512, (1, 20))
    out_super_net = super_network(input).detach()
    out_sub_net = sub_network(input).detach()
    assert torch.all(
        torch.round(out_sub_net, decimals=9) == torch.round(out_super_net, decimals=9)
    )


def test_save_config() -> None:
    config = Config.from_name("micro-llama-300M")
    config.fix_head_size = False

    super_network = GPT(config)

    # set norm weights to random
    # the default is unit/zero vector which does not test the extract
    make_norm_weights_random(super_network.transformer.ln_f)
    for i in range(config.n_layer):
        block = super_network.transformer.h[i]
        make_norm_weights_random(block.norm_1)
        make_norm_weights_random(block.post_attention_norm)
        make_norm_weights_random(block.norm_2)
        make_norm_weights_random(block.post_mlp_norm)

    # simulate a smaller network
    n_embd = 128
    intermediate_size = 1024
    n_layer = 6
    n_head = 12

    super_network.eval()
    super_network.set_sub_network(
        sub_network_n_embd=n_embd,
        sub_network_intermediate_size=intermediate_size,
        sub_network_num_heads=n_head,
        sub_network_n_layers=n_layer,
    )

    # instantiate a new model
    sub_network = extract_current_sub_network(super_network)
    save_dir = pathlib.Path("microllama")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(sub_network.config, save_dir)

    cfg = Config.from_file(save_dir / "model_config.yaml")
    for key in cfg.__dict__:
        assert cfg.__dict__[key] == sub_network.config.__dict__[key]


def make_norm_weights_random(norm):
    if norm is None or isinstance(norm, torch.nn.Identity):
        return

    assert isinstance(norm, RMSNorm) or isinstance(norm, LayerNorm)

    torch.nn.init.normal_(norm.weight)
    if isinstance(norm, LayerNorm):
        torch.nn.init.normal_(norm.bias)
