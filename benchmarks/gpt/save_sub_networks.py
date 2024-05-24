import json
import os
import yaml
import torch
import shutil

from dataclasses import fields
from pathlib import Path

from argparse import ArgumentParser

from litgpt.config import Config

from lobotomy.models.gpt import GPT
from lobotomy.models.gpt.extract import extract_sub_network


def save_checkpoint_sub_networks(args) -> None:

    checkpoint_dir = Path(args.checkpoint_dir)

    config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
    config.fix_head_size = False
    super_network = GPT(config)
    super_network.load_state_dict(torch.load(str(checkpoint_dir / "lit_model.pth")))

    sub_network_config = Config.from_name(args.sub_network)
    sub_network_config.fix_head_size = False
    sub_network_config.rope_n_elem = 16

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
    checkpoint = {
        "model": sub_network.state_dict(),
    }

    os.makedirs(args.output_dir, exist_ok=True)

    output_dir_sub = (
        Path(args.output_dir)
        / f"sub_network_{args.sub_network}_from_super_net_{args.super_network}"
    )
    os.makedirs(output_dir_sub, exist_ok=True)

    torch.save(
        checkpoint,
        output_dir_sub / "lit_model.pth",
    )

    config_dict = json.load(open(checkpoint_dir / "config.json"))
    mapping_litgpt_hf = {
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "n_embd": "hidden_size",
        "intermediate_size": "intermediate_size",
    }
    for name_litgpt, name_hf in mapping_litgpt_hf.items():
        config_dict[name_hf] = getattr(sub_network_config, name_litgpt)
    config_dict["model_type"] = "gpt_neox"
    json.dump(config_dict, open(output_dir_sub / "config.json", "w"))

    config_dict = dict()
    for field in fields(sub_network_config):
        config_dict[field.name] = getattr(sub_network_config, field.name)
    yaml.dump(config_dict, open(output_dir_sub / "model_config.yaml", "w"))

    shutil.copy(checkpoint_dir / "tokenizer.json", output_dir_sub / "tokenizer.json")
    shutil.copy(
        checkpoint_dir / "tokenizer_config.json",
        output_dir_sub / "tokenizer_config.json",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--super_network", type=str, default="pythia-70m")
    parser.add_argument("--sub_network", type=str, default="pythia-14m")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    args, _ = parser.parse_known_args()

    save_checkpoint_sub_networks(args)
