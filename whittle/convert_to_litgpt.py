from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
from jsonargparse import CLI
from litgpt import Config
from litgpt.utils import copy_config_files, lazy_load, save_config

from whittle.models.gpt import GPT
from whittle.models.gpt.extract import extract_current_sub_network


def setup(
    sub_network_dir: Path,
    out_dir: Path | None = None,
    parent_dir: Path | None = None,
    super_network_cls: type[GPT] = GPT,
    no_model_key: bool = True,
) -> None:
    """
    Convert a sub-network to a LitGPT model checkpoint.
    The sub-network checkpoints can have the following design:

    a) same as litgpt models:
        sub_network_dir/
        - lit_model.pth ... {model: sub_network.state_dict()} or sub_network.state_dict() (former - whittle model, latter - litgpt model)
        - configs (model_config.yaml, tokenizer.json, etc.)
    b) litgpt model with saving space (not copying the tokenizer and other configs):
        sub_network_dir/
        - lit_model.pth ... {model: sub_network.state_dict(), parent_dir: super-network checkpoint dir}
        - model_config.yaml
    c) compressed checkpoint:
        sub_network_dir/
        - lit_model.pth ... {sub_network_config: sub_network_config, parent_dir: super-network checkpoint dir}
        - model_config.yaml

    Conversion procedure:
    a) No modification (copying to a new directory if `sub_network_dir != out_dir`).
    b) Copy the other configs from the parent checkpoint directory, and model checkpoint and config from `sub_network_dir`.
    c) Extract the sub-network from the super-network, save the sub-network weights and config, and copy the other configs from the parent checkpoint directory.

    Arguments:
        sub_network_dir: The path to the sub-network directory to convert.
        out_dir: Directory in which to save the converted sub-network. If not provided, saving to `sub_network_dir` by default.
        parent_dir: The parent directory of the sub-network. Required if the checkpoint is in the format b) or c). If not provided, the parent directory is extracted from the checkpoint.
        super_network_cls: The super-network class to instantiate the super-network model. Defaults to `GPT`.
        no_model_key: Convert strictly to litgpt - lit_model.pth will contain only the sub-network weights, and not {'model': sub_network_weights}.
    """

    if out_dir is None:
        out_dir = sub_network_dir
    else:
        os.makedirs(out_dir, exist_ok=True)

    ckp = lazy_load(sub_network_dir / "lit_model.pth")

    # sub-network config loading (contains the config and checkpoint path of the parent)

    parent_dir = ckp.get("parent_dir", None) if parent_dir is None else parent_dir

    if "model" in ckp:
        model_path = sub_network_dir / "lit_model.pth"
        configs_path = (
            sub_network_dir if parent_dir is None else Path(parent_dir)
        )  # config files were copied

        # copy model only if the destination is different
        if out_dir != sub_network_dir and not no_model_key:
            shutil.copy(model_path, out_dir / "lit_model.pth")
        # re-save the model if the required format is state_dict instead of {'model': state_dict}
        elif no_model_key:
            ckp = torch.load(
                model_path, weights_only=False
            )  # this time not lazy loading because we're re-saving the weights
            torch.save(ckp["model"], out_dir / "lit_model.pth")

        if configs_path == parent_dir:
            # we don't want to overwrite the sub-network config in case `out_dir` == `sub_network_dir`
            shutil.copy(
                sub_network_dir / "model_config.yaml", out_dir / "model_config.yaml.tmp"
            )
            copy_config_files(configs_path, out_dir)
            shutil.copy(out_dir / "model_config.yaml.tmp", out_dir / "model_config.yaml")
        elif out_dir != sub_network_dir:
            copy_config_files(configs_path, out_dir)
    else:
        assert parent_dir is not None, (
            'Weights are not saved in the checkpoint under "model", but `parent_dir` is not saved in the checkpoints provided.'
        )
        configs_path = Path(parent_dir)
        model_path = configs_path / "lit_model.pth"

        # we will need to extract the sub-network config and weights
        assert "sub_network_config" in ckp, (
            '"model" or "sub_network_config" not found in checkpoint'
        )
        sub_network_config = ckp["sub_network_config"]

        # instantiate the super-network
        config = Config.from_file(model_path.parent / "model_config.yaml")
        config.fix_head_size = True
        model = super_network_cls(config)

        # set the sub-network via the saved sub-network config
        model.select_sub_network(sub_network_config)
        sub_network = extract_current_sub_network(model)

        # copy the config files, save the sub-network, and overwrite model_config.yaml with the sub-network config
        copy_config_files(configs_path, out_dir)
        # save the sub-network in the litgpt vs whittle format
        save_data = (
            sub_network.state_dict()
            if no_model_key
            else {"model": sub_network.state_dict()}
        )
        torch.save(save_data, out_dir / "lit_model.pth")
        save_config(sub_network.config, out_dir)


if __name__ == "__main__":
    CLI(setup)
