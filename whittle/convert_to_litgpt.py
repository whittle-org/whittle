from pathlib import Path
from litgpt.utils import lazy_load, copy_config_files, save_config
from litgpt import Config
import shutil
import torch
from typing import Any
from whittle.models.gpt import GPT
from whittle.models.gpt.extract import extract_current_sub_network


def setup(
    sub_network_dir: Path,
    out_dir: Path | None = None,
) -> None:
    """
    Convert a sub-network to a LitGPT model checkpoint.
    The sub-network checkpoints can have the following design:

    a) same as litgpt models:
        sub_network_dir/
        - lit_model.pth ... {model: sub_network.state_dict()}
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
    """

    if out_dir is None:
        out_dir = sub_network_dir

    sub_network_config: dict[str, Any] | None = None
    ckp = lazy_load(sub_network_dir / "lit_model.pth")

    # sub-network config loading (contains the config and checkpoint path of the parent)

    parent_dir = ckp.get("parent_dir", None)

    if "model" in ckp:
        model_path = sub_network_dir / "lit_model.pth"
        configs_path = (
            sub_network_dir if parent_dir is None else Path(parent_dir)
        )  # config files were copied

        # copy model only if the destination is different
        if out_dir != sub_network_dir:
            shutil.copy(model_path, out_dir / "lit_model.pth")

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
        config = Config.from_file(model_path / "model_config.yaml")
        config.fix_head_size = True
        model = GPT(config)

        # set the sub-network via the saved sub-network config
        model.select_sub_network(sub_network_config)
        sub_network = extract_current_sub_network(model)

        # copy the config files, save the sub-network, and overwrite model_config.yaml with the sub-network config
        copy_config_files(configs_path, out_dir)
        torch.save({"model": sub_network.state_dict()}, out_dir / "lit_model.pth")
        save_config(sub_network.config, out_dir)
