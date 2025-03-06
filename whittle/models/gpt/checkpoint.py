from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

import lightning as L
import torch
from litgpt.config import Config
from litgpt.utils import (
    check_valid_checkpoint_dir,
    copy_config_files as copy_config_files_func,
    lazy_load,
    save_config,
)

from whittle.lora.config import LoRAConfig
from whittle.models.gpt import GPT
from whittle.models.gpt.extract import extract_current_sub_network


def _save_checkpoint(
    data: dict[str, Any], save_path: Path, fabric: L.Fabric | None = None
):
    if fabric is None:
        torch.save(data, save_path)
    else:
        fabric.save(save_path, data)


def save_sub_network(
    super_network: GPT,
    checkpoint_dir: Path,
    save_dir: Path,
    sub_network_config: dict[str, Any] | None = None,
    save_checkpoints: bool = True,
    copy_config_files: bool = False,
    fabric: L.Fabric | None = None,
):
    if not save_checkpoints and sub_network_config is None:
        raise ValueError(
            "sub_network_config must be provided when save_checkpoints is False"
        )

    assert checkpoint_dir != save_dir, "Checkpoint and save directories must be different"
    save_path = save_dir / "lit_model.pth"
    save_dir.mkdir(parents=True, exist_ok=True)

    # either save the extracted checkpoint, or the config + path to super-network
    if save_checkpoints:
        if sub_network_config is not None:
            super_network.select_sub_network(sub_network_config)
        else:
            warn(
                "No sub-network config provided - saving the current active sub-network instead (assuming the user called .set_sub_network() before). If this is not the intended behavior, pass `sub_network_config` to `save_sub_network`."
            )
        sub_network = extract_current_sub_network(super_network)
        super_network.reset_super_network()

        # either save everything including config files, or only model_config.yaml and the weights
        if copy_config_files:
            copy_config_files_func(checkpoint_dir, save_dir)
            _save_checkpoint(
                {"model": sub_network.state_dict()}, save_path, fabric=fabric
            )
        else:
            _save_checkpoint(
                {"model": sub_network.state_dict(), "parent_dir": checkpoint_dir},
                save_path,
                fabric=fabric,
            )
        # the new model_config.yaml is different from the original one, so we rewrite it
        save_config(sub_network.config, save_dir)
    else:
        # minimalistic checkpoint - only sub-network config and path to super-network
        _save_checkpoint(
            {"sub_network_config": sub_network_config, "parent_dir": checkpoint_dir},
            save_path,
            fabric=fabric,
        )


def load_checkpoint(
    checkpoint_dir: Path,
    model_cls: type[GPT] = GPT,
    config_cls: type[Config] | type[LoRAConfig] = Config,
    config_attr: dict[str, Any] | None = None,
) -> GPT:
    sub_network_config: dict[str, Any] | None = None
    ckp = lazy_load(checkpoint_dir / "lit_model.pth")

    # sub-network config loading (contains the config and checkpoint path of the parent)
    sub_network_config = ckp.get("sub_network_config", None)
    parent_dir = ckp.get("parent_dir", None)

    # check if the checkpoint is valid only if it is not a sub-network config
    if sub_network_config is None:
        check_valid_checkpoint_dir(
            checkpoint_dir,
            ignore_tokenizer_files=parent_dir
            is not None,  # if parent_dir is not None, tokenizer files were not copied
        )
    # always check the parent config validity
    if parent_dir is not None:
        check_valid_checkpoint_dir(Path(parent_dir), ignore_tokenizer_files=False)

    # it's either a standalone litgpt model or a sub-network (depending on if there is also a parent_dir)
    if "model" not in ckp:
        # not None: sub-network, None: raw state dict
        if parent_dir is not None:
            checkpoint_dir = Path(parent_dir)

        ckp = lazy_load(checkpoint_dir / "lit_model.pth")

    config = config_cls.from_file(checkpoint_dir / "model_config.yaml")
    config_attr = {"fix_head_size": True} if config_attr is None else config_attr
    for k, val in config_attr.items():
        setattr(
            config, k, val
        )  # some args are not passed to __init__ - e.g. for config.fix_head_size = True

    model = model_cls(config)
    # for WhittleLM - it loads AutoTokenizer inside - either we copied it to checkpoint_dir, or it is referenced in parent_dir
    model.name_or_path = checkpoint_dir if parent_dir is None else parent_dir

    model.load_state_dict(ckp["model"] if "model" in ckp else ckp)
    del ckp

    # if the checkpoint was a sub-network, set it at this point
    if sub_network_config is not None:
        model.select_sub_network(sub_network_config)

    return model
