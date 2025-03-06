from __future__ import annotations

import os
import pathlib
from contextlib import redirect_stdout
from io import StringIO

import pytest
import torch
from litgpt import Config
from litgpt.model import GPT as LitGPT
from litgpt.scripts.download import download_from_hub
from litgpt.utils import check_valid_checkpoint_dir, lazy_load

from whittle import convert_to_litgpt
from whittle.models.gpt.checkpoint import load_checkpoint, save_sub_network
from whittle.models.gpt.model import GPT


@pytest.fixture(scope="session")
def checkpoint_dir(tmp_path_factory):
    checkpoint_dir = tmp_path_factory.getbasetemp()
    download_from_hub(repo_id="EleutherAI/pythia-14m", checkpoint_dir=checkpoint_dir)
    return pathlib.Path(checkpoint_dir) / "EleutherAI" / "pythia-14m"


def get_checkpoint_contents(copy_config_files, save_checkpoints):
    # model_config.yaml if in the parent super-net directory
    if not save_checkpoints:
        return {"lit_model.pth"}
    # we copied config files, model_config.yaml and tokenizer configs are the most important
    if copy_config_files:
        return {
            "lit_model.pth",
            "model_config.yaml",
            "tokenizer.json",
            "tokenizer_config.json",
        }
    # other config files are in the parent super-net directory
    return {"lit_model.pth", "model_config.yaml"}


@pytest.mark.parametrize("copy_config_files", [True, False])
@pytest.mark.parametrize("save_checkpoints", [True, False])
def test_checkpoints(tmp_path, checkpoint_dir, copy_config_files, save_checkpoints):
    config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
    config.fix_head_size = True

    model = GPT(config)
    ckp = lazy_load(checkpoint_dir / "lit_model.pth")
    model.load_state_dict(ckp)

    sub_network_dict = {"embed_dim": 8, "mlp_ratio": 2, "num_heads": 2, "depth": 2}

    for set_subnet_before_call in [True, False]:
        out_dir = (
            tmp_path
            / f"out_{copy_config_files}_{save_checkpoints}_{set_subnet_before_call}"
        )
        out_dir.mkdir(exist_ok=True)

        if set_subnet_before_call:
            model.select_sub_network(sub_network_dict)

        def save_call():
            save_sub_network(
                model,
                checkpoint_dir,
                out_dir,
                save_checkpoints=save_checkpoints,
                copy_config_files=copy_config_files,
                sub_network_config=None if set_subnet_before_call else sub_network_dict,
            )

        if set_subnet_before_call and save_checkpoints:
            # Check that a warning is raised if the sub-network is set before saving
            # (to avoid cases when only the super-network is saved by mistake)
            with pytest.warns(UserWarning):
                save_call()
        elif set_subnet_before_call:
            # An error should be raised when saving only a sub-network and parent_dir - we need to pass sub_network_config in that case
            with pytest.raises(ValueError):
                save_call()
            continue
        else:
            save_call()

        # Check that the checkpoint directory contains the expected files
        contents = get_checkpoint_contents(copy_config_files, save_checkpoints)
        print(set(os.listdir(out_dir)))
        assert contents.issubset(set(os.listdir(out_dir)))

        # Check the contents of lit_model.pth
        checkpoint = lazy_load(out_dir / "lit_model.pth")
        if save_checkpoints:
            assert "model" in checkpoint
            if not copy_config_files:
                assert "parent_dir" in checkpoint
        else:
            assert "sub_network_config" in checkpoint
            assert "parent_dir" in checkpoint

        loaded_model = load_checkpoint(out_dir)
        assert loaded_model.config.fix_head_size is True  # by default this should be set

        model.select_sub_network(sub_network_dict)

        input = torch.randint(0, 64, (1, 64))
        out_pre_save = model(input)
        out_after_save = loaded_model(input)

        assert torch.allclose(out_pre_save, out_after_save, atol=1e-3)


@pytest.mark.parametrize("no_model_key", [True, False])
@pytest.mark.parametrize("copy_config_files", [True, False])
@pytest.mark.parametrize("save_checkpoints", [True, False])
def test_convert_to_litgpt(
    tmp_path, checkpoint_dir, copy_config_files, save_checkpoints, no_model_key
):
    out_dir = tmp_path / f"out_{copy_config_files}_{save_checkpoints}"
    out_dir.mkdir(exist_ok=True)

    litgpt_dir = tmp_path / "litgpt"
    litgpt_dir.mkdir(exist_ok=True)

    config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
    config.fix_head_size = True

    model = GPT(config)
    ckp = lazy_load(checkpoint_dir / "lit_model.pth")
    model.load_state_dict(ckp)

    sub_network_dict = {"embed_dim": 8, "mlp_ratio": 2, "num_heads": 2, "depth": 2}
    save_sub_network(
        model,
        checkpoint_dir,
        out_dir,
        save_checkpoints=save_checkpoints,
        copy_config_files=copy_config_files,
        sub_network_config=sub_network_dict,
    )

    for target_dir in [litgpt_dir, out_dir]:
        stdout = StringIO()
        with redirect_stdout(stdout):
            convert_to_litgpt.setup(
                out_dir, out_dir=target_dir, no_model_key=no_model_key
            )

            # everything should be in there - tokenizer, config files, and the model
            check_valid_checkpoint_dir(target_dir)
            ckp = lazy_load(target_dir / "lit_model.pth")
            assert "model" in ckp if not no_model_key else "model" not in ckp
            ckp = ckp["model"] if not no_model_key else ckp

            # it should work for both litgpt and whittle models
            cfg = Config.from_file(target_dir / "model_config.yaml")
            model = LitGPT(cfg)
            model.load_state_dict(ckp)

            model = GPT(cfg)
            model.load_state_dict(ckp)

            # this should still work even for checkpoints in litgpt format
            load_checkpoint(target_dir)
