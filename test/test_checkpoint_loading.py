from __future__ import annotations

import pathlib

import pytest
import torch
from litgpt import Config
from litgpt.model import GPT as LitGPT
from litgpt.scripts.download import download_from_hub

from whittle.models.gpt.model import GPT as whittleGPT


@pytest.fixture(scope="session")
def checkpoint_dir(tmp_path_factory):
    # img = compute_expensive_image()
    checkpoint_dir = tmp_path_factory.getbasetemp()
    download_from_hub(repo_id="EleutherAI/pythia-70m", checkpoint_dir=checkpoint_dir)
    return pathlib.Path(checkpoint_dir) / "EleutherAI" / "pythia-70m"


def test_checkpoint_loading(checkpoint_dir):
    torch.manual_seed(0)
    config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
    input_ids = torch.randint(0, config.vocab_size, (1, config.block_size))  # .cuda()

    model = LitGPT(config)  # .cuda()
    model.load_state_dict(torch.load(str(checkpoint_dir / "lit_model.pth")))
    # test output
    model.eval()
    output_lit = model(input_ids)
    # from litgpt.super_model import Config
    # pip install litgpt
    # litgpt download --repo_id stabilityai/stablelm-base-alpha-3b
    config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
    config.fix_head_size = False
    model = whittleGPT(config)  # .cuda()
    model.load_state_dict(torch.load(str(checkpoint_dir / "lit_model.pth")))
    # test output
    model.eval()
    sample_intermediate_size = [4 * config.n_embd for i in range(config.n_layer)]
    model.set_sub_network(
        config.n_embd,
        sample_intermediate_size,
        [config.n_head for i in range(config.n_layer)],
        config.n_layer,
    )

    output_whittle = model(input_ids)
    assert torch.allclose(output_lit, output_whittle)
