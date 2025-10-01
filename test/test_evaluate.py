from __future__ import annotations

import json
import pathlib
import shutil

import pytest
import torch
from litgpt.scripts.download import download_from_hub
from litgpt.utils import (
    copy_config_files as copy_config_files_func,
)

from whittle import evaluate_network
from whittle.models.gpt import GPT


@pytest.fixture(scope="session")
def checkpoint_dir(tmp_path_factory):
    checkpoint_dir = tmp_path_factory.getbasetemp()
    download_from_hub(repo_id="EleutherAI/pythia-14m", checkpoint_dir=checkpoint_dir)
    return pathlib.Path(checkpoint_dir) / "EleutherAI" / "pythia-14m"


def convert_and_evaluate_mock(
    model, out_dir, tasks, seed, num_fewshot, batch_size, device, limit, **kwargs
):
    assert tasks == "arc_easy"
    assert seed == 42
    assert num_fewshot == 1
    assert batch_size == 1
    assert device == "cpu"
    assert limit == 1

    assert isinstance(model, GPT)


def setup_checkpoint_dir(checkpoint_dir, sub_network_dir, checkpoint_mode):
    # test all supported checkpoint formats
    if checkpoint_mode == "litgpt":
        copy_config_files_func(checkpoint_dir, sub_network_dir)
        shutil.copy(checkpoint_dir / "lit_model.pth", sub_network_dir / "lit_model.pth")

    elif checkpoint_mode == "whittle":
        copy_config_files_func(checkpoint_dir, sub_network_dir)
        ckp = torch.load(checkpoint_dir / "lit_model.pth")
        ckp = {"model": ckp, "parent_dir": checkpoint_dir}
        torch.save(ckp, sub_network_dir / "lit_model.pth")

    elif checkpoint_mode == "whittle-minimalistic":
        ckp = torch.load(checkpoint_dir / "lit_model.pth")
        ckp = {"model": ckp, "parent_dir": checkpoint_dir}
        torch.save(ckp, sub_network_dir / "lit_model.pth")
        shutil.copy(
            checkpoint_dir / "model_config.yaml", sub_network_dir / "model_config.yaml"
        )

    elif checkpoint_mode == "whittle-sub-network":
        sub_network_config = {
            "embed_dim": 2,
            "mlp_ratio": 1.5,
            "num_heads": 4,
            "depth": 1,
        }
        torch.save(
            {"sub_network_config": sub_network_config, "parent_dir": checkpoint_dir},
            sub_network_dir / "lit_model.pth",
        )


@pytest.mark.parametrize("measure_latency", [True, False])
@pytest.mark.parametrize("measure_flops", [True, False])
@pytest.mark.parametrize(
    "checkpoint_mode",
    ["litgpt", "whittle", "whittle-minimalistic", "whittle-sub-network"],
)
def test_evaluate(checkpoint_dir, checkpoint_mode, measure_flops, measure_latency):
    sub_network_dir = pathlib.Path(checkpoint_dir) / "sub_network"
    sub_network_dir.mkdir(parents=True, exist_ok=True)

    setup_checkpoint_dir(checkpoint_dir, sub_network_dir, checkpoint_mode)

    evaluate_network.convert_and_evaluate = convert_and_evaluate_mock

    evaluate_network.setup(
        sub_network_dir,
        measure_latency=measure_latency,
        measure_flops=measure_flops,
        tasks="arc_easy",
        seed=42,
        num_fewshot=1,
        batch_size=1,
        device="cpu",
        limit=1,
    )

    metrics_path = sub_network_dir / "eval" / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    assert "parameters" in metrics
    assert "latency" in metrics if measure_latency else "latency" not in metrics
    assert "flops" in metrics if measure_flops else "flops" not in metrics

    for v in metrics.values():
        assert isinstance(v, (int, float))
