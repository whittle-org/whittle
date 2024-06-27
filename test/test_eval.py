import torch
import pytest
from litgpt.config import Config
import torch
import pathlib
import contextlib
import shutil
import json
from litgpt.model import GPT as LitGPT
from lobotomy.models.gpt import GPT
from lobotomy.models.gpt.extract import extract_sub_network
import litgpt.eval.evaluate as module
from litgpt.scripts.download import download_from_hub


@pytest.fixture(scope="session")
def checkpoint_dir_lit(tmp_path_factory):
    checkpoint_dir = tmp_path_factory.getbasetemp()
    download_from_hub(repo_id="EleutherAI/pythia-14m", checkpoint_dir=checkpoint_dir)
    return pathlib.Path(checkpoint_dir) / "EleutherAI" / "pythia-14m"


@pytest.fixture(scope="session")
def checkpoint_dir_lob(tmp_path_factory):
    checkpoint_dir = tmp_path_factory.getbasetemp()
    yield pathlib.Path(checkpoint_dir) / "EleutherAI" / "pythia-14m_lobotomy"


def test_lm_eval_harness(checkpoint_dir_lit, checkpoint_dir_lob) -> None:
    torch.manual_seed(0)
    config = Config.from_name("pythia-70m")
    config.fix_head_size = False

    super_network = GPT(config)
    sub_network_config = Config.from_name("pythia-14m")
    super_network.set_sub_network(
        sub_network_n_embd=sub_network_config.n_embd,
        sub_network_intermediate_size=[sub_network_config.intermediate_size]
        * sub_network_config.n_layer,
        sub_network_num_heads=[sub_network_config.n_head] * sub_network_config.n_layer,
        sub_network_n_layers=sub_network_config.n_layer,
    )

    # instantiate a new model
    module.convert_and_evaluate(
        checkpoint_dir_lit,
        out_dir="test_eval/",
        device=None,
        tasks="hellaswag",
        force_conversion=True,
        batch_size=2,
        limit=5,
    )

    results_json = "test_eval/results.json"
    with open(results_json, "r") as f:
        results = json.load(f)
    result_litgpt = results["results"]["hellaswag"]["acc,none"]
    sub_network = extract_sub_network(super_network, sub_network_config)
    sub_network.load_state_dict(torch.load(str(checkpoint_dir_lit / "lit_model.pth")))
    sub_network.eval()
    shutil.move(checkpoint_dir_lit, checkpoint_dir_lob)
    torch.save(sub_network.state_dict(), str(checkpoint_dir_lob / "lit_model.pth"))
    module.convert_and_evaluate(
        checkpoint_dir_lob,
        out_dir="test_eval/",
        device=None,
        tasks="hellaswag",
        force_conversion=True,
        batch_size=2,
        limit=5,
    )
    results_json = "test_eval/results.json"
    with open(results_json, "r") as f:
        results = json.load(f)
    result_lobotomy = results["results"]["hellaswag"]["acc,none"]
    assert result_litgpt == result_lobotomy
