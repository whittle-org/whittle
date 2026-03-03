import os
import pytest
import json
import torch

from litgpt import Config
from whittle.models.gpt import GPT
from whittle.convert import convert_subnet_to_litgpt


@pytest.fixture(scope="session")
def supernet():
    model_id = "EleutherAI/pythia-14m"
    config_path = os.path.join("checkpoints", model_id, "model_config.yaml")
    config = Config.from_file(config_path)
    config.fix_head_size = True
    supernet = GPT(config)
    return supernet


CONFIG_IDS = list(range(100))
SEARCH_SPACE_TYPES = ["coarse", "finegrained"]


@pytest.mark.parametrize("config_id", CONFIG_IDS)
@pytest.mark.parametrize("search_space_type", SEARCH_SPACE_TYPES)
def test_equivalence(supernet, search_space_type, config_id):

    with open(f"test/config_{search_space_type}.json", "r") as f:
        subnet_config = json.load(f)[config_id]

    lit_model = convert_subnet_to_litgpt(supernet, subnet_config)

    x = torch.tensor([[9856, 23, 491, 1536, 304, 1234]], dtype=torch.int32)
    supernet.set_sub_network(**subnet_config)
    out_whittle = supernet(x)
    lit_out = lit_model(x)

    assert torch.allclose(out_whittle, lit_out)
