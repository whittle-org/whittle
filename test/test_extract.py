import torch

from litgpt.config import Config

from lobotomy.models.gpt import GPT
from lobotomy.models.gpt.extract import extract_sub_network


def test_extract_sub_network() -> None:
    config = Config.from_name("pythia-70m")
    config.fix_head_size = False

    super_network = GPT(config)
    sub_network_config = Config.from_name("pythia-14m")

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
    sub_network.eval()
    input = torch.randint(0, 512, (1, 20))
    out_super_net = super_network(input).detach()
    out_sub_net = sub_network(input).detach()
    assert torch.all(
        torch.round(out_sub_net, decimals=9) == torch.round(out_super_net, decimals=9)
    )
