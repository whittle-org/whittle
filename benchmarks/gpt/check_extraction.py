import torch
from pathlib import Path
from litgpt import Config
from lobotomy.models.gpt.model import GPT
from lobotomy.models.gpt.extract import extract_sub_network


super_network_type = "pythia-1b"
checkpoint_dir = Path(
    f"/home/ubuntu/sky_workdir/benchmarks/gpt/checkpoints/EleutherAI/{super_network_type}"
)
config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
config.fix_head_size = False

super_network = GPT(config)  # .cuda()
super_network.load_state_dict(torch.load(str(checkpoint_dir / "lit_model.pth")))

sub_network_config = Config.from_name("pythia-70m")
sub_network_config.fix_head_size = False

sub_network = GPT(sub_network_config)
checkpoint_dir_sub_network = Path(
    f"/home/ubuntu/sky_workdir/benchmarks/gpt/sub_network_checkpoints/sub_network_pythia-70m_from_super_net_pythia-1b"
)
sub_network.load_state_dict(
    torch.load(str(checkpoint_dir_sub_network / "lit_model.pth"))["model"]
)

super_network.eval()
super_network.set_sub_network(
    sub_network_n_embd=sub_network_config.n_embd,
    sub_network_intermediate_size=[sub_network_config.intermediate_size]
    * sub_network_config.n_layer,
    sub_network_num_heads=[sub_network_config.n_head] * sub_network_config.n_layer,
    sub_network_n_layers=sub_network_config.n_layer,
)
input_ids = torch.randint(0, config.vocab_size, (1, config.block_size))  # .cuda()

# instantiate a new model
out_super_net = super_network(input_ids).detach()
out_sub_net = sub_network(input_ids).detach()
assert torch.all(
    torch.round(out_sub_net, decimals=9) == torch.round(out_super_net, decimals=9)
)
