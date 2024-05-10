import torch
from litgpt import Config
from litgpt.model import GPT as LitGPT
from lobotomy.models.gpt.model import GPT as LobotomyGPT
def test_checkpoint_loading():
    
    torch.manual_seed(0)
    config = Config.from_file("checkpoints/stabilityai/stablelm-base-alpha-3b/model_config.yaml")
    input_ids = torch.randint(0, config.vocab_size, (1, config.block_size))#.cuda()
    
    model = LitGPT(config)#.cuda()
    model.load_state_dict(torch.load("checkpoints/stabilityai/stablelm-base-alpha-3b/lit_model.pth"))
    # test output
    model.eval()
    output_lit = model(input_ids)
    #from litgpt.super_model import Config
    # pip install litgpt
    # litgpt download --repo_id stabilityai/stablelm-base-alpha-3b
    config = Config.from_file("checkpoints/stabilityai/stablelm-base-alpha-3b/model_config.yaml")
    config.fix_head_size = True
    model = LobotomyGPT(config)#.cuda()
    model.load_state_dict(torch.load("checkpoints/stabilityai/stablelm-base-alpha-3b/lit_model.pth"))
    # test output 
    model.eval()
    sample_intermediate_size = [4*config.n_embd for i in range(config.n_layer)]
    model.set_sub_network(config.n_embd, sample_intermediate_size, [config.n_head for i in range(config.n_layer)], config.n_layer)

    output_lobotomy = model(input_ids)
    assert torch.allclose(output_lit, output_lobotomy)
