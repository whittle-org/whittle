from lobotomy.models.gpt import GPT
from litgpt import Config
import torch
from litgpt.model import GPT as LitGPT
def test_gpt():
    config = Config()
    config.padded_vocab_size = 512
    config.n_embd = 64
    config.intermediate_size = 64*4
    config.n_head = 8
    config.n_query_groups = 4
    config.head_size = 8
    config.n_layer = 8
    config.block_size = 512
    config.norm_class_name = "RMSNorm"
    config.mlp_class_name = "LLaMAMLP"
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)
    config.norm_eps = 1e-5
    config.lm_head_bias = True
    config.fix_head_size = False
    gpt = GPT(config)
    gpt.transformer.wte.weight.data = torch.ones_like(gpt.transformer.wte.weight.data)
    gpt.lm_head.weight.data = torch.ones_like(gpt.lm_head.weight.data)
    gpt.lm_head.bias.data = torch.ones_like(gpt.lm_head.bias.data)
    gpt.transformer.ln_f.weight.data = torch.ones_like(gpt.transformer.ln_f.weight.data)
    for block in gpt.transformer.h:
        block.attn.attn.weight.data = torch.ones_like(block.attn.attn.weight.data)
        block.attn.attn.bias.data = torch.ones_like(block.attn.attn.bias.data)
        block.attn.proj.bias.data = torch.ones_like(block.attn.proj.bias.data)
        block.attn.proj.weight.data = torch.ones_like(block.attn.proj.weight.data)
        block.mlp.fc_1.weight.data = torch.ones_like(block.mlp.fc_1.weight.data)
        block.mlp.fc_1.bias.data = torch.ones_like(block.mlp.fc_1.bias.data)
        block.mlp.fc_2.weight.data = torch.ones_like(block.mlp.fc_2.weight.data)
        block.mlp.fc_2.bias.data = torch.ones_like(block.mlp.fc_2.bias.data)
        block.mlp.proj.weight.data = torch.ones_like(block.mlp.proj.weight.data)
        block.mlp.proj.bias.data = torch.ones_like(block.mlp.proj.bias.data)
        block.norm_1.weight.data = torch.ones_like(block.norm_1.weight.data)
        block.norm_2.weight.data = torch.ones_like(block.norm_2.weight.data)
        
    gpt.reset_super_network()
    input = torch.randint(0, 512, (8, 512))
    out_large = gpt(input)
    assert out_large.shape == (8, 512, 512)
    gpt.set_sub_network(sub_network_n_embd=32, sub_network_intermediate_size=[32*4 for i in range(4)], sub_network_num_heads=[4 for i in range(4)], sub_network_n_layers=4)
    out_small = gpt(input)
    assert out_small.shape == (8, 512, 512)

    lit_gpt = LitGPT(config)
    lit_gpt.lm_head.weight.data = gpt.lm_head.weight.data
    lit_gpt.lm_head.bias.data = gpt.lm_head.bias.data
    lit_gpt.transformer.wte.weight.data = gpt.transformer.wte.weight.data
    lit_gpt.transformer.ln_f.weight.data = gpt.transformer.ln_f.weight.data
    for block in lit_gpt.transformer.h:
        block.attn.attn.weight.data = torch.ones_like(block.attn.attn.weight.data)
        block.attn.attn.bias.data = torch.ones_like(block.attn.attn.bias.data)
        block.attn.proj.bias.data = torch.ones_like(block.attn.proj.bias.data)
        block.attn.proj.weight.data = torch.ones_like(block.attn.proj.weight.data)
        block.mlp.fc_1.weight.data = torch.ones_like(block.mlp.fc_1.weight.data)
        block.mlp.fc_1.bias.data = torch.ones_like(block.mlp.fc_1.bias.data)
        block.mlp.fc_2.weight.data = torch.ones_like(block.mlp.fc_2.weight.data)
        block.mlp.fc_2.bias.data = torch.ones_like(block.mlp.fc_2.bias.data)
        block.mlp.proj.weight.data = torch.ones_like(block.mlp.proj.weight.data)
        block.mlp.proj.bias.data = torch.ones_like(block.mlp.proj.bias.data)
        block.norm_1.weight.data = torch.ones_like(block.norm_1.weight.data)
        block.norm_2.weight.data = torch.ones_like(block.norm_2.weight.data)
    out_lit_large = lit_gpt(input)
    assert torch.all(out_lit_large == out_large)

    config.n_embd = 32
    config.n_head = 4
    config.n_query_groups = 2
    config.intermediate_size = 32*4
    config.n_layer = 4
    lit_gpt_small = LitGPT(config)
    lit_gpt_small.lm_head.weight.data = torch.ones_like(lit_gpt_small.lm_head.weight.data)
    lit_gpt_small.lm_head.bias.data = torch.ones_like(lit_gpt_small.lm_head.bias.data)
    lit_gpt_small.transformer.wte.weight.data = torch.ones_like(lit_gpt_small.transformer.wte.weight.data)
    lit_gpt_small.transformer.ln_f.weight.data = torch.ones_like(lit_gpt_small.transformer.ln_f.weight.data)
    for block in lit_gpt_small.transformer.h:
        block.attn.attn.weight.data = torch.ones_like(block.attn.attn.weight.data)
        block.attn.attn.bias.data = torch.ones_like(block.attn.attn.bias.data)
        block.attn.proj.bias.data = torch.ones_like(block.attn.proj.bias.data)
        block.attn.proj.weight.data = torch.ones_like(block.attn.proj.weight.data)
        block.mlp.fc_1.weight.data = torch.ones_like(block.mlp.fc_1.weight.data)
        block.mlp.fc_1.bias.data = torch.ones_like(block.mlp.fc_1.bias.data)
        block.mlp.fc_2.weight.data = torch.ones_like(block.mlp.fc_2.weight.data)
        block.mlp.fc_2.bias.data = torch.ones_like(block.mlp.fc_2.bias.data)
        block.mlp.proj.weight.data = torch.ones_like(block.mlp.proj.weight.data)
        block.mlp.proj.bias.data = torch.ones_like(block.mlp.proj.bias.data)
    out_lit_small = lit_gpt_small(input)
    assert torch.all(out_lit_small == out_small)




