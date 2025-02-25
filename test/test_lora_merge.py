import torch

import pytest
import pathlib
from copy import deepcopy
from whittle.lora.utils import merge_lora_weights, merge_lora
from whittle.lora.config import LoRAConfig as Config
from whittle.lora.lora_gpt import GPT as LoRAGPT
from whittle.models.gpt import GPT
from litgpt.lora import mark_only_lora_as_trainable, LoRALayer, lora_filter
from litgpt.utils import save_config
from whittle.lora.lora_qkv_linear import LoRAQKVLinear
from whittle.lora.lora_embedding import LoRAEmbedding
import os
from whittle.lora.lora_embedding import LoRAEmbedding
from whittle.lora.lora_qkv_linear import LoRAQKVLinear
from whittle.lora.lora_linear import LoRALinearProj, LoRALinearQKV, LoRALinear


def test_lora_merge():
    config = Config(
        n_layer=1,
        n_head=2,
        n_embd=8,
        block_size=8,
        vocab_size=8,
        lora_r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        lora_query=True,
        lora_value=True,
        lora_key=True,
        lora_projection=True,
        lora_emb=True,
    )
    config.lm_head_bias = True
    config.fix_head_size = True

    model = LoRAGPT(config)    

    model.reset_super_network()
    model.train()
    attn_proj = model.transformer.h[0].attn.proj

    initial_weight = attn_proj.linear.weight.clone().detach()
    assert torch.equal(attn_proj.linear.weight, initial_weight)

    # perform an update to the LoRA weights
    mark_only_lora_as_trainable(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    y = model(torch.randint(0, 8, size=(2, 4), dtype=torch.int64))
    y.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    # the weight remains unchanged (only lora A and B change)
    assert torch.equal(attn_proj.linear.weight, initial_weight)

    model.eval()
    old_model = deepcopy(model)

    # calling merge() multiple times in a row should not merge multiple times
    merge_lora_weights(model)
    assert attn_proj.merged
    weight_after = attn_proj.linear.weight.clone()
    merge_lora_weights(model)
    merge_lora_weights(model)
    assert torch.equal(attn_proj.linear.weight, weight_after)

    # check that `W_after = W_initial + (A x B)`
    delta_w = attn_proj.get_lora_AB()
    torch.testing.assert_close(weight_after, initial_weight + delta_w)

    for (name, module), (old_name, old_module) in zip(model.named_modules(), old_model.named_modules()):
        assert name == old_name
        if isinstance(module, LoRALayer):
            if isinstance(module, LoRAQKVLinear):
                weight = module.linear.linear.weight
                old_weight = old_module.linear.linear.weight
            elif isinstance(module, LoRAEmbedding):
                weight = module.embedding.weight
                old_weight = old_module.embedding.weight
            else:
                weight = module.linear.weight
                old_weight = old_module.linear.weight

            if module.r > 0:
                delta_w = module.get_lora_AB()
                torch.testing.assert_close(weight, old_weight + delta_w)

@pytest.fixture(scope="session")
def checkpoint_dir(tmp_path_factory):
    checkpoint_dir = tmp_path_factory.getbasetemp()
    return pathlib.Path(checkpoint_dir)


def test_merge_extract(checkpoint_dir):
    config = Config()
    config.padded_vocab_size = 128
    config.n_embd = 64
    config.intermediate_size = 64 * 4
    config.n_head = 8
    config.n_query_groups = 4
    config.head_size = 8
    config.n_layer = 2
    config.block_size = 128
    config.norm_class_name = "RMSNorm"
    config.mlp_class_name = "LLaMAMLP"
    config.rope_n_elem = int(config.rotary_percentage * config.head_size)
    config.norm_eps = 1e-5
    config.lm_head_bias = True
    config.fix_head_size = False

    config.lora_r = 8
    config.lora_alpha = 8
    config.lora_dropout = 0.1
    config.lora_query = True
    config.lora_value = True
    config.lora_projection = True
    config.lora_emb = True

    torch.manual_seed(0)

    model = LoRAGPT(config)
    model.transformer.wte.embedding.weight.data = torch.randn_like(
        model.transformer.wte.embedding.weight.data
    )
    model.lm_head.linear.weight.data = torch.randn_like(model.lm_head.linear.weight.data)
    model.lm_head.linear.bias.data = torch.randn_like(model.lm_head.linear.bias.data)
    model.transformer.ln_f.weight.data = torch.randn_like(model.transformer.ln_f.weight.data)

    for block in model.transformer.h:
        block.attn.attn.linear.linear.weight.data = torch.randn_like(
            block.attn.attn.linear.linear.weight.data
        )
        block.attn.attn.linear.linear.bias.data = torch.randn_like(
            block.attn.attn.linear.linear.bias.data
        )
        block.attn.proj.linear.bias.data = torch.randn_like(
            block.attn.proj.linear.bias.data
        )
        block.attn.proj.linear.weight.data = torch.randn_like(
            block.attn.proj.linear.weight.data
        )
        block.mlp.fc_1.linear.weight.data = torch.randn_like(
            block.mlp.fc_1.linear.weight.data
        )
        block.mlp.fc_1.linear.bias.data = torch.randn_like(
            block.mlp.fc_1.linear.bias.data
        )
        block.mlp.fc_2.linear.weight.data = torch.randn_like(
            block.mlp.fc_2.linear.weight.data
        )
        block.mlp.fc_2.linear.bias.data = torch.randn_like(
            block.mlp.fc_2.linear.bias.data
        )
        block.mlp.proj.linear.weight.data = torch.randn_like(
            block.mlp.proj.linear.weight.data
        )
        block.mlp.proj.linear.bias.data = torch.randn_like(
            block.mlp.proj.linear.bias.data
        )
        block.norm_1.weight.data = torch.randn_like(block.norm_1.weight.data)
        block.norm_2.weight.data = torch.randn_like(block.norm_2.weight.data)

    model.reset_super_network()
    
    state_dict = {
        k: v for k, v in model.state_dict().items()
        if not lora_filter(k, v)
    }
    torch.save(state_dict, checkpoint_dir / "lit_model.pth")

    (checkpoint_dir / "tokenizer.json").touch()
    (checkpoint_dir / "tokenizer_config.json").touch()
    (checkpoint_dir / "model_config.yaml").touch()

    model.train()

    print(model.sub_network_head_size)
    print(config.head_size)
    print(model.transformer.h[0].attn.sub_network_head_size)
    # perform an update to the LoRA weights
    mark_only_lora_as_trainable(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    ints = torch.randint(0, 64, size=(2, 4), dtype=torch.int64)
    y = model(ints)
    model_o = y
    y.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    model.eval()
    pre_model = deepcopy(model)
    model.train()
    pre_model.eval()

    state_dict = {
        k: v for k, v in model.state_dict().items()
        if lora_filter(k, v)
    }
    
    lora_checkpoint_dir = pathlib.Path('lora/')
    lora_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, lora_checkpoint_dir / "lit_model.pth.lora")

    # seed torch
    torch.manual_seed(40)
    random_input = torch.randint(0, 64, size=(2, 4), dtype=torch.int64)
    pre_merge_y = pre_model(random_input)

    if os.path.exists(lora_checkpoint_dir / "lit_model.pth"):
        os.remove(lora_checkpoint_dir / "lit_model.pth")
    (lora_checkpoint_dir / "tokenizer.json").touch()
    (lora_checkpoint_dir / "tokenizer_config.json").touch()
    save_config(model.config, lora_checkpoint_dir)
    merge_lora(lora_checkpoint_dir, checkpoint_dir)

    # load merged weights
    post_merge_model = GPT(config)
    post_merge_model.load_state_dict(torch.load(lora_checkpoint_dir / "lit_model.pth"))

    post_merge_model.eval()
    post_merge_y = post_merge_model(random_input)

    pre_modules = []
    pre_modules.append(pre_model.lm_head)
    pre_modules.append(pre_model.transformer.wte)
    for block in pre_model.transformer.h:
        pre_modules.append(block.attn.attn)
        pre_modules.append(block.attn.proj)
        # TODO also mlp and lm_head

    post_modules = []
    post_modules.append(post_merge_model.lm_head)
    post_modules.append(post_merge_model.transformer.wte)
    for block in post_merge_model.transformer.h:
        post_modules.append(block.attn.attn)
        post_modules.append(block.attn.proj)
        # TODO also mlp and lm_head

    # TODO bf16

    for pre_module, post_module in zip(pre_modules, post_modules):
        if isinstance(pre_module, (LoRALinear, LoRALinearProj)):
            input_data = torch.randn(8, pre_module.in_features)
        elif isinstance(pre_module, LoRAQKVLinear):
            input_data = torch.randn((2, 4, 64))
        elif isinstance(pre_module, LoRAEmbedding):
            input_data = torch.randint(0, 64, size=(2, 4), dtype=torch.int64)
        else:
            continue
        pre_y = pre_module(input_data)
        post_y = post_module(input_data)
        assert torch.allclose(pre_y, post_y)

    assert torch.allclose(pre_merge_y, post_merge_y)
