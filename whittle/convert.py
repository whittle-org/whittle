from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

import torch
from litgpt import Config
from litgpt.model import GPT as LitGPT
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import save_config

from whittle.convert_to_litgpt import setup as convert_to_litgpt_setup
from whittle.models.gpt import GPT
from whittle.models.gpt.checkpoint import save_sub_network


def create_litgpt_config_for_subnet(supernet):
    config = copy.deepcopy(supernet.config)
    config.fix_head_size = True
    config.n_embd = supernet.sub_network_n_embd
    config.intermediate_size = supernet.sub_network_intermediate_size
    config.n_head = supernet.sub_network_num_heads
    config.n_layer = supernet.sub_network_n_layers
    config.head_size = supernet.sub_network_head_size
    config.rope_n_elem = supernet.sub_network_rope_n_elem
    config.n_query_groups = supernet.sub_network_query_groups
    # Whittle's attention recomputes the softmax scaler from the sub-network
    # dimensions when `attention_scores_scalar` is set; mirror that so the
    # converted LitGPT model uses the same scale.
    if supernet.config.attention_scores_scalar:
        config.attention_scores_scalar = (
            supernet.sub_network_n_embd // supernet.sub_network_num_heads
        )
    return config


def copy_weights_to_litgpt(whittle_model, lit_model):
    def _copy_weights_and_biases(whittle_module, lit_module):
        if hasattr(whittle_module, "extract_weights"):
            W, b = whittle_module.extract_weights()
        else:
            print(f"{whittle_module} has no method extract_weights")

        if hasattr(lit_module, "weight"):
            w_data = W.data
            # Whittle's RMSNorm.extract_weights folds `add_unit_offset` into the
            # returned weight, but LitGPT's RMSNorm re-applies `(1 + weight)` in
            # forward — subtract 1 to avoid applying the offset twice.
            if getattr(lit_module, "add_unit_offset", False):
                w_data = w_data - 1
            lit_module.weight.data = w_data

        if hasattr(lit_module, "bias"):
            if b is None:
                lit_module.bias = None
            else:
                lit_module.bias.data = b.data

    _copy_weights_and_biases(whittle_model.lm_head, lit_model.lm_head)
    _copy_weights_and_biases(whittle_model.transformer.wte, lit_model.transformer.wte)
    _copy_weights_and_biases(whittle_model.transformer.ln_f, lit_model.transformer.ln_f)

    chosen_layers = (
        range(whittle_model.sub_network_n_layers)
        if whittle_model.sampled_layer_indices is None
        else whittle_model.sampled_layer_indices
    )
    whittle_blocks = [whittle_model.transformer.h[idx] for idx in chosen_layers]
    litgpt_blocks = lit_model.transformer.h

    for whittle_block, litgpt_block in zip(whittle_blocks, litgpt_blocks):
        _copy_weights_and_biases(whittle_block.norm_1, litgpt_block.norm_1)
        _copy_weights_and_biases(whittle_block.attn.qkv, litgpt_block.attn.qkv)
        _copy_weights_and_biases(whittle_block.attn.proj, litgpt_block.attn.proj)
        if whittle_block.attn.config.norm_qk:
            _copy_weights_and_biases(whittle_block.attn.norm_q, litgpt_block.attn.norm_q)
            _copy_weights_and_biases(whittle_block.attn.norm_k, litgpt_block.attn.norm_k)
        _copy_weights_and_biases(
            whittle_block.post_attention_norm, litgpt_block.post_attention_norm
        )
        _copy_weights_and_biases(whittle_block.norm_2, litgpt_block.norm_2)

        if hasattr(whittle_block.mlp, "fc"):
            _copy_weights_and_biases(whittle_block.mlp.fc, litgpt_block.mlp.fc)
        if hasattr(whittle_block.mlp, "fc_1"):
            _copy_weights_and_biases(whittle_block.mlp.fc_1, litgpt_block.mlp.fc_1)
        if hasattr(whittle_block.mlp, "fc_2"):
            _copy_weights_and_biases(whittle_block.mlp.fc_2, litgpt_block.mlp.fc_2)

        _copy_weights_and_biases(whittle_block.mlp.proj, litgpt_block.mlp.proj)
        _copy_weights_and_biases(whittle_block.post_mlp_norm, litgpt_block.post_mlp_norm)


def convert_subnet_to_litgpt(whittle_model, subnet_config):
    whittle_model.set_sub_network(**subnet_config)
    litgpt_config = create_litgpt_config_for_subnet(whittle_model)
    lit_model = LitGPT(litgpt_config)
    copy_weights_to_litgpt(whittle_model, lit_model)
    whittle_model.reset_super_network()
    return lit_model


def save_as_hf_checkpoint(
    whittle_model: GPT,
    sub_network_config: dict[str, Any],
    out_dir: Path | str,
) -> None:
    """
    Convert a Whittle super-network + sub-network config into a Hugging Face
    checkpoint on disk.

    Staged internally through a temp directory:
      1. Save the super-network as a parent LitGPT checkpoint.
      2. Extract the sub-network with `save_sub_network` (format b).
      3. Normalize to a plain LitGPT checkpoint via `whittle.convert_to_litgpt`.
      4. Convert LitGPT -> HF via `litgpt.scripts.convert_lit_checkpoint`.

    The final `out_dir` contains `model.pth` (HF state dict) and
    `model_config.yaml` (the sub-network's LitGPT config).

    Arguments:
        whittle_model: The Whittle super-network to extract from.
        sub_network_config: Sub-network config in the `select_sub_network`
            schema (`embed_dim`, `mlp_ratio`, `num_heads`, `depth`,
            `n_query_groups`, `head_size`).
        out_dir: Directory to write the HF checkpoint into. Created if missing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        parent_dir = tmp / "parent"
        parent_dir.mkdir()
        save_config(whittle_model.config, parent_dir)
        torch.save(whittle_model.state_dict(), parent_dir / "lit_model.pth")

        subnet_dir = tmp / "subnet"
        save_sub_network(
            whittle_model,
            checkpoint_dir=parent_dir,
            save_dir=subnet_dir,
            sub_network_config=sub_network_config,
            save_checkpoints=True,
            copy_config_files=False,
        )
        whittle_model.reset_super_network()

        litgpt_dir = tmp / "litgpt"
        litgpt_dir.mkdir()
        convert_to_litgpt_setup(
            sub_network_dir=subnet_dir,
            out_dir=litgpt_dir,
            no_model_key=True,
        )

        convert_lit_checkpoint(checkpoint_dir=litgpt_dir, output_dir=out_dir)

        # Keep the sub-network's model_config.yaml alongside the HF weights so
        # callers can reconstruct the matching HF config.
        save_config(Config.from_file(litgpt_dir / "model_config.yaml"), out_dir)
