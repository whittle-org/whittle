from __future__ import annotations

import pytest
import torch
from litgpt import Config
from transformers.models.gemma2 import Gemma2Config, Gemma2ForCausalLM
from transformers.models.gpt_neox import GPTNeoXConfig, GPTNeoXForCausalLM
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

from whittle.convert import save_as_hf_checkpoint
from whittle.models.gpt import GPT


def _qwen3_hf(cfg):
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=cfg.padded_vocab_size,
            hidden_size=cfg.n_embd,
            head_dim=cfg.head_size,
            num_attention_heads=cfg.n_head,
            num_hidden_layers=cfg.n_layer,
            intermediate_size=cfg.intermediate_size,
            num_key_value_heads=cfg.n_query_groups,
            max_position_embeddings=cfg.block_size,
            rms_norm_eps=cfg.norm_eps,
            rope_theta=cfg.rope_base,
            tie_word_embeddings=False,
        )
    )


def _llama_hf(cfg):
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=cfg.padded_vocab_size,
            hidden_size=cfg.n_embd,
            head_dim=cfg.head_size,
            num_attention_heads=cfg.n_head,
            num_hidden_layers=cfg.n_layer,
            intermediate_size=cfg.intermediate_size,
            num_key_value_heads=cfg.n_query_groups,
            max_position_embeddings=cfg.block_size,
            rms_norm_eps=cfg.norm_eps,
            rope_theta=cfg.rope_base,
            tie_word_embeddings=False,
        )
    )


def _gemma2_hf(cfg):
    return Gemma2ForCausalLM(
        Gemma2Config(
            vocab_size=cfg.padded_vocab_size,
            hidden_size=cfg.n_embd,
            head_dim=cfg.head_size,
            num_attention_heads=cfg.n_head,
            num_hidden_layers=cfg.n_layer,
            intermediate_size=cfg.intermediate_size,
            num_key_value_heads=cfg.n_query_groups,
            max_position_embeddings=cfg.block_size,
            sliding_window=cfg.sliding_window_size,
            rms_norm_eps=cfg.norm_eps,
            rope_theta=cfg.rope_base,
            attention_bias=cfg.bias,
            tie_word_embeddings=True,
            hidden_act="gelu_pytorch_tanh",
            attn_logit_softcapping=cfg.attention_logit_softcapping,
            final_logit_softcapping=cfg.final_logit_softcapping,
            attn_implementation="eager",
            query_pre_attn_scalar=cfg.attention_scores_scalar,
        )
    )


def _pythia_hf(cfg):
    return GPTNeoXForCausalLM(
        GPTNeoXConfig(
            hidden_act="gelu",
            hidden_size=cfg.n_embd,
            num_attention_heads=cfg.n_head,
            num_hidden_layers=cfg.n_layer,
            intermediate_size=cfg.intermediate_size,
            max_position_embeddings=cfg.block_size,
            rotary_emb_base=cfg.rope_base,
            rotary_pct=cfg.rotary_percentage,
            vocab_size=cfg.padded_vocab_size,
            use_parallel_residual=cfg.parallel_residual,
        )
    )


MODEL_CASES = [
    pytest.param(
        "Qwen3-0.6B",
        dict(block_size=64, n_layer=16, n_head=16, n_embd=32, intermediate_size=86),
        _qwen3_hf,
        False,
        id="Qwen3-0.6B",
    ),
    pytest.param(
        "Llama-3-8B",
        dict(
            block_size=64,
            n_layer=16,
            n_embd=32,
            intermediate_size=86,
            padded_vocab_size=10000,
        ),
        _llama_hf,
        False,
        id="Llama-3-8B",
    ),
    pytest.param(
        "gemma-2-9b",
        dict(
            block_size=64,
            sliding_window_size=3,
            n_layer=16,
            n_embd=32,
            intermediate_size=86,
        ),
        _gemma2_hf,
        True,
        id="gemma-2-9b",
    ),
    pytest.param(
        "pythia-14m",
        dict(block_size=64, n_layer=16, n_head=4, n_embd=32, intermediate_size=128),
        _pythia_hf,
        False,
        id="pythia-14m",
    ),
]


@torch.inference_mode()
@pytest.mark.parametrize("model_name,config_kwargs,hf_builder,ties", MODEL_CASES)
def test_whittle_to_hf_matches(tmp_path, model_name, config_kwargs, hf_builder, ties):
    # 1) Load a small whittle model
    config = Config.from_name(model_name, **config_kwargs)
    config.fix_head_size = True
    whittle_model = GPT(config)
    if ties:
        whittle_model.lm_head.weight = whittle_model.transformer.wte.weight
    whittle_model.eval()

    # 2) Halve sub-network dimensions
    sub_network_config = {
        "embed_dim": config.n_embd // 2,
        "mlp_ratio": (config.intermediate_size // 2) / (config.n_embd // 2),
        "num_heads": config.n_head // 2,
        "depth": config.n_layer // 2,
        "n_query_groups": config.n_query_groups // 2,
        "head_size": config.head_size,
    }

    # 3) Save as a Hugging Face checkpoint
    save_as_hf_checkpoint(
        whittle_model, sub_network_config=sub_network_config, out_dir=tmp_path
    )

    # 4) Load the HF model
    sub_cfg = Config.from_file(tmp_path / "model_config.yaml")
    hf_model = hf_builder(sub_cfg)
    hf_model.load_state_dict(
        torch.load(tmp_path / "model.pth", weights_only=True), strict=False
    )
    hf_model.eval()

    if ties:
        # save_as_hf_checkpoint resets the super-net and drops tying.
        whittle_model.lm_head.weight = whittle_model.transformer.wte.weight

    # 5) Compare outputs
    x = torch.randint(
        0, config.n_embd, size=(2, 64)
    )  # use random input to avoid hitting tied weights
    whittle_model.set_sub_network(
        sub_network_n_embd=sub_cfg.n_embd,
        sub_network_intermediate_size=sub_cfg.intermediate_size,
        sub_network_num_heads=sub_cfg.n_head,
        sub_network_n_layers=sub_cfg.n_layer,
        sub_network_query_groups=sub_cfg.n_query_groups,
        sub_network_head_size=sub_cfg.head_size,
    )
    out_whittle = whittle_model(x)
    out_hf = hf_model(x)["logits"]

    torch.testing.assert_close(out_whittle, out_hf, atol=1e-6, rtol=1e-6)
