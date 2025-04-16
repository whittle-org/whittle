from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from litgpt import Config

from whittle.lora_model.config import LoRAConfig
from whittle.lora_model.lora_attention import CausalSelfAttention
from whittle.lora_model.lora_qkv_linear import LoRAQKVLinear
from whittle.models.gpt.blocks import CausalSelfAttention as CausalSelfAttentionWhittle

lora_qkv_configs = {
    "mha_enable_qk": {
        "n_query_groups": 16,
        "enable_lora": [True, True, False],
    },
    "mha_enable_qv": {
        "n_query_groups": 16,
        "enable_lora": [True, False, True],
    },
    "mha_enable_kv": {
        "n_query_groups": 16,
        "enable_lora": [False, True, True],
    },
    "mha_enable_qkv": {
        "n_query_groups": 16,
        "enable_lora": [True, True, True],
    },
    "mha_enable_q": {
        "n_query_groups": 16,
        "enable_lora": [True, False, False],
    },
    "mha_enable_k": {
        "n_query_groups": 16,
        "enable_lora": [False, True, False],
    },
    "mha_enable_v": {
        "n_query_groups": 16,
        "enable_lora": [False, False, True],
    },
    "gqa_enable_qkv": {
        "n_query_groups": 4,
        "enable_lora": [True, True, True],
    },
    "gqa_enable_qk": {
        "n_query_groups": 4,
        "enable_lora": [True, True, False],
    },
    "gqa_enable_qv": {
        "n_query_groups": 4,
        "enable_lora": [True, False, True],
    },
    "gqa_enable_kv": {
        "n_query_groups": 4,
        "enable_lora": [False, True, True],
    },
    "gqa_enable_q": {
        "n_query_groups": 4,
        "enable_lora": [True, False, False],
    },
    "gqa_enable_k": {
        "n_query_groups": 4,
        "enable_lora": [False, True, False],
    },
    "gqa_enable_v": {
        "n_query_groups": 4,
        "enable_lora": [False, False, True],
    },
    "mqa_enable_qkv": {
        "n_query_groups": 1,
        "enable_lora": [True, True, True],
    },
    "mqa_enable_qk": {
        "n_query_groups": 1,
        "enable_lora": [True, True, False],
    },
    "mqa_enable_qv": {
        "n_query_groups": 1,
        "enable_lora": [True, False, True],
    },
    "mqa_enable_kv": {
        "n_query_groups": 1,
        "enable_lora": [False, True, True],
    },
    "mqa_enable_q": {
        "n_query_groups": 1,
        "enable_lora": [True, False, False],
    },
    "mqa_enable_k": {
        "n_query_groups": 1,
        "enable_lora": [False, True, False],
    },
    "mqa_enable_v": {
        "n_query_groups": 1,
        "enable_lora": [False, False, True],
    },
}


@pytest.mark.parametrize("qkv_config", lora_qkv_configs.keys())
def test_qkv_linear_forward(qkv_config):
    config = lora_qkv_configs[qkv_config]
    seq_length, batch_size = 4, 2
    in_features, n_head, head_size, r = 128, 16, 2, 8
    enable_lora = config["enable_lora"]
    n_query_groups = config["n_query_groups"]
    sub_network_n_head = 8
    if n_head == n_query_groups:
        sub_network_query_groups = sub_network_n_head
    elif n_query_groups == 1:
        sub_network_query_groups = 1
    else:
        sub_network_query_groups = 2

    out_features = (n_head + 2 * n_query_groups) * head_size
    lora_config = LoRAConfig(
        n_embd=in_features,
        n_head=n_head,
        n_query_groups=n_query_groups,
        head_size=head_size,
    )
    lora_config.fix_head_size = True
    whittle_config = Config(
        n_embd=in_features,
        n_head=n_head,
        n_query_groups=n_query_groups,
        head_size=head_size,
    )
    whittle_attention = CausalSelfAttentionWhittle(whittle_config, 0)
    qkv = LoRAQKVLinear(
        lora_config,
        in_features,
        out_features,
        head_size,
        n_head,
        n_query_groups,
        True,
        r=r,
        enable_lora=enable_lora,
    )
    qkv.reset_super_network()
    whittle_attention.reset_super_network()
    whittle_attention.set_sub_network(
        whittle_config.n_embd,
        sub_network_n_head,
        sub_network_query_groups,
        head_size,
    )
    sub_network_out_features = (
        (whittle_attention.sub_network_q_per_kv + 2)
        * head_size
        * sub_network_query_groups
    )
    qkv.set_sub_network(
        in_features,
        sub_network_out_features,
        whittle_attention.qkv_indices,
        sub_network_query_groups=sub_network_query_groups,
        sub_network_n_head=sub_network_n_head,
        sub_network_head_size=head_size,
        sub_network_q_per_kv=whittle_attention.sub_network_q_per_kv,
    )
    inp = torch.rand(batch_size, seq_length, whittle_config.n_embd)
    qkv_out_whittle = whittle_attention.qkv(inp)

    query_size = whittle_attention.sub_network_q_per_kv * whittle_attention.sub_network_head_size * whittle_attention.sub_network_query_groups
    key_size = whittle_attention.sub_network_head_size * whittle_attention.sub_network_query_groups
    value_size = whittle_attention.sub_network_head_size * whittle_attention.sub_network_query_groups
    q, k, v = qkv_out_whittle.split(
        (query_size, key_size, value_size), dim=-1
    )
    nn.init.xavier_uniform_(qkv.lora_A)
    nn.init.xavier_uniform_(qkv.lora_B)
    after_A = torch.nn.functional.linear(
        qkv.lora_dropout(inp), qkv.lora_A[:, : qkv.sub_network_in_features]
    )
    after_B = qkv.conv1d(after_A.transpose(-2, -1), qkv.lora_B)
    lora = qkv.zero_pad([a.transpose(-2, -1) * qkv.scaling for a in after_B])
    q_lora, k_lora, v_lora = lora.split((query_size, key_size, value_size), dim=-1)
    qkv_out_lora = qkv_out_whittle + lora
    q_lora_and_whittle, k_lora_and_whittle, v_lora_and_whittle = (
        qkv_out_lora.split((query_size, key_size, value_size), dim=-1)
    )
    assert torch.allclose(q_lora_and_whittle - q_lora, q, atol=1e-6)
    assert torch.allclose(k_lora_and_whittle - k_lora, k, atol=1e-6)
    assert torch.allclose(v_lora_and_whittle - v_lora, v, atol=1e-6)
    if not enable_lora[0]:
        assert torch.allclose(q_lora_and_whittle, q, atol=1e-6)
    if not enable_lora[1]:
        assert torch.allclose(k_lora_and_whittle, k, atol=1e-6)
    if not enable_lora[2]:
        assert torch.allclose(v_lora_and_whittle, v, atol=1e-6)
