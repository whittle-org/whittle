from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from litgpt import Config

from whittle.lora.config import LoRAConfig
from whittle.lora.lora_attention import CausalSelfAttention
from whittle.lora.lora_qkv_linear import LoRAQKVLinear
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
def test_zero_pad(qkv_config):
    config = lora_qkv_configs[qkv_config]
    seq_length, batch_size = 4, 2
    in_features, n_head, head_size, r = 128, 16, 2, 8
    enable_lora = config["enable_lora"]
    n_query_groups = config["n_query_groups"]
    out_features = (n_head + 2 * n_query_groups) * head_size

    lora_config = LoRAConfig(
        n_embd=out_features,
        n_head=n_head,
        n_query_groups=n_query_groups,
        head_size=head_size,
    )
    lora_config.fix_head_size = True

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

    q_shape = (batch_size, seq_length, n_head * head_size)
    k_shape = (batch_size, seq_length, n_query_groups * head_size)
    v_shape = (batch_size, seq_length, n_query_groups * head_size)
    # Init weights so that we know a) what's the original position in the pre-shuffle matrix
    # b) which one is from q, k or v
    qkv_weights = []
    for j, (i, shape) in enumerate(zip([1, 1000, 10000], [q_shape, k_shape, v_shape])):
        if enable_lora[j]:
            qkv_weights.append(
                (i + torch.arange(torch.prod(torch.tensor(shape)).item())).reshape(shape)
            )
    result = qkv.zero_pad(qkv_weights)
    qkv_group_size = qkv.sub_network_q_per_kv + 2

    # **Fix: Compute correct q_id indices**
    flat_result = result.flatten()
    q_ids = torch.arange(flat_result.shape[0])  # Corrected to match `flat_result` size

    group_ids = (q_ids // head_size) % qkv_group_size
    cond_q = group_ids < qkv_group_size - 2
    cond_k = group_ids == qkv_group_size - 2
    cond_v = group_ids > qkv_group_size - 2

    # Extract values based on conditions
    q_values, k_values, v_values = (
        flat_result[cond_q],
        flat_result[cond_k],
        flat_result[cond_v],
    )
    # now we check back the order
    qkv_idx = 0
    if enable_lora[0]:
        assert torch.all(q_values < 1000)
        assert q_values.numel() == qkv_weights[qkv_idx].numel()
        assert torch.allclose(q_values, qkv_weights[qkv_idx].flatten(), atol=1e-6)
        qkv_idx += 1
    else:
        assert torch.allclose(q_values, torch.zeros_like(q_values), atol=1e-6)
    if enable_lora[1]:
        assert torch.all((1000 <= k_values) & (k_values < 10000))
        assert k_values.numel() == qkv_weights[qkv_idx].numel()
        assert torch.allclose(k_values, qkv_weights[qkv_idx].flatten(), atol=1e-6)
        qkv_idx += 1
    else:
        assert torch.allclose(k_values, torch.zeros_like(k_values), atol=1e-6)
    if enable_lora[2]:
        assert torch.all(v_values >= 10000)
        assert v_values.numel() == qkv_weights[qkv_idx].numel()
        assert torch.allclose(v_values, qkv_weights[qkv_idx].flatten(), atol=1e-6)
    else:
        assert torch.allclose(v_values, torch.zeros_like(v_values), atol=1e-6)

    # Reshape and permute for final check
    result = result.view(
        batch_size, seq_length, n_query_groups, qkv_group_size, head_size
    )
    result = result.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

    q, k, v = result.split((qkv.sub_network_q_per_kv, 1, 1), dim=2)

    def check_qkv(post_split, weights):
        assert torch.allclose(
            weights, post_split.permute(0, 3, 1, 2, 4).flatten(), atol=1e-6
        )

    qkv_idx = 0
    if enable_lora[0]:
        check_qkv(q, qkv_weights[qkv_idx].flatten())
        qkv_idx += 1
    else:
        assert torch.allclose(q, torch.zeros_like(q), atol=1e-6)  # Q should be zero
    if enable_lora[1]:
        check_qkv(k, qkv_weights[qkv_idx].flatten())
        qkv_idx += 1
    else:
        assert torch.allclose(k, torch.zeros_like(k), atol=1e-6)  # K should be zero
    if enable_lora[2]:
        check_qkv(v, qkv_weights[qkv_idx].flatten())
    else:
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)  # V should be zero


@pytest.mark.parametrize("qkv_config", lora_qkv_configs.keys())
def test_zero_pad_sub_network(qkv_config):
    config = lora_qkv_configs[qkv_config]
    seq_length, batch_size = 4, 2
    in_features, n_head, head_size, r = 128, 16, 2, 8
    enable_lora = config["enable_lora"]
    n_query_groups = config["n_query_groups"]

    out_features = (n_head + 2 * n_query_groups) * head_size

    lora_config = LoRAConfig(
        n_embd=out_features,
        n_head=n_head,
        n_query_groups=n_query_groups,
        head_size=head_size,
    )
    lora_config.fix_head_size = True

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

    # determine sub-network configuration
    if n_query_groups == 1:  # MQA
        # super-network has 16 heads and 1 group
        sub_n_query_groups = 1
        sub_n_head = 8
    elif n_query_groups == n_head:  # MHA
        # super-network has 16 heads in 16 groups
        sub_n_query_groups = 8
        sub_n_head = 8
    else:  # GQA
        # super-network has 16 heads in 4 groups (4 heads per group)
        sub_n_query_groups = 2
        sub_n_head = 8  # 8/16 heads, 2/4 active groups (resulting into 2 heads per group)

    attn = CausalSelfAttention(lora_config, 0)
    attn.set_sub_network(lora_config.n_embd, sub_n_head, sub_n_query_groups, head_size)
    qkv_indices = attn.qkv_indices
    qkv.set_sub_network(
        in_features,
        (attn.sub_network_q_per_kv + 2) * head_size * sub_n_query_groups,
        qkv_indices,
        sub_network_query_groups=sub_n_query_groups,
        sub_network_n_head=sub_n_head,
        sub_network_head_size=head_size,
        sub_network_q_per_kv=attn.sub_network_q_per_kv,
    )

    q_shape = (
        batch_size,
        seq_length,
        qkv.sub_network_q_per_kv * sub_n_query_groups * head_size,
    )
    k_shape = (batch_size, seq_length, sub_n_query_groups * head_size)
    v_shape = (batch_size, seq_length, sub_n_query_groups * head_size)

    # **Generate sequential weights for easy tracking**
    qkv_weights = []
    for j, (i, shape) in enumerate(zip([1, 1000, 10000], [q_shape, k_shape, v_shape])):
        if enable_lora[j]:
            qkv_weights.append(
                (i + torch.arange(torch.prod(torch.tensor(shape)).item())).reshape(shape)
            )
    result = qkv.zero_pad(qkv_weights)
    qkv_weights = [qkv_weights[i].flatten() for i in range(len(qkv_weights))]
    qkv_group_size = qkv.sub_network_q_per_kv + 2

    # need the indices to check if correct values are at correct places
    flat_result = result.flatten()
    q_ids = torch.arange(flat_result.shape[0])  # Ensure correct shape

    group_ids = (q_ids // head_size) % qkv_group_size
    cond_q = group_ids < qkv.sub_network_q_per_kv
    cond_k = group_ids == qkv.sub_network_q_per_kv
    cond_v = group_ids > qkv.sub_network_q_per_kv

    # extract q,k,v values
    q_values = flat_result[cond_q]
    k_values = flat_result[cond_k]
    v_values = flat_result[cond_v]

    # assert conditions
    qkv_idx = 0
    if enable_lora[0]:
        assert torch.all(q_values < 1000)
        assert q_values.numel() == qkv_weights[qkv_idx].numel()
        assert torch.all(q_values == qkv_weights[qkv_idx])
        qkv_idx += 1
    else:
        assert torch.all(q_values == 0)
    if enable_lora[1]:
        assert torch.all((1000 <= k_values) & (k_values < 10000))
        assert k_values.numel() == qkv_weights[qkv_idx].numel()
        assert torch.all(k_values == qkv_weights[qkv_idx])
        qkv_idx += 1
    else:
        assert torch.all(k_values == 0)
    if enable_lora[2]:
        assert torch.all(v_values >= 10000)
        assert v_values.numel() == qkv_weights[qkv_idx].numel()
        assert torch.all(v_values == qkv_weights[qkv_idx])
    else:
        assert torch.all(v_values == 0)

    # one last check to see what happens after splitting it in causal self attention
    result = result.view(
        batch_size, seq_length, sub_n_query_groups, qkv_group_size, head_size
    )
    result = result.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

    # split batched computation into three
    q, k, v = result.split((qkv.sub_network_q_per_kv, 1, 1), dim=2)

    def check_qkv(post_split, weights):
        assert torch.allclose(
            weights, post_split.permute(0, 3, 1, 2, 4).flatten(), atol=1e-6
        )

    qkv_idx = 0
    if enable_lora[0]:
        check_qkv(q, qkv_weights[qkv_idx])
        qkv_idx += 1
    else:
        assert torch.allclose(q, torch.zeros_like(q), atol=1e-6)  # Q should be zero
    if enable_lora[1]:
        check_qkv(k, qkv_weights[qkv_idx])
        qkv_idx += 1
    else:
        assert torch.allclose(k, torch.zeros_like(k), atol=1e-6)  # K should be zero
    if enable_lora[2]:
        check_qkv(v, qkv_weights[qkv_idx])
    else:
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)  # V should be zero


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
    qkv_out_whittle = whittle_attention.attn(inp)
    print(qkv_out_whittle.shape)
    total_qkv = (whittle_attention.sub_network_q_per_kv) + 2
    qkv_out_whittle_reshaped = qkv_out_whittle.view(
        batch_size,
        seq_length,
        whittle_attention.sub_network_query_groups,
        total_qkv,
        whittle_attention.sub_network_head_size,
    )
    qkv_out_whittle_reshaped = qkv_out_whittle_reshaped.permute(0, 2, 3, 1, 4)
    q, k, v = qkv_out_whittle_reshaped.split(
        (whittle_attention.sub_network_q_per_kv, 1, 1), dim=2
    )
    nn.init.xavier_uniform_(qkv.lora_A)
    nn.init.xavier_uniform_(qkv.lora_B)
    after_A = torch.nn.functional.linear(
        qkv.lora_dropout(inp), qkv.lora_A[:, : qkv.sub_network_in_features]
    )
    after_B = qkv.conv1d(after_A.transpose(-2, -1), qkv.lora_B)
    lora = qkv.zero_pad([a.transpose(-2, -1) * qkv.scaling for a in after_B])
    print(lora.shape)
    lora_reshaped = lora.view(
        batch_size,
        seq_length,
        whittle_attention.sub_network_query_groups,
        total_qkv,
        whittle_attention.sub_network_head_size,
    )
    lora_reshaped = lora_reshaped.permute(0, 2, 3, 1, 4)
    q_lora, k_lora, v_lora = lora_reshaped.split(
        (whittle_attention.sub_network_q_per_kv, 1, 1), dim=2
    )
    qkv_out_lora = qkv_out_whittle + lora
    qkv_out_lora_reshaped = qkv_out_lora.view(
        batch_size,
        seq_length,
        whittle_attention.sub_network_query_groups,
        total_qkv,
        whittle_attention.sub_network_head_size,
    )
    qkv_out_lora_reshaped = qkv_out_lora_reshaped.permute(0, 2, 3, 1, 4)
    q_lora_and_whittle, k_lora_and_whittle, v_lora_and_whittle = (
        qkv_out_lora_reshaped.split((whittle_attention.sub_network_q_per_kv, 1, 1), dim=2)
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
