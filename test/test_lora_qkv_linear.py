from __future__ import annotations

import pytest
import torch

from whittle.lora.config import LoRAConfig
from whittle.lora.lora_attention import CausalSelfAttention
from whittle.lora.lora_qkv_linear import LoRAQKVLinear

lora_qkv_configs = {
    "mha_enable_qkv": {
        "n_query_groups": 16,
        "enable_lora": [True, True, True],
    },
    "gqa_enable_qkv": {
        "n_query_groups": 4,
        "enable_lora": [True, True, True],
    },
    "mqa_enable_qkv": {
        "n_query_groups": 1,
        "enable_lora": [True, True, True],
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
    qkv_weights = [
        (i + torch.arange(torch.prod(torch.tensor(shape)).item())).reshape(shape)
        for i, shape in zip([1, 1000, 10000], [q_shape, k_shape, v_shape])
    ]

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
    assert torch.all(q_values < 1000)
    assert torch.allclose(
        q_values, qkv_weights[0].flatten()[: q_values.numel()], atol=1e-6
    )

    assert torch.all((1000 <= k_values) & (k_values < 10000))
    assert torch.allclose(
        k_values, qkv_weights[1].flatten()[: k_values.numel()], atol=1e-6
    )

    assert torch.all(v_values >= 10000)
    assert torch.allclose(
        v_values, qkv_weights[2].flatten()[: v_values.numel()], atol=1e-6
    )

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

    check_qkv(q, qkv_weights[0].flatten())
    check_qkv(k, qkv_weights[1].flatten())
    check_qkv(v, qkv_weights[2].flatten())


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
    qkv_indices = attn.get_qkv_indices()
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
    qkv_weights = [
        (i + torch.arange(torch.prod(torch.tensor(shape)).item())).reshape(shape)
        for i, shape in zip([1, 1000, 10000], [q_shape, k_shape, v_shape])
    ]

    result = qkv.zero_pad(qkv_weights)

    # now we check back the order
    qkv_weights = [
        qkv_weights[0].flatten(),
        qkv_weights[1].flatten(),
        qkv_weights[2].flatten(),
    ]
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
    assert torch.all(q_values < 1000)
    assert torch.all(q_values == qkv_weights[0][: q_values.numel()])

    assert torch.all((1000 <= k_values) & (k_values < 10000))
    assert torch.all(k_values == qkv_weights[1][: k_values.numel()])

    assert torch.all(v_values >= 10000)
    assert torch.all(v_values == qkv_weights[2][: v_values.numel()])

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

    check_qkv(q, qkv_weights[0])
    check_qkv(k, qkv_weights[1])
    check_qkv(v, qkv_weights[2])
