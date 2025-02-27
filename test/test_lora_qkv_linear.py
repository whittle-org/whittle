from __future__ import annotations

import pytest
import torch

from whittle.lora.config import LoRAConfig
from whittle.lora.lora_attention import CausalSelfAttention
from whittle.lora.lora_qkv_linear import LoRAQKVLinear


@pytest.mark.parametrize("n_query_groups", [1, 4, 16])
def test_zero_pad(n_query_groups):
    seq_length = 4
    batch_size = 2

    in_features = 128
    n_head = 16
    head_size = 2
    r = 8
    enable_lora = True  # TODO test for different subsets too

    out_features = (n_head + 2 * n_query_groups) * head_size

    config = LoRAConfig(
        n_embd=out_features,
        n_head=n_head,
        n_query_groups=n_query_groups,
        head_size=head_size,
    )
    config.fix_head_size = True

    qkv = LoRAQKVLinear(
        config,
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

    qkv_weights = []
    for i, shape in zip([1, 1000, 10000], [q_shape, k_shape, v_shape]):
        # so that we know a) what's the original position in the pre-shuffle matrix
        # b) which one is from q, k or v
        qkv_weights.append(
            (i + torch.arange(shape[0] * shape[1] * shape[2])).reshape(shape)
        )

    result = qkv.zero_pad(qkv_weights)

    # now we check back the order
    qkv_weights = [
        qkv_weights[0].flatten(),
        qkv_weights[1].flatten(),
        qkv_weights[2].flatten(),
    ]
    qkv_group_size = qkv.sub_network_q_per_kv + 2
    # need the indices to check if correct values are at correct places
    curr_q, curr_k, curr_v = 0, 0, 0
    for batch_el in result:
        for seq_el in batch_el:
            for q_id, q_el in enumerate(seq_el):
                if (q_id // head_size) % qkv_group_size < qkv_group_size - 2:
                    assert q_el < 1000  # elements from the q weight
                    assert q_el == qkv_weights[0][curr_q]
                    curr_q += 1
                elif (q_id // head_size) % qkv_group_size == qkv_group_size - 2:
                    print(
                        q_id,
                        q_el,
                        (q_id // head_size) % qkv_group_size == qkv_group_size - 2,
                    )
                    assert 1000 <= q_el < 10000
                    assert q_el == qkv_weights[1][curr_k]
                    curr_k += 1
                else:
                    assert q_el >= 10000
                    assert q_el == qkv_weights[2][curr_v]
                    curr_v += 1

    # one last check to see what happens after splitting it in causal self attention
    result = result.view(
        batch_size,
        seq_length,
        n_query_groups,
        qkv_group_size,
        head_size,
    )

    result = result.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

    # split batched computation into three
    q, k, v = result.split((qkv.sub_network_q_per_kv, 1, 1), dim=2)

    def check_qkv(post_split, weights):
        # permute back and flatten to get the original order
        for i, q_el in enumerate(post_split.permute(0, 3, 1, 2, 4).flatten()):
            assert q_el == weights[i]

    check_qkv(q, qkv_weights[0])
    check_qkv(k, qkv_weights[1])
    check_qkv(v, qkv_weights[2])


@pytest.mark.parametrize("n_query_groups", [1, 4, 16])
def test_zero_pad_sub_network(n_query_groups):
    # TODO config vs in_features repeated in __init__?)
    seq_length = 4
    batch_size = 2

    in_features = 128
    n_head = 16
    head_size = 2
    r = 8
    enable_lora = True  # TODO test for different subsets too

    out_features = (n_head + 2 * n_query_groups) * head_size

    config = LoRAConfig(
        n_embd=out_features,
        n_head=n_head,
        n_query_groups=n_query_groups,
        head_size=head_size,
    )
    config.fix_head_size = True

    qkv = LoRAQKVLinear(
        config,
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

    # writing out all cases for clarity
    # MQA
    if n_query_groups == 1:
        # super-network has 16 heads and 1 group
        sub_n_query_groups = 1
        sub_n_head = 8  # 8/16 heads, only 1 group total
    # MHA
    elif n_query_groups == n_head:
        # super-network has 16 heads in 16 groups
        sub_n_query_groups = 8
        sub_n_head = 8  # 8/16 heads, 8/16 groups (1 group per head)
    # GQA
    else:
        # super-network has 16 heads in 4 groups (4 heads per group)
        sub_n_query_groups = 2
        sub_n_head = 8  # 8/16 heads, 2/4 active groups (resulting into 2 heads per group)

    attn = CausalSelfAttention(config, 0)  # to avoid copying get_qkv_indices
    attn.set_sub_network(config.n_embd, sub_n_head, sub_n_query_groups, head_size)
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

    qkv_weights = []
    for i, shape in zip([1, 1000, 10000], [q_shape, k_shape, v_shape]):
        # so that we know a) what's the original position in the pre-shuffle matrix
        # b) which one is from q, k or v
        qkv_weights.append(
            (i + torch.arange(shape[0] * shape[1] * shape[2])).reshape(shape)
        )

    result = qkv.zero_pad(qkv_weights)

    # now we check back the order
    qkv_weights = [
        qkv_weights[0].flatten(),
        qkv_weights[1].flatten(),
        qkv_weights[2].flatten(),
    ]
    qkv_group_size = qkv.sub_network_q_per_kv + 2

    # need the indices to check if correct values are at correct places
    curr_q, curr_k, curr_v = 0, 0, 0
    for batch_el in result:
        for seq_el in batch_el:
            for q_id, q_el in enumerate(seq_el):
                if (q_id // head_size) % qkv_group_size < qkv_group_size - 2:
                    assert q_el < 1000  # elements from the q weight
                    assert q_el == qkv_weights[0][curr_q]
                    curr_q += 1
                elif (q_id // head_size) % qkv_group_size == qkv_group_size - 2:
                    print(
                        q_id,
                        q_el,
                        (q_id // head_size) % qkv_group_size == qkv_group_size - 2,
                    )
                    assert 1000 <= q_el < 10000
                    assert q_el == qkv_weights[1][curr_k]
                    curr_k += 1
                else:
                    assert q_el >= 10000
                    assert q_el == qkv_weights[2][curr_v]
                    curr_v += 1

    # one last check to see what happens after splitting it in causal self attention
    result = result.view(
        batch_size,
        seq_length,
        sub_n_query_groups,
        qkv_group_size,
        head_size,
    )

    result = result.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

    # split batched computation into three
    q, k, v = result.split((qkv.sub_network_q_per_kv, 1, 1), dim=2)

    def check_qkv(post_split, weights):
        # permute back and flatten to get the original order
        for i, q_el in enumerate(post_split.permute(0, 3, 1, 2, 4).flatten()):
            assert q_el == weights[i]

    check_qkv(q, qkv_weights[0])
    check_qkv(k, qkv_weights[1])
    check_qkv(v, qkv_weights[2])
