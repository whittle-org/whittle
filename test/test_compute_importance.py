from __future__ import annotations

from unittest import mock

from litgpt.config import Config

from whittle import compute_importance
from whittle.models.gpt import GPT
from whittle.search.search_spaces import HWGPTBench


@mock.patch("whittle.compute_importance.evaluate_wikitext", return_value=5.0)
def test_evaluate_configs(evaluate_mock):
    config = Config(block_size=4, n_layer=2, n_embd=8, n_head=4, intermediate_size=32)
    model = GPT(config)
    search_space = HWGPTBench(config)
    configs = [
        {"embed_dim": 4, "num_heads": 2, "mlp_ratio": 2, "depth": 1},
        {"embed_dim": 8, "num_heads": 4, "mlp_ratio": 4, "depth": 2},
    ]

    with mock.patch.object(model, "set_sub_network", wraps=model.set_sub_network) as spy:
        ppls, params = compute_importance.evaluate_configs(
            model, configs, search_space, 4, "tokenizer", 2, 3
        )
        assert "sampled_layer_indices" not in spy.call_args_list[0].kwargs

        compute_importance.evaluate_configs(
            model, configs, search_space, 4, "tokenizer", 2, 3, layer_order=[1, 0]
        )
        assert spy.call_args_list[2].kwargs["sampled_layer_indices"] == [1]
        assert spy.call_args_list[3].kwargs["sampled_layer_indices"] == [0, 1]

    assert ppls == [5.0, 5.0]
    assert params[0] < params[1]
    evaluate_mock.assert_called_with(4, model, "tokenizer", 2, 3)
    # the caller's configs stay unchanged
    assert "sampled_layer_indices" not in configs[0]
