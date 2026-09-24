from __future__ import annotations

import os
from unittest import mock
from unittest.mock import Mock

import torch
from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import Config
from litgpt.utils import save_config
from torch.utils.data import DataLoader

from whittle import distill
from whittle.args import DistillArgs
from whittle.models.gpt import GPT


@mock.patch.dict(os.environ, {"DISABLE_TORCH_COMPILE": "1"})
# See `test_pretrain` in test_pretrain_super_network.py for why we mock this
@mock.patch("whittle.distill.save_hyperparameters")
def test_distill(save_hyperparameters_mock, tmp_path):
    teacher_config = Config(
        block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8
    )
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    save_config(teacher_config, teacher_dir)
    torch.save(GPT(teacher_config).state_dict(), teacher_dir / "lit_model.pth")

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)

    student_config = {
        "sub_network_n_embd": 4,
        "sub_network_intermediate_size": 8,
        "sub_network_num_heads": 2,
        "sub_network_n_layers": 1,
    }

    out_dir = tmp_path / "out"
    with (
        mock.patch(
            "whittle.distill.get_dataloaders", Mock(return_value=(dataloader, dataloader))
        ),
        # a single-element side_effect stops the test if the student sampling retries
        mock.patch(
            "whittle.sampling.random_sampler.RandomSampler.sample",
            side_effect=[student_config],
        ),
    ):
        distill.setup(
            out_dir=out_dir,
            precision="32-true",
            teacher_checkpoint_dir=teacher_dir,
            train=TrainArgs(
                global_batch_size=2,
                micro_batch_size=1,
                max_tokens=8,
                save_interval=1,
                max_norm=1.0,
                lr_warmup_steps=1,
            ),
            distill=DistillArgs(),
            eval=EvalArgs(interval=1, max_iters=1, initial_validation=False),
            optimizer="RMSprop",
            devices=1,
            min_ratio=0.01,
            max_ratio=0.99,
        )

    for checkpoint_dir in ["step-00000001", "step-00000002", "distill"]:
        assert (out_dir / checkpoint_dir / "lit_model.pth").is_file()
        student = Config.from_file(out_dir / checkpoint_dir / "model_config.yaml")
        assert (student.n_embd, student.n_layer, student.n_head) == (4, 1, 2)

    save_hyperparameters_mock.assert_called()
