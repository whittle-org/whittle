from __future__ import annotations

import os
os.environ["PYTORCH_NVML_DISABLE"] = "1"

import tempfile
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytest

from whittle.distillation.knowledge_distill import KD
from whittle.distillation.utils import create_student_training_data

class DummyTeacher(nn.Module):
    def __init__(self, vocab_size=10, seq_len=8):
        super().__init__()
        self.linear = nn.Linear(seq_len, vocab_size)
        self.config = type("Config", (), {"n_head": 2, "head_size": 4, "vocab_size": vocab_size})
        self.sub_network_num_heads = 2

    def forward(self, x):
        x = x.float()
        out = self.linear(x)
        # Expand dimension to mimic logits output per token
        return out.unsqueeze(1).expand(-1, x.shape[1], -1)

class DummyStudent(nn.Module):
    def __init__(self, vocab_size=10, seq_len=8):
        super().__init__()
        self.linear = nn.Linear(seq_len, vocab_size)
        self.sub_network_n_embd = seq_len
        self.sub_network_num_heads = 1
        self.sub_network_intermediate_size = seq_len
        self.config = type("Config", (), {"vocab_size": vocab_size})
    
    def forward(self, x):
        x = x.float()
        out = self.linear(x)
        return out.unsqueeze(1).expand(-1, x.shape[1], -1)

def get_dummy_dataloader(batch_size=2, seq_len=8, num_batches=3):
    data = []
    for _ in range(num_batches):
        batch = {
            "input_ids": torch.randint(0, 10, (batch_size, seq_len)),
            "labels": torch.randint(0, 10, (batch_size, seq_len))
        }
        data.append(batch)
    # Data is already batched so use batch_size=None
    return DataLoader(data, batch_size=None)

def test_student_training_data_and_distillation():
    device = 'cpu'
    seq_len = 8
    batch_size = 2
    epochs = 1

    teacher = DummyTeacher(vocab_size=10, seq_len=seq_len)
    student = DummyStudent(vocab_size=10, seq_len=seq_len)

    dataloader = get_dummy_dataloader(batch_size=batch_size, seq_len=seq_len, num_batches=3)

    # Test student training data creation with new parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        training_data_output = os.path.join(tmpdir, "student_training_data")
        create_student_training_data(
            teacher,
            dataloader,
            device,
            training_data_output,
            top_k=5,
            subset_size=10,
            use_top_k_logits=True,
            chunk_size=2,  # Force saving in chunks for test
            precision='full'
        )
        # Expect at least one chunk file to be created (chunk index 0)
        expected_file = f"{training_data_output}_part0.pt"
        assert os.path.exists(expected_file), "Student training data file was not created."

    # Test knowledge distillation
    kd = KD(
        teacher=teacher,
        student=student,
        seq_len=seq_len,
        method='logits',
        dataloader=dataloader,
        device=device,
        seed=42,
        verbose=False,
        kd_epochs=epochs,
        distillation_weight=0.5,
        temperature=3
    )
    distilled_student = kd.distill()
    
    # Verify the distilled student produces outputs of expected shape.
    sample_input = torch.randint(0, 10, (batch_size, seq_len)).to(device)
    logits = distilled_student(sample_input)
    assert logits.shape == (batch_size, seq_len, 10), "Distilled student output has unexpected shape."