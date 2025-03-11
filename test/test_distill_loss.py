from __future__ import annotations

import torch

from whittle.loss.kd_loss import DistillLoss


def test_distill_loss():
    student_logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    teacher_logits = torch.tensor([[10.0, 0.0]])
    labels = torch.tensor([0])

    temperature = 1.0
    alpha = 0.5
    beta = 0.5
    loss_type = "forward_kld"
    weight_scheme = "other"

    loss_fn = DistillLoss(
        alpha=alpha,
        beta=beta,
        temperature=temperature,
        loss=loss_type,
        weight_scheme=weight_scheme,
    )
    loss = loss_fn(student_logits, labels, teacher_logits)

    expected_loss = 0.6931
    assert torch.isclose(loss, torch.tensor(expected_loss, dtype=loss.dtype), atol=1e-3)

    loss.backward()
    assert student_logits.grad is not None
