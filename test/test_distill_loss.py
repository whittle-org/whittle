from __future__ import annotations

import torch

from whittle.loss.kd_loss import DistillLoss
from whittle.loss.losses import forward_kl


def test_distill_loss():
    student_logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    teacher_logits = torch.tensor([[10.0, 0.0]])
    labels = torch.tensor([0])

    temperature = 1.0
    alpha = 0.5
    beta = 0.5
    loss_type = "forward_kld"

    loss_fn = DistillLoss(
        alpha=alpha,
        beta=beta,
        temperature=temperature,
        loss=loss_type,
    )
    loss = loss_fn(student_logits, labels, teacher_logits)

    expected_loss = 0.6931
    assert torch.isclose(loss, torch.tensor(expected_loss, dtype=loss.dtype), atol=1e-3)

    loss.backward()
    assert student_logits.grad is not None


def test_default_weight_scheme_coefficient_is_constant():
    student_logits = torch.tensor([[0.0, 1.0], [2.0, 0.0]], requires_grad=True)
    teacher_logits = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])

    loss_fn = DistillLoss(weight_scheme="default")
    loss = loss_fn(student_logits, labels, teacher_logits)
    loss.backward()

    reference = student_logits.detach().clone().requires_grad_(True)
    hard = torch.nn.functional.cross_entropy(reference, labels)
    soft = forward_kl(reference, teacher_logits, 1.0)
    (hard + (hard / soft).detach() * soft).backward()

    assert torch.allclose(loss, hard + hard)
    assert torch.allclose(student_logits.grad, reference.grad)
