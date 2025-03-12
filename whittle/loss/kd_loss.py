from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from whittle.loss.losses import (
    cosine_similarity,
    forward_kl,
    js_distance,
    l1_loss,
    l2_loss,
    mmd_loss,
    reverse_kl,
    simple_cross_entropy,
    symmetric_kl,
)


class DistillLoss(nn.Module):
    """
    Custom loss function for knowledge distillation

    This loss function combines the standard cross-entropy loss with the KL divergence
    between the soft targets produced by a teacher model and a student model.

    Attributes:
        alpha (float): The weight factor for the hard target loss. Higher values give more
                          importance to the cross entropy loss between student logits and ground truth labels.
        beta (float): The weight factor for the soft target loss. Higher values give more
                            importance to the loss between student and teacher logits.
        temperature (float): The temperature used for distillation. Higher temperatures
                             produce softer probability distributions.
        loss (str): The loss function to use for distillation.
        weight_scheme (str): The weight scheme to use for the distillation loss.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        temperature: float = 1.0,
        loss: str = "forward_kld",
        weight_scheme: str = "other",
    ):
        """
        Initializes the DistillLoss module.

        Args:
            alpha (float): The weight factor for the hard target loss. Default is 1.0.
            beta (float): The weight factor for the soft target loss. Default is 0.0.
            temperature (float): The temperature for distillation. Default is 1.0.
            loss (str): The distillation loss function to use. Default is 'forward_kld'.
            weight_scheme (str): The weight scheme to use. Default is 'other'.
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.distill_loss = loss
        self.weight_scheme = weight_scheme

    def forward(self, logits, labels, teacher_logits) -> nn.Tensor:
        """
        Compute the distillation loss.

        This method computes the loss as a weighted sum of the soft target loss (KL divergence
        between student and teacher logits) and the hard target loss (cross-entropy loss
        between student logits and ground truth labels).

        Args:
            logits (torch.Tensor): The logits output by the student model. Shape (batch_size, num_classes).
            labels (torch.Tensor): The ground truth labels. Shape (batch_size).
            teacher_logits (torch.Tensor): The logits output by the teacher model. Shape (batch_size, num_classes).

        Returns:
            torch.Tensor: The combined loss.
        """
        soft_target_loss = 0
        teacher_logits = teacher_logits.detach()

        hard_target_loss = F.cross_entropy(logits, labels, reduction="mean")

        if self.distill_loss == "forward_kld":
            soft_target_loss = forward_kl(logits, teacher_logits, self.temperature)

        elif self.distill_loss == "reverse_kld":
            soft_target_loss = reverse_kl(logits, teacher_logits, self.temperature)

        elif self.distill_loss == "symmetric_kld":
            soft_target_loss = symmetric_kl(logits, teacher_logits, self.temperature)

        elif self.distill_loss == "js_distance":
            soft_target_loss = js_distance(logits, teacher_logits, self.temperature)

        elif self.distill_loss == "simple_cross_entropy":
            soft_target_loss = simple_cross_entropy(
                logits, teacher_logits, self.temperature
            )

        elif self.distill_loss == "cosine_similarity":
            soft_target_loss = cosine_similarity(logits, teacher_logits, self.temperature)

        elif self.distill_loss == "l1_loss":
            soft_target_loss = l1_loss(logits, teacher_logits)

        elif self.distill_loss == "l2_loss":
            soft_target_loss = l2_loss(logits, teacher_logits)

        elif self.distill_loss == "mmd_loss":
            soft_target_loss = mmd_loss(logits, teacher_logits)

        else:
            raise ValueError(f"Invalid distillation loss: {self.distill_loss}")

        if self.weight_scheme == "default":
            coefficient1 = 1.0
            if soft_target_loss == 0:
                coefficient2 = 1.0
            else:
                coefficient2 = hard_target_loss / soft_target_loss
        else:
            coefficient1 = self.alpha
            coefficient2 = self.beta

        total_loss = coefficient1 * hard_target_loss + coefficient2 * soft_target_loss

        return total_loss
