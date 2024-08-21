from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class DistillLoss(nn.Module):
    """
    Custom loss function for knowledge distillation

    This loss function combines the standard cross-entropy loss with the KL divergence
    between the soft targets produced by a teacher model and a student model.

    Attributes:
        temperature (float): The temperature used for distillation. Higher temperatures
                             produce softer probability distributions.
        distillation_weight (float): The weight factor that balances the importance of
                                     the distillation loss and the hard target loss.
        kldiv (nn.KLDivLoss): The KL divergence loss function.
    """

    def __init__(self, temperature, distillation_weight):
        """
        Initializes the DistillLoss module.

        Args:
            temperature (float): The temperature for distillation.
            distillation_weight (float): The weight factor for the distillation loss.
        """
        super().__init__()

        self.temperature = temperature
        self.distillation_weight = distillation_weight
        self.kldiv = nn.KLDivLoss(reduction="batchmean")

    def forward(self, outputs, labels, outputs_teacher):
        """
        Compute the distillation loss.

        This method computes the loss as a weighted sum of the soft target loss (KL divergence
        between student and teacher outputs) and the hard target loss (cross-entropy loss
        between student outputs and ground truth labels).

        Args:
            outputs (torch.Tensor): The logits output by the student model. Shape (batch_size, num_classes).
            labels (torch.Tensor): The ground truth labels. Shape (batch_size).
            outputs_teacher (torch.Tensor): The logits output by the teacher model. Shape (batch_size, num_classes).

        Returns:
            torch.Tensor: The combined loss.
        """
        soft_target_loss = 0
        outputs_teacher = outputs_teacher.detach()
        if outputs_teacher is not None and self.distillation_weight > 0:
            soft_target_loss = self.kldiv(
                F.log_softmax(outputs / self.temperature, dim=1),
                F.softmax(outputs_teacher / self.temperature, dim=1),
            ) * (self.temperature**2)

        hard_target_loss = F.cross_entropy(outputs, labels, reduction="mean")

        total_loss = soft_target_loss * self.distillation_weight + hard_target_loss * (
            1 - self.distillation_weight
        )

        return total_loss
