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

    def __init__(self, temperature, distillation_weight, loss):
        """
        Initializes the DistillLoss module.
        Args:
            temperature (float): The temperature for distillation.
            distillation_weight (float): The weight factor for the distillation loss.
            loss (str): The distillation loss function to use. Options are 'kld' (KL divergence), 'mse' (mean squared error) or 'l2',
                'mae' (mean absolute error) or 'l1', 'reverse_kld' (reverse KL divergence), 'cosine' (cosine similarity)
                or 'jsd' (Jensen-Shannon divergence)
        """
        super().__init__()

        self.temperature = temperature
        self.distillation_weight = distillation_weight
        self.distill_loss = loss

    def forward(self, outputs, labels, outputs_teacher) -> nn.Tensor:
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
            if self.distill_loss == "kld":
                soft_target_loss = F.kl_div(
                    F.log_softmax(outputs / self.temperature, dim=1),
                    F.softmax(outputs_teacher / self.temperature, dim=1),
                    reduction="batchmean",
                ) * (self.temperature**2)

            elif self.distill_loss == "reverse_kld":
                soft_target_loss = F.kl_div(
                    F.log_softmax(outputs_teacher / self.temperature, dim=1),
                    F.softmax(outputs / self.temperature, dim=1),
                    reduction="batchmean",
                ) * (self.temperature**2)

            elif self.distill_loss == "l1" or self.distill_loss == "mae":
                soft_target_loss = F.l1_loss(outputs, outputs_teacher, reduction="mean")

            elif self.distill_loss == "l2" or self.distill_loss == "mse":
                soft_target_loss = F.mse_loss(outputs, outputs_teacher, reduction="mean")

            elif self.distill_loss == "cosine":
                soft_target_loss = (
                    1 - F.cosine_similarity(outputs, outputs_teacher, dim=1).mean()
                )

            elif self.distill_loss == "jsd":
                m = 0.5 * (outputs + outputs_teacher)
                soft_target_loss = 0.5 * (
                    F.kl_div(
                        F.log_softmax(outputs, dim=1),
                        F.softmax(m, dim=1),
                        reduction="batchmean",
                    )
                    + F.kl_div(
                        F.log_softmax(outputs_teacher, dim=1),
                        F.softmax(m, dim=1),
                        reduction="batchmean",
                    )
                )

            else:
                raise ValueError(f"Invalid distillation loss: {self.distill_loss}")

        hard_target_loss = F.cross_entropy(outputs, labels, reduction="mean")

        total_loss = soft_target_loss * self.distillation_weight + hard_target_loss * (
            1 - self.distillation_weight
        )

        return total_loss
