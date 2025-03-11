from __future__ import annotations

import torch
import torch.nn.functional as F


def forward_kl(logits, teacher_logits, temperature):
    # Calculate log-softmax of student and teacher logits
    student_logprobs = F.log_softmax(logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Compute KL divergence using PyTorch's F.kl_div
    distill_loss = F.kl_div(student_logprobs, teacher_probs, reduction="batchmean")

    # Scale by temperature squared
    distill_loss = distill_loss * (temperature**2)
    return distill_loss


def reverse_kl(logits, teacher_logits, temperature):
    # Calculate log-softmax of teacher and student logits
    teacher_logprobs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.softmax(logits / temperature, dim=-1)

    # Compute KL divergence using PyTorch's F.kl_div
    distill_loss = F.kl_div(teacher_logprobs, student_probs, reduction="batchmean")

    # Scale by temperature squared
    distill_loss = distill_loss * (temperature**2)
    return distill_loss


def symmetric_kl(logits, teacher_logits, temperature, lam=0.9):
    # Forward KL Divergence (student || teacher)
    for_kl = forward_kl(logits, teacher_logits, temperature)

    # Reverse KL Divergence (teacher || student)
    rev_kl = reverse_kl(logits, teacher_logits, temperature)

    # Weighted combination of forward and reverse KL
    distill_loss = (1 - lam) * for_kl + lam * rev_kl
    return distill_loss


def js_distance(logits, teacher_logits, temperature):
    # Calculate log-softmax of student and teacher logits
    student_logprobs = F.log_softmax(logits / temperature, dim=-1)
    teacher_logprobs = F.log_softmax(teacher_logits / temperature, dim=-1)

    # Average the probabilities
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.softmax(logits / temperature, dim=-1)
    avg_probs = 0.5 * (teacher_probs + student_probs)

    # Compute JS Divergence using the sum of two KL divergences
    js_div = 0.5 * (
        F.kl_div(student_logprobs, avg_probs, reduction="batchmean")
        + F.kl_div(teacher_logprobs, avg_probs, reduction="batchmean")
    )

    # Scale by temperature squared
    distill_loss = js_div * (temperature**2)
    return distill_loss


def simple_cross_entropy(logits, teacher_logits, temperature):
    # Use temperature-scaled logits for cross-entropy loss
    teacher_logits = teacher_logits / temperature
    logits = logits / temperature

    # Apply CrossEntropyLoss
    ce_loss = F.cross_entropy(logits, teacher_logits)
    return ce_loss


def cosine_similarity(logits, teacher_logits, temperature):
    # Calculate softmax probabilities
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.softmax(logits / temperature, dim=-1)

    # Compute cosine similarity
    cos_sim = F.cosine_similarity(teacher_probs, student_probs, dim=-1)

    # The distill loss can be defined as (1 - cosine similarity)
    distill_loss = 1 - cos_sim.mean()
    return distill_loss


# L1 Loss
def l1_loss(logits, teacher_logits):
    distill_loss = F.l1_loss(logits, teacher_logits, reduction="mean")
    return distill_loss


# L2 Loss (Mean Squared Error)
def l2_loss(logits, teacher_logits):
    distill_loss = F.mse_loss(logits, teacher_logits, reduction="mean")
    return distill_loss


# Maximum Mean Discrepancy (MMD)
def mmd_loss(x, y, kernel="rbf"):
    """
    Maximum Mean Discrepancy (MMD) loss between distributions x and y.
    You can use different kernels: 'rbf', 'linear', etc.

    Args:
    x: Tensor of shape [batch_size, feature_dim]
    y: Tensor of shape [batch_size, feature_dim]
    kernel: The type of kernel to use, default is 'rbf' (Radial Basis Function)

    Returns:
    MMD loss
    """

    def rbf_kernel(x, y, sigma=1.0):
        """Computes the RBF kernel between two tensors."""
        dist = torch.cdist(x, y, p=2)  # Compute pairwise Euclidean distance
        return torch.exp(-(dist**2) / (2 * sigma**2))

    def linear_kernel(x, y):
        """Computes the linear kernel between two tensors."""
        return torch.matmul(x, y.T)

    # Select kernel
    if kernel == "rbf":
        Kxx = rbf_kernel(x, x)
        Kyy = rbf_kernel(y, y)
        Kxy = rbf_kernel(x, y)
    elif kernel == "linear":
        Kxx = linear_kernel(x, x)
        Kyy = linear_kernel(y, y)
        Kxy = linear_kernel(x, y)
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")

    # Compute MMD loss
    mmd_loss_value = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd_loss_value
