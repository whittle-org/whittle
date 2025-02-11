from typing import Any
import torch
from torch import nn
from tqdm import tqdm

from whittle.loss.kd_loss import DistillLoss

class KD(nn.Module):
    def __init__(
        self,
        teacher: Any,
        student: Any,
        seq_len: int,
        method: str,
        dataloader: Any,
        device: str,
        seed: int,
        verbose: bool = False,
        kd_epochs: int = 10,
        teacher_logits_loader: Any = None,  # NEW: Dynamic loader for teacher logits
        **kwargs: Any,
    ) -> None:
        """
        Initializes the KD object for knowledge distillation.

        Args:
            teacher: The teacher model.
            student: The student model.
            seq_len: The sequence length.
            method: Distillation method (only "logits" is supported).
            dataloader: DataLoader for training.
            device: Device identifier.
            seed: Random seed.
            verbose: Whether to print verbose output.
            kd_epochs: Number of distillation epochs.
            teacher_logits_loader: A separate loader for precomputed teacher logits.
            **kwargs: Optional hyperparameters and configurations.
        """
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.method = method
        self.dataloader = dataloader
        self.kwargs = kwargs
        self.device = device
        self.verbose = verbose
        self.seed = seed
        self.seq_len = seq_len
        self.kd_epochs = kd_epochs
        self.teacher_logits_loader = teacher_logits_loader

    def distill(self) -> Any:
        """
        Performs distillation using logits distillation.
        Returns the distilled student model.
        """
        self.temperature = self.kwargs.get("temperature", 3)
        self.distillation_weight = self.kwargs.get("distillation_weight", 0.5)

        if self.method != "logits":
            raise ValueError("Unsupported distillation method. Only 'logits' method is supported.")
        
        return self.logits_distillation()

    def logits_distillation(self) -> Any:
        """
        Performs logits distillation using cross-entropy and KL-divergence losses.
        Returns the distilled student model.
        """
        epochs = self.kd_epochs
        optimizer = torch.optim.Adam(self.student.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=4.5e-7)

        # Ensure teacher model is in evaluation mode
        if self.teacher is not None:
            self.teacher.to(self.device)
            self.teacher.eval()
        self.student.to(self.device)
        self.student.train()

        for epoch in range(epochs):
            loss_avg = 0.0
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Processing batches", unit="batch")):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                batch_size, seq_len = input_ids.shape
                vocab_size = self.kwargs.get("vocab_size", self.student.config.vocab_size)

                # Fetch teacher logits dynamically (if using precomputed logits)
                if self.teacher_logits_loader is not None:
                    teacher_values, teacher_indices = self.teacher_logits_loader.get_logits(batch_idx, batch_size)
                    teacher_values = teacher_values.to(self.device)
                    teacher_indices = teacher_indices.to(self.device)

                    teacher_logits_full = torch.full((batch_size, seq_len, vocab_size), -1e9, device=self.device)

                    print(f"Shape of teacher_values: {teacher_values.shape}")
                    print(f"Shape of teacher_indices: {teacher_indices.shape}")
                    print(f"Shape of input_ids: {input_ids.shape}")
                    print(f"Shape of labels: {labels.shape}")
                    print(f"Shape of student_logits: {self.student(input_ids).shape}")
                    print(f"Shape of teacher_logits: {self.teacher(input_ids).shape}")
                    print(f"Shape of teacher_logits_full: {teacher_logits_full.shape}")
                    
                    # Scatter top-k logits into full logits tensor
                    teacher_logits_full.scatter_(-1, teacher_indices, teacher_values)

                else:
                    # Compute teacher logits dynamically
                    with torch.no_grad():
                        teacher_logits_full = self.teacher(input_ids)

                # Compute student logits
                student_logits = self.student(input_ids)

                # Compute distillation loss
                distill_loss = DistillLoss(
                    temperature=self.temperature,
                    distillation_weight=self.distillation_weight
                )
                loss = distill_loss(student_logits, labels, teacher_logits_full)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_avg += loss.item()

            scheduler.step()
            print(f"Epoch {epoch}, average loss: {loss_avg / len(self.dataloader):.4f}")

        return self.student