import os
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, AutoTokenizer
from jsonargparse import CLI
from pathlib import Path
import matplotlib.pyplot as plt

from litgpt import Config

from whittle.models.gpt.model import GPT
from whittle.args import DistillArgs
from whittle.distillation.utils import (
    create_dataloader,
    TeacherLogitsLoader
)
from whittle.distillation.knowledge_distill import KD
from whittle.sampling.param_bins import ParamBins, ParamsEstimator
from whittle.metrics.parameters import compute_parameters
from whittle.pretrain_super_network import get_search_space

def evaluate(model: torch.nn.Module, dataloader, device: str) -> float:
    model.to(device)
    model.eval()
    total_loss = 0.0
    num_batches = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    print(f"Average loss: {avg_loss:.4f}")
    return avg_loss

def main(
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu',
    seed: int = 42,
    seq_len: int = 64,
    batch_size: int = 5,
    verbose: bool = False,
    save_dir: str = 'save_dir',
    dataset_path: str = "roneneldan/TinyStories",
    teacher_path: Path = Path('./checkpoints/standard-step-00150000'),

    distill: DistillArgs = DistillArgs(
        method='logits',
        on_cluster=False,
        use_topk_logits=False,
        use_precomputed_logits=False,
        subnetwork=True,
        temperature=0.5,
        alpha=0.5
    )    
):
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = load_dataset(dataset_path, split="train")
    test_dataset  = load_dataset(dataset_path, split="test")
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  

    train_dataloader = create_dataloader(train_dataset, tokenizer, seq_len, batch_size, device, verbose, seed)
    test_dataloader  = create_dataloader(test_dataset, tokenizer, seq_len, batch_size, device, verbose, seed)
    
    if distill.use_precomputed_logits:
        if not distill.precomputed_logits_path:
            raise ValueError("Provide `precomputed_logits_path` when using precomputed logits.")
        print("Using separate teacher logits loader...")
        teacher_logits_loader = TeacherLogitsLoader(distill.precomputed_logits_path)
    else:
        if teacher_path is None:
            raise ValueError("Provide teacher checkpoint path")
        print(f"Loading teacher model from {teacher_path}")
        teacher_logits_loader = None
        teacher_config = Config.from_file(teacher_path / "model_config.yaml")
        teacher_config.fix_head_size = True
        teacher = GPT(teacher_config)
        ckpt = torch.load(teacher_path / "lit_model.pth", map_location=device, weights_only=True)
        teacher.load_state_dict(ckpt, strict=False)
        teacher.reset_super_network()
        print(f"Loaded teacher model from {teacher_path}") 

    if distill.subnetwork:
        from whittle.sampling.random_sampler import RandomSampler

        search_space = get_search_space(teacher_config)
        sampler = RandomSampler(search_space, seed=seed)

        min_config = sampler.get_smallest_sub_network()
        max_config = sampler.get_largest_sub_network()

        params_estimator = ParamsEstimator(teacher)

        param_bins = ParamBins(min_config, max_config, params_estimator, num_bins=3)

        subnetworks = []

        while len(subnetworks) < 10:
            random_config = sampler.sample()
            if param_bins.put_in_bin(random_config):
                subnetworks.append(random_config)
                print(f"Subnetwork accepted: {random_config}")

        print("Selected Subnetworks:")
        print(subnetworks)

        val_losses = []
        param_counts = []
        
        print(f"Training {len(subnetworks)} subnetworks...")
        for i, subnetwork_config in enumerate(subnetworks):
            print("\n" + "="*50)
            print(f"\nTraining subnetwork {i+1} with config: {subnetwork_config}")
            print("="*50 + "\n")
            student = GPT(teacher_config)
            subnetwork = {
                    "sub_network_n_embd": subnetwork_config["embed_dim"],
                    "sub_network_intermediate_size": int(subnetwork_config["mlp_ratio"] * subnetwork_config["embed_dim"]),
                    "sub_network_num_heads": subnetwork_config["num_heads"],
                    "sub_network_n_layers": subnetwork_config["depth"]
                    }   

            student.set_sub_network(**subnetwork)
            param_count = compute_parameters(student)
            print(f"Student model {i+1} set to sub-network with parameter count: {param_count}") 

            print(f"\nEvaluating student model {i+1} BEFORE distillation:")
            _ = evaluate(student, test_dataloader, device)

            kd = KD(
                teacher=teacher,
                student=student,
                seq_len=seq_len,
                method=distill.method,
                dataloader=train_dataloader,
                device=device,
                seed=seed,
                verbose=verbose,
                kd_epochs=distill.kd_epochs,
                teacher_logits_loader=teacher_logits_loader,
                distillation_weight=distill.alpha,
                temperature=distill.temperature
            )
            
            student = kd.distill()

            print(f"\nEvaluating student model {i+1} AFTER distillation:")
            eval_loss = evaluate(student, test_dataloader, device)
            
            val_losses.append(eval_loss)
            param_counts.append(param_count)

            student_ckpt = os.path.join(save_dir, f"distilled_student_model_{i+1}_checkpoint.pth")
            torch.save(student.state_dict(), student_ckpt)
            print(f"Student model {i+1} checkpoint saved to {student_ckpt}")

        plt.figure()
        plt.scatter(param_counts, val_losses)
        plt.xlabel("Parameter Count")
        plt.ylabel("Validation Loss")
        plt.title("TinyStories - GPT2 standard-step-00150000")
        plot_path = os.path.join(save_dir, "TinyStories - GPT2 standard-step-00150000.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Scatter plot saved to {plot_path}")

    else:
        print("Please use distill_student.py to distill a separate student model.")

if __name__ == '__main__':
    CLI(main)