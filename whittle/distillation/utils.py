from __future__ import annotations

import glob
import yaml
import logging
from tqdm import tqdm
from pathlib import Path
from dataclasses import asdict
from typing import Any, Optional, Callable, Union

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset

from litgpt import Config
from transformers import PreTrainedTokenizer

from whittle.models.gpt import GPT

class TextDataset(Dataset):
    '''
    A dataset class for tokenized text data, wraps huggingface datasets into a torch dataset
    '''
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data["input_ids"]
        self.labels = tokenized_data["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.squeeze(torch.tensor(self.input_ids[idx])).long(),
            "labels": torch.squeeze(torch.tensor(self.labels[idx])).long(),
        }

def load_teacher_model(checkpoint: str, config_file: str, device: str) -> Optional[GPT]:
    """
    Loads a teacher model from a checkpoint and config file.
    If no checkpoint is provided, returns None.
    """
    if checkpoint:
        config = Config.from_file(config_file) if config_file else None
        # load model state and build a GPT instance using the config
        if config is None:
            raise ValueError("A config_file is required when loading a teacher checkpoint.")
        teacher = GPT(config)
        teacher.load_state_dict(torch.load(checkpoint, map_location=device))
        teacher.to(device)
        teacher.eval()
        return teacher
    return None

def save_config_to_file(config: Config, path: Union[str, Path]) -> None:
    with open(path, 'w', encoding='utf-8') as fp:
        yaml.safe_dump(asdict(config), fp)

def load_teacher_predictions(
    path: str, 
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load teacher predictions (Top-K logits) from a .pth file.

    Args:
        path (str): File path to the saved predictions (.pth format).
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of top-K logits and indices as tensors.
    """
    data = torch.load(path, weights_only=True)

    topk_values = data['values']
    topk_indices = data['indices'].to(dtype=torch.long)

    print(f"Loaded logits from {path} (stored as Top-{topk_values.shape[-1]} logits per token)")

    return topk_values, topk_indices

class TeacherLogitsLoader:
    def __init__(self, teacher_logits_path):
        """
        Loads teacher logits dynamically from separate files.
        """
        import glob
        self.chunk_files = sorted(glob.glob(f"{teacher_logits_path}_part*.pt"))
        print(f"Found {len(self.chunk_files)} teacher logits files.")

        # Load metadata to determine dataset size
        self.chunk_sizes = []
        for file in self.chunk_files:
            data = torch.load(file, weights_only=True)
            self.chunk_sizes.append(data["values"].size(0))

        self.total_samples = sum(self.chunk_sizes)
        self.loaded_chunk = None
        self.current_chunk_idx = -1

    def get_logits(self, batch_idx, batch_size):
        """
        Fetches the teacher logits for a specific batch index.

        Args:
            batch_idx (int): Batch index.
            batch_size (int): The batch size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: teacher_values, teacher_indices
        """
        # Find which chunk this batch belongs to
        offset = batch_idx * batch_size
        for i, size in enumerate(self.chunk_sizes):
            if offset < size:
                if self.current_chunk_idx != i:
                    # Load the correct chunk
                    print(f"Loading teacher logits chunk: {self.chunk_files[i]}")
                    self.loaded_chunk = torch.load(self.chunk_files[i], weights_only=True)
                    self.current_chunk_idx = i

                # Slice the correct batch from the chunk
                teacher_values = self.loaded_chunk["values"][offset : offset + batch_size]
                teacher_indices = self.loaded_chunk["indices"][offset : offset + batch_size]

                # Ensure correct shape
                teacher_values = teacher_values.view(batch_size, -1, teacher_values.shape[-1])
                teacher_indices = teacher_indices.view(batch_size, -1, teacher_indices.shape[-1])

                return teacher_values, teacher_indices

            offset -= size

        raise IndexError(f"Index {batch_idx} out of range in teacher logits.")
        
def tokenize_function(examples, tokenizer, seq_len ,verbose=True):
    encodings = tokenizer("\n\n".join(examples["text"]), return_tensors="pt")
    seq_len_orig = encodings.input_ids.size(1)
    input_ids_li = []
    target_ids_li = []

    for begin_loc in range(0, seq_len_orig, seq_len):
        end_loc = min(begin_loc + seq_len, seq_len_orig)
        if end_loc != begin_loc + seq_len:  # ignore last batch
            break
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = torch.zeros_like(input_ids)
        target_ids[:, 0:-1] = input_ids[:, 1:]
        target_ids[:, -1] = -100
        #target_ids[:, -1] = tokenizer.pad_token_id  # Target padding
        input_ids_li.append(torch.squeeze(input_ids))
        target_ids_li.append(torch.squeeze(target_ids))

    if verbose:
        print('information about tokenized dataset:')
        print(f"Tokenized {len(input_ids_li)} batches")
        print(f"First batch size: {input_ids_li[0].size()}")
        print(f"Last batch size: {input_ids_li[-1].size()}")
        print(f'first batch text: {tokenizer.decode(input_ids_li[0].squeeze().tolist(), skip_special_tokens=True)}')
        print(f'last batch text: {tokenizer.decode(input_ids_li[-1].squeeze().tolist(), skip_special_tokens=True)}')

    return {
        "input_ids": input_ids_li,
        "labels": target_ids_li,
    }

def create_tiny_gpt(config: Optional[Union[Config, str]] = None, 
                    verbose: bool = False) -> tuple[GPT, Config]:
    """
    Create a tiny GPT model with a simplified configuration for testing.
    
    You can optionally supply a litgpt Config instance or a path to a YAML configuration file.
    If a config file path is provided, the configuration will be loaded from that file.
    
    Args:
        config (Optional[Union[Config, str]]): Either a litgpt Config instance, a path to a YAML config file,
                                               or None to use default parameters.
        verbose (bool): If True, prints the configuration.
    
    Returns:
        tuple[GPT, Config]: An instance of the GPT model and its configuration.
    """
    if isinstance(config, str):
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f)
        tiny_config = Config(**config_dict)

    elif isinstance(config, Config):
        tiny_config = config
    
    else:
        # Create default tiny configuration if none provided
        tiny_config = Config(
            vocab_size=50688,    # Vocabulary size 
            head_size=32,        # Head size for the attention mechanism
            n_embd=256,          # Embedding size
            n_layer=10,          # Number of layers
            n_head=4,            # Number of attention heads
            bias=True            # Include bias in linear layers
        )
        tiny_config.fix_head_size = True
        tiny_config.model_type = "gpt"
        tiny_config.tie_embeddings = True

    if verbose:
        print(f"Tiny GPT Config: {tiny_config}\n\n")

    tiny_gpt = GPT(tiny_config)
    tiny_gpt.reset_super_network()

    # Initialize model parameters
    for name, param in tiny_gpt.named_parameters():
        if param.dim() > 1:
            init.xavier_uniform_(param)
        else:
            init.zeros_(param)

    return tiny_gpt, tiny_config

def create_student_training_data(
        teacher: Optional[GPT],
        dataloader: DataLoader,
        device: str,
        output_path: str,
        top_k: int = 100,  
        subset_size: int = 0, 
        store_topk: bool = True, 
        chunk_size: int = 2000, 
        precision: str = 'full' # ('full' or 'half') 
    ) -> None:
    """
    Passes original data through the teacher model, collects predictions (top-K logits), 
    and saves them in an efficient compressed format. If store_intermediates is True, 
    intermediate activations are also stored for each batch.

    Args:
        teacher (Optional[GPT]): Pre-trained teacher model; if None, a tinyGPT model is created.
        dataloader (DataLoader): DataLoader containing the original training data.
        device (str): Device to run the model.
        output_path (str): File path to save teacher predictions.
        top_k (int): Number of top-K logits to store per token instead of full vocabulary.
        subset_size (int): Total number of batches to store.
        store_topk (bool): Whether to store only top-K logits.
        chunk_size (int): Number of batches to store per chunk.
        precision (str): Precision to store logits ('full' or 'half').
    """
    if teacher is None:
        teacher = create_tiny_gpt()

    teacher.eval()
    teacher.to(device)

    stored_values, stored_indices = [], []
    file_counter = 0

    if store_topk:
        print(f"Storing Top-{top_k} logits")
    else:
        print("Storing full logits")

    if subset_size > 0:
        print(f"Storing {subset_size} full batches.")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating teacher predictions")):
            if subset_size > 0 and batch_idx >= subset_size:
                break
            
            input_ids = batch["input_ids"].to(device)
            outputs = teacher(input_ids)

            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Extract logits if tuple

            if store_topk:
                topk_values, topk_indices = torch.topk(outputs, top_k, dim=-1)
            else:
                topk_values = outputs
                topk_indices = torch.arange(outputs.size(-1)).unsqueeze(0).expand_as(outputs)

            if precision == 'half':
                stored_values.append(topk_values.half().cpu()) 
                stored_indices.append(topk_indices.cpu())
            else:
                stored_values.append(topk_values.cpu())
                stored_indices.append(topk_indices.cpu())

            # Save in chunks to avoid memory explosion
            if (batch_idx + 1) % chunk_size == 0:
                chunk_output_path = f"{output_path}_part{file_counter}.pt"
                torch.save({'values': torch.cat(stored_values), 'indices': torch.cat(stored_indices)}, chunk_output_path)
                print(f"Saved chunk {file_counter} to {chunk_output_path}")
                stored_values, stored_indices = [], []  # Free memory
                file_counter += 1

    # Save remaining data
    if stored_values:
        chunk_output_path = f"{output_path}_part{file_counter}.pt"
        torch.save({'values': torch.cat(stored_values), 'indices': torch.cat(stored_indices)}, chunk_output_path)
        print(f"Saved final chunk {file_counter} to {chunk_output_path}")

    print("Finished saving all teacher predictions.")

def create_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    seq_len: int,
    batch_size: int,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    seed: int = 42,
    tokenize_function: Callable = tokenize_function
) -> DataLoader:
    '''
    Create a dataloader from a dataset, after tokenizing it
    '''
    # Ensure pad_token is set for tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.eos_token_id

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, seq_len, verbose),
        batched=True,
        remove_columns=["text"],
    )

    # Initialize Dataset and DataLoader
    dataset = TextDataset(tokenized_dataset)
    print(f"Dataset size: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    return dataloader