from __future__ import annotations

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.init as init

from tqdm import tqdm
import logging
import os
import numpy as np

from litgpt import Config

from whittle.models.gpt import GPT
from whittle.models.gpt.blocks import CausalSelfAttention
from whittle.modules.embedding import Embedding
from whittle.modules.linear import Linear
from whittle.modules.layernorm import LayerNorm
from whittle.modules.rmsnorm import RMSNorm
from whittle.models.gpt.blocks import GptNeoxMLP, GemmaMLP, LLaMAMLP

from typing import Any, Optional, Callable, Union
from collections.abc import Generator
from transformers import PreTrainedTokenizer

import yaml
from pathlib import Path
from dataclasses import asdict


import glob


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
            teacher_values (Tensor), teacher_indices (Tensor)
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
        
def merge_saved_logits(output_path: str, merged_output_path: str):
    """
    Merges multiple saved teacher logits files into a single dataset file.

    Args:
        output_path (str): File path pattern of the saved chunks (e.g., 'teacher_logits_part*.pt').
        merged_output_path (str): Final file path to save the merged dataset.
    """
    # Find all matching chunk files
    chunk_files = sorted(glob.glob(f"{output_path}_part*.pt"))  # Sort to maintain order
    print(f"Found {len(chunk_files)} chunk files to merge.")

    all_values, all_indices = [], []

    # Load each chunk and append to lists
    for chunk_file in chunk_files:
        print(f"Loading {chunk_file} ...")
        data = torch.load(chunk_file, weights_only=True)
        all_values.append(data["values"])
        all_indices.append(data["indices"])

    # Concatenate all tensors along the batch dimension
    merged_values = torch.cat(all_values, dim=0)
    merged_indices = torch.cat(all_indices, dim=0)

    # Save the merged file
    torch.save({"values": merged_values, "indices": merged_indices}, merged_output_path)
    print(f"Saved merged dataset to {merged_output_path}")

def collate_fn(batch):
    input_ids, labels, teacher_logits = zip(*batch)
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "teacher_logits": torch.stack(teacher_logits)
    }


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

def create_tiny_gpt(verbose=False) -> GPT:
    """
    Create a tiny GPT model with a simplified configuration for testing.
    
    Returns:
        An instance of the GPT model.
    """
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
        top_k: int = 100,  # Store only top-K logits per token
        subset_size: int = 1024, # Number of batches to store
        use_top_k_logits: bool = True,
        chunk_size: int = 2000, # Number of batches to store per chunk
        precision: str = 'full' # Precision of the stored logits ('full' or 'half') 
    ) -> None:
    """
    Passes original data through the teacher model, collects predictions (top-K logits), 
    and saves them in an efficient compressed format. If store_intermediates is True, 
    intermediate activations are also stored for each batch.

    Args:
        teacher (Optional[GPT]): Pre-trained teacher model; if None, a tinyGPT model is created.
        dataloader (DataLoader): DataLoader containing the original training data.
        device (str): Device to run the model (e.g., 'cuda:0' or 'cpu').
        output_path (str): File path to save teacher predictions.
        store_intermediates (bool): If True, save intermediate activations along with logits.
        top_k (int): Number of top logits to store per token instead of full vocabulary.
        subset_size (int): Number of tokens to store per batch.
    """
    if teacher is None:
        teacher = create_tiny_gpt()

    teacher.eval()
    teacher.to(device)

    stored_values, stored_indices = [], []
    file_counter = 0

    if use_top_k_logits:
        print(f"Storing Top-{top_k} logits")
    else:
        print("Storing full logits")

    # print(f"Storing {subset_size} full batches.")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating teacher predictions")):
            # if batch_idx >= subset_size:
            #     break  # Stop after subset_size batches

            input_ids = batch["input_ids"].to(device)
            outputs = teacher(input_ids)

            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Extract logits if tuple

            if use_top_k_logits:
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

def compute_all_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def params_linear_layer(layer: Linear):
    params = layer.sub_network_in_features * layer.sub_network_out_features
    if layer.use_bias:
        params += layer.sub_network_out_features
    return params


def params_embedding_layer(embedding: Embedding):
    return embedding.num_embeddings * embedding.sub_network_embedding_dim


def params_layer_normalization(normalization_layer: nn.Module):
    if normalization_layer is None:
        return 0
    if isinstance(normalization_layer, LayerNorm):
        return 2 * normalization_layer.sub_network_in_features
    elif isinstance(normalization_layer, RMSNorm):
        return normalization_layer.sub_network_in_features
    else:
        logging.error(
            f"Normalization layer type: {type(normalization_layer)} not supported!"
        )
        raise


def params_attention_layer(attention: CausalSelfAttention):
    dmodel = attention.sub_network_n_embd
    dhead = attention.sub_network_head_size
    num_heads = attention.sub_network_n_head
    num_query_groups = attention.sub_network_query_groups
    qkv_dim = (num_heads + 2 * num_query_groups) * dhead
    n_attention = dmodel * qkv_dim
    if attention.attn.use_bias:
        n_attention += qkv_dim
    n_attention += dmodel * dmodel  # output
    if attention.proj.use_bias:
        n_attention += dmodel

    return n_attention


def params_mlp(mlp: nn.Module):
    layers = []
    if isinstance(mlp, GptNeoxMLP):
        layers = [mlp.proj, mlp.fc]

    elif isinstance(mlp, LLaMAMLP) or isinstance(mlp, GemmaMLP):
        layers = [mlp.proj, mlp.fc_1, mlp.fc_2]

    num_params = 0
    for layer in layers:
        num_params += params_linear_layer(layer)
    return num_params

def register_intermediate_hooks(model: Any) -> None:
    """
    Attaches forward hooks to the embedding layer and each transformer block
    in the model to record their outputs in a dedicated dictionary.
    The keys used are:
      - "block_0": output of the embedding layer,
      - "block_{i+1}": output of the i-th transformer block,
      - "norm2_{i}": output of the second layer-norm in the i-th block.
    """
    model.intermediate_out = {}

    if hasattr(model.transformer, 'wte'):
        def embed_hook(module: nn.Module, input: Any, output: Any) -> None:
            model.intermediate_out['block_0'] = output
        model.transformer.wte.register_forward_hook(embed_hook)

    for i, block in enumerate(model.transformer.h):
        def block_hook(module: nn.Module, input: Any, output: Any, idx: int = i) -> None:
            model.intermediate_out[f'block_{idx+1}'] = output
        block.register_forward_hook(block_hook)

        if hasattr(block, 'ln_2'):
            def norm_hook(module: nn.Module, input: Any, output: Any, idx: int = i) -> None:
                model.intermediate_out[f'norm2_{idx}'] = output
            block.ln_2.register_forward_hook(norm_hook)

def compute_parameters(model: GPT) -> float:
    """
    Computes parameters of the current sub-network of a GPT mmodel. Make sure to set the sub-network before
    calling this function.

    Refs:
        https://towardsdatascience.com/how-to-estimate-the-number-of-parameters-in-transformer-models-ca0f57d8dff0

    Args:
        model: GPT model

    Returns:
        float: number of parameters of the activated sub-network
    """

    num_params = 0
    num_params += params_linear_layer(model.lm_head)
    num_params += params_embedding_layer(model.transformer.wte)
    for i in range(model.sub_network_n_layers):
        block = model.transformer.h[i]
        num_params += params_mlp(block.mlp)
        num_params += params_attention_layer(block.attn)

        num_params += params_layer_normalization(block.norm_1)
        num_params += params_layer_normalization(block.norm_2)
    num_params += params_layer_normalization(model.transformer.ln_f)
    return num_params