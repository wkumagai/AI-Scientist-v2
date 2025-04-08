
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from huggingface_hub import login

if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])

scan_dataset = load_dataset("scan", "simple", split="train", trust_remote_code=True)
scan_test = load_dataset("scan", "simple", split="test", trust_remote_code=True)

scan_example = scan_dataset[0]
print(f"SCAN input: {scan_example['commands']}, output: {scan_example['actions']}")

def compositional_regularization(hidden_states, factor=0.1):
    """
    Apply compositional regularization to hidden states.
    This encourages the model to learn compositional representations.
    
    Args:
        hidden_states: Tensor of shape [batch_size, seq_len, hidden_dim]
        factor: Regularization strength
        
    Returns:
        Regularization loss term
    """
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(1)
    
    if hidden_states.size(1) > 1:
        return factor * torch.norm(hidden_states[:, 1:] - hidden_states[:, :-1])
    else:
        return torch.tensor(0.0, device=hidden_states.device)
