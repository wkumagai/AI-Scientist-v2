import os
import torch
from datasets import load_dataset
from huggingface_hub import login

if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])

print("Loading SCAN dataset with 'simple' configuration...")
scan_dataset = load_dataset("scan", "simple", split="train", trust_remote_code=True)
scan_test = load_dataset("scan", "simple", split="test", trust_remote_code=True)

print(f"SCAN train dataset loaded successfully with {len(scan_dataset)} examples")
print(f"SCAN test dataset loaded successfully with {len(scan_test)} examples")

print("\nExample from train dataset:")
print(f"Input: {scan_dataset[0]['commands']}")
print(f"Output: {scan_dataset[0]['actions']}")

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

print("\nTesting compositional regularization function:")
batch_size, seq_len, hidden_dim = 2, 3, 4
hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
print(f"Hidden states shape: {hidden_states.shape}")

reg_loss = compositional_regularization(hidden_states)
print(f"Regularization loss: {reg_loss.item()}")
