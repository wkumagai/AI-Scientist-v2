import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from huggingface_hub import login
import numpy as np

if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading SCAN dataset with 'simple' configuration...")
scan_dataset = load_dataset("scan", "simple", split="train", trust_remote_code=True)
scan_test = load_dataset("scan", "simple", split="test", trust_remote_code=True)

print(f"SCAN train dataset loaded successfully with {len(scan_dataset)} examples")
print(f"SCAN test dataset loaded successfully with {len(scan_test)} examples")

def create_vocab(sentences):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for sentence in sentences:
        for token in sentence.split():
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

cmd_vocab = create_vocab([ex["commands"] for ex in scan_dataset])
act_vocab = create_vocab([ex["actions"] for ex in scan_dataset])

print(f"Command vocabulary size: {len(cmd_vocab)}")
print(f"Action vocabulary size: {len(act_vocab)}")

class SCANDataset(Dataset):
    def __init__(self, dataset, cmd_vocab, act_vocab):
        self.dataset = dataset
        self.cmd_vocab = cmd_vocab
        self.act_vocab = act_vocab
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        cmd_tokens = example["commands"].split()
        cmd_indices = [self.cmd_vocab.get(token, self.cmd_vocab["<unk>"]) for token in cmd_tokens]
        cmd_indices = [self.cmd_vocab["<sos>"]] + cmd_indices + [self.cmd_vocab["<eos>"]]
        
        act_tokens = example["actions"].split()
        act_indices = [self.act_vocab.get(token, self.act_vocab["<unk>"]) for token in act_tokens]
        act_indices = [self.act_vocab["<sos>"]] + act_indices + [self.act_vocab["<eos>"]]
        
        return torch.tensor(cmd_indices), torch.tensor(act_indices)

def collate_fn(batch):
    cmds, acts = zip(*batch)
    
    cmds_padded = nn.utils.rnn.pad_sequence(cmds, batch_first=True, padding_value=0)
    acts_padded = nn.utils.rnn.pad_sequence(acts, batch_first=True, padding_value=0)
    
    return cmds_padded, acts_padded

train_dataset = SCANDataset(scan_dataset, cmd_vocab, act_vocab)
test_dataset = SCANDataset(scan_test, cmd_vocab, act_vocab)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        
        outputs, hidden = self.rnn(embedded)
        
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):
        
        embedded = self.dropout(self.embedding(input))
        
        output, hidden = self.rnn(embedded, hidden)
        
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, output

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, trg_len-1, trg_vocab_size).to(self.device)
        
        decoder_hiddens = torch.zeros(batch_size, trg_len-1, self.decoder.rnn.hidden_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
        
        input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, hidden, decoder_hidden = self.decoder(input, hidden)
            
            outputs[:, t-1] = output
            decoder_hiddens[:, t-1] = decoder_hidden.squeeze(1)
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            top1 = output.argmax(1)
            
            input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs, decoder_hiddens

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
    if hidden_states.size(1) > 1:
        return factor * torch.norm(hidden_states[:, 1:] - hidden_states[:, :-1])
    else:
        return torch.tensor(0.0, device=hidden_states.device)

input_dim = len(cmd_vocab)
output_dim = len(act_vocab)
emb_dim = 128
hid_dim = 256

encoder = Encoder(input_dim, emb_dim, hid_dim).to(device)
decoder = Decoder(output_dim, emb_dim, hid_dim).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

def train(model, train_loader, optimizer, criterion, clip=1.0, comp_reg_factor=0.1):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(train_loader):
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        
        output, decoder_hiddens = model(src, trg)
        
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)  # Skip <sos> token
        
        loss = criterion(output, trg)
        
        comp_reg = compositional_regularization(decoder_hiddens, comp_reg_factor)
        loss = loss + comp_reg
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}, Comp Reg: {comp_reg.item():.4f}")
    
    return epoch_loss / len(train_loader)

def evaluate(model, test_loader, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(test_loader):
            src, trg = src.to(device), trg.to(device)
            
            output, _ = model(src, trg, 0)  # Turn off teacher forcing
            
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)  # Skip <sos> token
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(test_loader)

n_epochs = 3
best_test_loss = float('inf')

print("Starting training...")
for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}/{n_epochs}")
    
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = evaluate(model, test_loader, criterion)
    
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'scan_model.pt')
        print(f"Model saved with test loss: {test_loss:.4f}")

print("Training complete!")

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()
    
    tokens = sentence.split()
    indices = [src_vocab.get(token, src_vocab["<unk>"]) for token in tokens]
    indices = [src_vocab["<sos>"]] + indices + [src_vocab["<eos>"]]
    
    src_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    
    trg_idx = [trg_vocab["<sos>"]]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_idx[-1]]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output, hidden, _ = model.decoder(trg_tensor, hidden)
        
        pred_token = output.argmax(1).item()
        
        if pred_token == trg_vocab["<eos>"]:
            break
        
        trg_idx.append(pred_token)
    
    trg_tokens = [list(trg_vocab.keys())[list(trg_vocab.values()).index(i)] for i in trg_idx[1:]]  # Skip <sos>
    
    return trg_tokens

print("\nTesting model on examples:")
test_sentences = [
    "jump",
    "jump twice",
    "jump thrice",
    "turn left",
    "jump opposite right"
]

rev_cmd_vocab = {v: k for k, v in cmd_vocab.items()}
rev_act_vocab = {v: k for k, v in act_vocab.items()}

for sentence in test_sentences:
    translation = translate_sentence(sentence, cmd_vocab, act_vocab, model, device)
    print(f"Input: {sentence}")
    print(f"Predicted: {' '.join(translation)}")
    
    found = False
    for ex in scan_dataset:
        if ex["commands"] == sentence:
            print(f"Ground truth: {ex['actions']}")
            found = True
            break
    
    if not found:
        print("Ground truth: Not found in dataset")
    
    print()

results = {
    "train_loss": train_loss,
    "test_loss": test_loss,
    "model_params": {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "emb_dim": emb_dim,
        "hid_dim": hid_dim
    }
}

import pickle
with open("scan_experiment_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("Experiment results saved to scan_experiment_results.pkl")
