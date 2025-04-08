import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SCAN dataset
scan_train = load_dataset("scan", "simple", split="train", trust_remote_code=True)
scan_test = load_dataset("scan", "simple", split="test", trust_remote_code=True)


# Tokenizer for commands and actions
class Tokenizer:
    def __init__(self, vocab):
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, sequence):
        return [self.word2idx[word] for word in sequence.split()]

    def decode(self, indices):
        return " ".join([self.idx2word[idx] for idx in indices])


command_vocab = set(
    word for example in scan_train for word in example["commands"].split()
)
action_vocab = set(
    word for example in scan_train for word in example["actions"].split()
)

command_tokenizer = Tokenizer(command_vocab)
action_tokenizer = Tokenizer(action_vocab)


# Dataset preparation
class SCANDataset(torch.utils.data.Dataset):
    def __init__(self, data, command_tokenizer, action_tokenizer):
        self.data = data
        self.command_tokenizer = command_tokenizer
        self.action_tokenizer = action_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        command = self.command_tokenizer.encode(self.data[idx]["commands"])
        action = self.action_tokenizer.encode(self.data[idx]["actions"])
        return torch.tensor(command), torch.tensor(action)


train_dataset = SCANDataset(scan_train, command_tokenizer, action_tokenizer)
test_dataset = SCANDataset(scan_test, command_tokenizer, action_tokenizer)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: x
)


# Simple Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers=1):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.Embedding(input_dim, hidden_dim)
        self.decoder = nn.Embedding(output_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        # Encode
        embedded = self.encoder(src)
        lstm_out, _ = self.lstm(embedded)

        # Decode (training with teacher forcing)
        if trg is not None:
            trg_embedded = self.decoder(trg)
            output, _ = self.lstm(trg_embedded)
            output = self.fc(output)
        else:
            output = self.fc(lstm_out)
        return output


# Loss function with compositional regularization
def compositional_regularization(hidden_states, factor=0.1):
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(1)
    if hidden_states.size(1) > 1:
        return factor * torch.norm(hidden_states[:, 1:] - hidden_states[:, :-1])
    else:
        return torch.tensor(0.0, device=hidden_states.device)


# Training setup
model = Seq2Seq(len(command_vocab), len(action_vocab), hidden_dim=128).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
experiment_data = {
    "metrics": {"train": [], "val": []},
    "losses": {"train": [], "val": []},
}


# Training loop
def train_model(model, train_loader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            src, trg = zip(*batch)
            src, trg = torch.cat([x.unsqueeze(0) for x in src]).to(device), torch.cat(
                [y.unsqueeze(0) for y in trg]
            ).to(device)

            optimizer.zero_grad()
            outputs = model(src, trg)
            trg = trg.view(-1)
            outputs = outputs.view(-1, outputs.size(-1))

            loss = criterion(outputs, trg)
            hidden_states = outputs if outputs.dim() == 3 else outputs.unsqueeze(1)
            loss += compositional_regularization(hidden_states)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {epoch_loss / len(train_loader):.4f}")


train_model(model, train_loader, optimizer, criterion)


# Evaluation of Compositional Generalization Accuracy
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            src, trg = zip(*batch)
            src, trg = torch.cat([x.unsqueeze(0) for x in src]).to(device), torch.cat(
                [y.unsqueeze(0) for y in trg]
            ).to(device)

            outputs = model(src)
            predictions = outputs.argmax(dim=2)

            for pred, truth in zip(predictions, trg):
                total += 1
                if torch.equal(pred, truth):
                    correct += 1
    return 100.0 * correct / total


accuracy = evaluate_model(model, test_loader)
print(f"Compositional Generalization Accuracy: {accuracy:.2f}%")

# Save experiment results
experiment_data["metrics"]["val"].append(accuracy)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
