# ===================================================================
#
#   Long-Range Arena: The Adding Problem
#
# This script serves as a benchmark for long-range dependency capabilities,
# comparing the definitive, hybrid Zarvan architecture against a standard
# Transformer baseline on the "Adding Problem".
#
# The task requires the model to find two marked numbers in a long
# sequence of noise, remember them, and output their sum, testing the
# model's ability to selectively retrieve and combine sparse information
# over long distances.
#
# Author: Yasser Sajjadi, in collaboration with Google's Gemini
#
# ===================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import time
from tqdm.auto import tqdm

# --- 1. Configuration ---
FLAGS = {
    "seq_len": 200,
    "embed_dim": 128,
    "hidden_dim": 256,
    "ff_dim": 256, # Feed-forward dim for Transformer
    "batch_size": 128,
    "num_epochs": 5,
    "learning_rate": 2e-4,
    "num_heads": 4,
    "num_layers": 4,
    "max_value": 5, # Input values will be between 0 and 4
}

# ============================================================================
# Part 2: Definitive Zarvan Architecture
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class HolisticContextExtractor(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.s_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        s, v = self.s_proj(x), self.v_proj(x)
        s = s.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        weights = F.softmax(s, dim=-1)
        head_outputs = (weights * v).sum(dim=2)
        concatenated_heads = head_outputs.reshape(B, self.embed_dim)
        return self.combine(concatenated_heads)

class AssociativeContextExtractor(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.importance_scorer = nn.Sequential(nn.Linear(embed_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.importance_scorer(x)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

class ZarvanBlock(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.holistic_ctx = HolisticContextExtractor(embed_dim, num_heads)
        self.associative_ctx = AssociativeContextExtractor(embed_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim * 2)
        )
        self.update_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        q_holistic = self.holistic_ctx(x)
        q_associative = self.associative_ctx(x)
        q_holistic_exp = q_holistic.unsqueeze(1).expand(-1, S, -1)
        q_associative_exp = q_associative.unsqueeze(1).expand(-1, S, -1)
        gate_input = torch.cat([x, q_holistic_exp, q_associative_exp], dim=-1)
        input_gate, forget_gate = self.gate_net(gate_input).chunk(2, dim=-1)
        gated_x = torch.sigmoid(input_gate) * x + torch.sigmoid(forget_gate) * self.update_proj(x)
        output = self.ffn(gated_x)
        return self.norm(x + output)

# --- 3. Models for the Adding Problem ---
class ZarvanForAdding(nn.Module):
    def __init__(self, vocab_size, num_classes, seq_len, embed_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        self.layers = nn.ModuleList([
            ZarvanBlock(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        # Use the last token's vector for the final prediction
        z = x[:, -1, :]
        return self.fc(z)

class TransformerForAdding(nn.Module):
    def __init__(self, vocab_size, num_classes, seq_len, embed_dim, ff_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        z = x[:, -1, :]
        return self.fc(z)

# --- 4. Custom Dataset for the Adding Problem ---
class AddingProblemDataset(Dataset):
    def __init__(self, seq_len, num_samples, max_value):
        self.seq_len, self.num_samples, self.max_value = seq_len, num_samples, max_value
        # Vocabulary: 0=NOISE, 1=MARKER, 2=QUERY, 3...=(numbers)
        self.noise_token, self.marker_token, self.query_token, self.first_num_token = 0, 1, 2, 3
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        sequence = torch.full((self.seq_len,), self.noise_token, dtype=torch.long)
        pos1, pos2 = random.sample(range(self.seq_len // 2), 2)
        val1, val2 = random.randint(0, self.max_value - 1), random.randint(0, self.max_value - 1)
        sequence[pos1], sequence[pos1 + 1] = self.marker_token, val1 + self.first_num_token
        sequence[pos2], sequence[pos2 + 1] = self.marker_token, val2 + self.first_num_token
        sequence[-1] = self.query_token
        target = torch.tensor(val1 + val2, dtype=torch.long)
        return sequence, target

# --- 5. Training and Evaluation Loop ---
def run_experiment(model_name, model, train_loader, test_loader, device):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=FLAGS['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    print(f"\n--- Training {model_name} ---")
    start_time = time.time()
    for epoch in range(1, FLAGS['num_epochs'] + 1):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{FLAGS['num_epochs']}", leave=False)
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        model.eval()
        total_correct, total_samples = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                total_correct += (output.argmax(1) == target).sum().item()
                total_samples += target.size(0)
        accuracy = 100. * total_correct / total_samples
        print(f"Epoch {epoch}/{FLAGS['num_epochs']} -> Test Accuracy: {accuracy:.2f}%")
    total_time = time.time() - start_time
    print(f"Finished Training {model_name}. Total Time: {total_time:.2f}s")
    return accuracy

# --- 6. Main Execution Block ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    train_dataset = AddingProblemDataset(FLAGS['seq_len'], 20000, FLAGS['max_value'])
    test_dataset = AddingProblemDataset(FLAGS['seq_len'], 1000, FLAGS['max_value'])
    train_loader = DataLoader(train_dataset, batch_size=FLAGS['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS['batch_size'], num_workers=0)

    vocab_size = 3 + FLAGS['max_value']
    num_classes = 2 * (FLAGS['max_value'] - 1) + 1
    
    zarvan_model = ZarvanForAdding(
        vocab_size=vocab_size, num_classes=num_classes, seq_len=FLAGS['seq_len'],
        embed_dim=FLAGS['embed_dim'], hidden_dim=FLAGS['hidden_dim'],
        num_heads=FLAGS['num_heads'], num_layers=FLAGS['num_layers']
    )
    
    transformer_model = TransformerForAdding(
        vocab_size=vocab_size, num_classes=num_classes, seq_len=FLAGS['seq_len'],
        embed_dim=FLAGS['embed_dim'], ff_dim=FLAGS['ff_dim'],
        num_heads=FLAGS['num_heads'], num_layers=FLAGS['num_layers']
    )
    
    zarvan_acc = run_experiment("Definitive Zarvan", zarvan_model, train_loader, test_loader, device)
    transformer_acc = run_experiment("Transformer", transformer_model, train_loader, test_loader, device)
    
    print("\n--- FINAL RESULTS ---")
    print(f"Definitive Zarvan Final Accuracy: {zarvan_acc:.2f}%")
    print(f"Transformer Final Accuracy: {transformer_acc:.2f}%")

    if zarvan_acc > 95:
        print("\n✅ The definitive Zarvan architecture has successfully solved the long-range dependency task!")
    else:
        print("\n❌ The definitive Zarvan architecture struggled. The design is sound but may need further tuning.")
