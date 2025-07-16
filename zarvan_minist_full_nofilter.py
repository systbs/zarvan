import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm

# --- 1. Configuration --- #
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seq_len": 28 * 28,
    "embed_dim": 64,
    "hidden_dim": 128,
    "batch_size": 64,
    "num_epochs": 10,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "num_heads": 4,
}
print(f"Using device: {CONFIG['device']}")

# --- 2. Model Architectures --- #
class MultiHeadLinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        num_local, num_global = num_heads // 2, num_heads - num_heads // 2
        self.local_proj = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_local)])
        self.local_score = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_local)])
        self.global_proj = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_global)])
        self.global_score = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_global)])
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)

    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        head_outputs = []
        for proj, score_fn in zip(self.local_proj, self.local_score):
            scores = score_fn(x).squeeze(-1)
            attn_weights = F.softmax(scores, dim=0).unsqueeze(-1)
            values = proj(x)
            head_outputs.append((values * attn_weights).sum(dim=0))
        g = x.mean(dim=0)
        for proj, score_fn in zip(self.global_proj, self.global_score):
            scores = score_fn(g)
            attn_weights = F.softmax(scores, dim=0).unsqueeze(0).expand(seq_len, -1, -1)
            values = proj(x)
            head_outputs.append((values * attn_weights).sum(dim=0))
        concatenated = torch.cat(head_outputs, dim=-1)
        return self.fc_out(concatenated)

class Zarvan(nn.Module):
    def __init__(self, seq_len, embed_dim, hidden_dim, num_heads, mode='no_filter'):
        super().__init__()
        self.mode = mode
        self.attention = MultiHeadLinearAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )
        if self.mode in ['full_plus']: # The original 'full' model had a different structure
            self.filter_net = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1), nn.Sigmoid()
            )
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, 1, embed_dim))

    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        x = x + self.pos_encoding[:seq_len]
        residual = x
        attn_out = self.attention(x)
        simple_attn_branch = attn_out.unsqueeze(0)
        if self.mode == 'full_plus':
            filter_input = torch.cat([x.mean(dim=0), attn_out], dim=-1)
            f = self.filter_net(filter_input).unsqueeze(0).expand(seq_len, -1, -1)
            gated_branch = residual * f
            combined_output = simple_attn_branch + gated_branch
        else: # 'no_filter'
            combined_output = simple_attn_branch
        x = self.norm1(residual + combined_output)
        residual = x
        ffn_out = self.ffn(x)
        z = self.norm2(residual + ffn_out)
        return z

class ZarvanClassifier(nn.Module):
    def __init__(self, seq_len, embed_dim, hidden_dim, num_classes, num_heads, mode='no_filter'):
        super().__init__()
        self.embedding = nn.Linear(1, embed_dim)
        self.zarvan = Zarvan(seq_len, embed_dim, hidden_dim, num_heads, mode=mode)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1, 1)
        x = self.embedding(x).permute(1, 0, 2)
        z = self.zarvan(x).mean(dim=0)
        return self.fc(z)

class TransformerClassifier(nn.Module):
    def __init__(self, seq_len, embed_dim, hidden_dim, num_classes, num_heads):
        super().__init__()
        self.embedding = nn.Linear(1, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=0.1, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1, 1)
        x = self.embedding(x).permute(1, 0, 2)
        x = x + self.pos_encoding
        x = self.transformer(x).mean(dim=0)
        return self.fc(x)

# --- 4. Data & Training Loop --- #
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

def run_epoch(model, loader, optimizer, criterion, device, is_training):
    model.train(is_training)
    total_loss, total_correct, total_samples = 0, 0, 0
    desc = "Training" if is_training else "Evaluating"
    with torch.set_grad_enabled(is_training):
        for data, target in tqdm(loader, desc=desc):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_correct += (output.argmax(1) == target).sum().item()
            total_samples += target.size(0)
    return total_loss / len(loader), 100. * total_correct / total_samples

# --- 5. Main Execution --- #
models_to_test = {
    "Zarvan (no-filter)": ZarvanClassifier(mode='no_filter', num_classes=10, **CONFIG),
    "Zarvan (full+no_filter)": ZarvanClassifier(mode='full_plus', num_classes=10, **CONFIG),
    "Transformer": TransformerClassifier(num_classes=10, **CONFIG)
}
results = {name: {'test_acc': []} for name in models_to_test.keys()}

for name, model in models_to_test.items():
    print(f"\n🚀 Training {name} on MNIST...")
    model.to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, CONFIG['device'], is_training=True)
        test_loss, test_acc = run_epoch(model, test_loader, None, criterion, CONFIG['device'], is_training=False)
        scheduler.step()
        results[name]['test_acc'].append(test_acc)
        print(f"Epoch {epoch}/{CONFIG['num_epochs']} -> Test Acc: {test_acc:.2f}%")

# --- 6. Plotting --- #
plt.figure(figsize=(10, 7))
plt.title("Final MNIST Experiment: Model Comparison")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy (%)")
epochs_range = range(1, CONFIG['num_epochs'] + 1)
colors = {'Zarvan (no-filter)': 'blue', 'Zarvan (full+no_filter)': 'green', 'Transformer': 'red'}
markers = {'Zarvan (no-filter)': 'o', 'Zarvan (full+no_filter)': 'X', 'Transformer': 's'}

for name, history in results.items():
    if name in colors:
        plt.plot(epochs_range, history['test_acc'], color=colors[name], marker=markers[name], label=name)

plt.legend()
plt.grid(True)
plt.savefig("final_mnist_comparison.png")
print("\nSaved final MNIST comparison plot to 'final_mnist_comparison.png'")
plt.show()