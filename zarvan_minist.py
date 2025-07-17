import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import math
from tqdm import tqdm

# --- 1. Configuration --- #
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seq_len": 28 * 28,
    "embed_dim": 64,
    "hidden_dim": 64,    # For Zarvan's filter network
    "ff_dim": 128,       # For Transformer's FFN
    "batch_size": 64,
    "num_epochs": 10,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "num_heads": 4,
}
print(f"Using device: {CONFIG['device']}")

# --- 2. Model Architectures --- #

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return x

class ParallelUnifiedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_net = nn.Linear(embed_dim, embed_dim * num_heads)
        self.combine = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        scores_all_heads = self.query_net(x)
        scores_all_heads = scores_all_heads.view(seq_len, batch_size * self.num_heads, self.embed_dim)
        attn_weights = F.softmax(scores_all_heads, dim=0)
        x_repeated = x.repeat_interleave(self.num_heads, dim=1)
        weighted_values = x_repeated * attn_weights
        head_outputs = weighted_values.sum(dim=0)
        concatenated_heads = head_outputs.view(batch_size, self.num_heads * self.embed_dim)
        return self.combine(concatenated_heads)

class FinalZarvanBlock(nn.Module):
    """ The Final Zarvan architecture, using Parallel Unified Attention and Filter, but NO global_ctx. """
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        self.attention = ParallelUnifiedAttention(embed_dim, num_heads)
        self.interactive_context = nn.Linear(embed_dim, embed_dim)
        self.filter_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(embed_dim)
        self.Linear_xw = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        q = self.attention(x)
        i_ctx = F.relu(self.interactive_context(q))
        q_exp = q.unsqueeze(0).expand(x.size(0), -1, -1)
        i_ctx_exp = i_ctx.unsqueeze(0).expand(x.size(0), -1, -1)
        filter_in = torch.cat([x, q_exp, i_ctx_exp], dim=-1)
        f_weights = self.filter_net(filter_in) - self.filter_net(filter_in).mean(dim=0, keepdim=True)
        z = x + self.Linear_xw(x * f_weights)
        return self.norm(z)

class ZarvanClassifier(nn.Module):
    def __init__(self, seq_len, embed_dim, hidden_dim, num_classes, num_heads):
        super().__init__()
        self.embedding = nn.Linear(1, embed_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, seq_len)
        self.zarvan = FinalZarvanBlock(embed_dim, hidden_dim, num_heads)
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1, 1)
        x = self.embedding(x).permute(1, 0, 2)
        x = self.pos_encoder(x)
        z = self.zarvan(x).mean(dim=0)
        return self.fc(z)

class TransformerClassifier(nn.Module):
    def __init__(self, seq_len, embed_dim, ff_dim, num_classes, num_heads):
        super().__init__()
        self.embedding = nn.Linear(1, embed_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1, 1)
        x = self.embedding(x).permute(1, 0, 2)
        x = self.pos_encoder(x)
        return self.fc(self.transformer(x).mean(dim=0))

# --- 3. Data & Training --- #
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=CONFIG["batch_size"], shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=CONFIG["batch_size"], shuffle=False)

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
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            total_correct += (output.argmax(1) == target).sum().item()
            total_samples += target.size(0)
    return total_loss / len(loader), 100. * total_correct / total_samples

# --- 4. Main Execution --- #
models_to_test = {
    "Zarvan": ZarvanClassifier(
        seq_len=CONFIG['seq_len'], embed_dim=CONFIG['embed_dim'], hidden_dim=CONFIG['hidden_dim'],
        num_classes=10, num_heads=CONFIG['num_heads']),
    "Transformer": TransformerClassifier(
        seq_len=CONFIG['seq_len'], embed_dim=CONFIG['embed_dim'], ff_dim=CONFIG['ff_dim'],
        num_classes=10, num_heads=CONFIG['num_heads'])
}
history = {name: {'train_acc': [], 'test_acc': []} for name in models_to_test.keys()}
criterion = nn.CrossEntropyLoss()

for name, model in models_to_test.items():
    print(f"\n🚀 Training {name} on MNIST...")
    model.to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"])
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        _, train_acc = run_epoch(model, train_loader, optimizer, criterion, CONFIG['device'], True)
        _, test_acc = run_epoch(model, test_loader, None, criterion, CONFIG['device'], False)
        scheduler.step()
        history[name]['train_acc'].append(train_acc)
        history[name]['test_acc'].append(test_acc)
        print(f"Epoch {epoch}/{CONFIG['num_epochs']} -> Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

# --- 5. Plotting --- #
plt.figure(figsize=(12, 8))
plt.title('Final Zarvan vs. Transformer on MNIST', fontsize=16)
epochs_range = range(1, CONFIG['num_epochs'] + 1)
for name, hist in history.items():
    plt.plot(epochs_range, hist['train_acc'], marker='o', linestyle='-', label=f'{name} Train Acc')
    plt.plot(epochs_range, hist['test_acc'], marker='s', linestyle='--', label=f'{name} Test Acc')
plt.xlabel('Epochs'); plt.ylabel('Test Accuracy (%)')
plt.legend(); plt.grid(True)
plt.savefig("final_zarvan_vs_transformer_mnist.png")
print("\nSaved final plot to 'final_zarvan_vs_transformer_mnist.png'")
plt.show()