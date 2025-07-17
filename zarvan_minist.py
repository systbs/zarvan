import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
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

class ZarvanBlock(nn.Module):
    def __init__(self, seq_len, embed_dim, hidden_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadLinearAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, 1, embed_dim))

    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        x = x + self.pos_encoding[:seq_len]
        residual = x
        attn_out = self.attention(x)
        x = self.norm1(residual + attn_out.unsqueeze(0))
        residual = x
        ffn_out = self.ffn(x)
        z = self.norm2(residual + ffn_out)
        return z

class ZarvanClassifier(nn.Module):
    def __init__(self, seq_len, embed_dim, hidden_dim, num_classes, num_heads):
        super().__init__()
        self.embedding = nn.Linear(1, embed_dim)
        self.zarvan_block = ZarvanBlock(seq_len, embed_dim, hidden_dim, num_heads)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1, 1)
        x = self.embedding(x).permute(1, 0, 2)
        z = self.zarvan_block(x).mean(dim=0)
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

# --- 3. Data & Training Loop --- #
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

# --- 4. Main Execution --- #
# Initialize models
zarvan_model = ZarvanClassifier(**CONFIG, num_classes=10).to(CONFIG['device'])
transformer_model = TransformerClassifier(**CONFIG, num_classes=10).to(CONFIG['device'])

# Optimizers and Schedulers
zarvan_optimizer = optim.AdamW(zarvan_model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
transformer_optimizer = optim.AdamW(transformer_model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
zarvan_scheduler = optim.lr_scheduler.CosineAnnealingLR(zarvan_optimizer, T_max=CONFIG["num_epochs"])
transformer_scheduler = optim.lr_scheduler.CosineAnnealingLR(transformer_optimizer, T_max=CONFIG["num_epochs"])

criterion = nn.CrossEntropyLoss()
history = {
    'zarvan_train_loss': [], 'zarvan_train_acc': [], 'zarvan_test_loss': [], 'zarvan_test_acc': [],
    'transformer_train_loss': [], 'transformer_train_acc': [], 'transformer_test_loss': [], 'transformer_test_acc': []
}

for epoch in range(1, CONFIG["num_epochs"] + 1):
    print(f"\n--- Epoch {epoch}/{CONFIG['num_epochs']} ---")
    # Zarvan
    z_train_loss, z_train_acc = run_epoch(zarvan_model, train_loader, zarvan_optimizer, criterion, CONFIG['device'], True)
    z_test_loss, z_test_acc = run_epoch(zarvan_model, test_loader, None, criterion, CONFIG['device'], False)
    zarvan_scheduler.step()
    history['zarvan_train_loss'].append(z_train_loss); history['zarvan_train_acc'].append(z_train_acc)
    history['zarvan_test_loss'].append(z_test_loss); history['zarvan_test_acc'].append(z_test_acc)
    
    # Transformer
    t_train_loss, t_train_acc = run_epoch(transformer_model, train_loader, transformer_optimizer, criterion, CONFIG['device'], True)
    t_test_loss, t_test_acc = run_epoch(transformer_model, test_loader, None, criterion, CONFIG['device'], False)
    transformer_scheduler.step()
    history['transformer_train_loss'].append(t_train_loss); history['transformer_train_acc'].append(t_train_acc)
    history['transformer_test_loss'].append(t_test_loss); history['transformer_test_acc'].append(t_test_acc)

    print(f"Zarvan      -> Train Acc: {z_train_acc:.2f}%, Test Acc: {z_test_acc:.2f}% | Train Loss: {z_train_loss:.4f}, Test Loss: {z_test_loss:.4f}")
    print(f"Transformer -> Train Acc: {t_train_acc:.2f}%, Test Acc: {t_test_acc:.2f}% | Train Loss: {t_train_loss:.4f}, Test Loss: {t_test_loss:.4f}")

# --- 5. Plotting --- #
epochs_range = range(1, CONFIG['num_epochs'] + 1)
plt.figure(figsize=(16, 7))
plt.suptitle('Zarvan vs. Transformer on MNIST', fontsize=16, weight='bold')

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['zarvan_train_acc'], 'b-o', label='Zarvan Train Acc')
plt.plot(epochs_range, history['zarvan_test_acc'], 'b--o', markersize=8, label='Zarvan Test Acc')
plt.plot(epochs_range, history['transformer_train_acc'], 'r-s', label='Transformer Train Acc')
plt.plot(epochs_range, history['transformer_test_acc'], 'r--s', markersize=8, label='Transformer Test Acc')
plt.title('Accuracy Comparison')
plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)')
plt.legend(); plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['zarvan_train_loss'], 'b-o', label='Zarvan Train Loss')
plt.plot(epochs_range, history['zarvan_test_loss'], 'b--o', markersize=8, label='Zarvan Test Loss')
plt.plot(epochs_range, history['transformer_train_loss'], 'r-s', label='Transformer Train Loss')
plt.plot(epochs_range, history['transformer_test_loss'], 'r--s', markersize=8, label='Transformer Test Loss')
plt.title('Loss Comparison')
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.legend(); plt.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('mnist_comparison_final.png')
print("\nSaved final comparison plot to 'mnist_comparison_final.png'")
plt.show()