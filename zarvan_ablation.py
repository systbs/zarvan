import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time

# --- 1. Configuration --- #
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seq_len": 256,
    "embed_dim": 128,
    "hidden_dim": 128,
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "num_heads": 4,
}

DATASETS_CONFIG = {
    "imdb": {"text_col": "text", "label_col": "label", "num_classes": 2},
    "ag_news": {"text_col": "text", "label_col": "label", "num_classes": 4},
    "sst2": {"text_col": "sentence", "label_col": "label", "num_classes": 2},
}

ABLATION_MODES = ['full', 'no_filter', 'local_only', 'global_only']

# --- 2. Data Loading --- #
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def get_data_loaders(dataset_name, d_config, tokenizer, batch_size, seq_len):
    """Loads and prepares a dataset, returning DataLoaders and number of classes."""
    print(f"\n-- Loading and preparing dataset: {dataset_name} --")
    
    def tokenize_function(batch):
        return tokenizer(
            batch[d_config["text_col"]],
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt"
        )

    if dataset_name == 'sst2':
        dataset = load_dataset("glue", "sst2")
    else:
        dataset = load_dataset(dataset_name)

    dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=[col for col in dataset['train'].column_names if col not in ['input_ids', d_config['label_col']]]
    )
    
    if d_config['label_col'] != 'label':
        dataset = dataset.rename_column(d_config['label_col'], 'label')

    dataset.set_format(type="torch", columns=["input_ids", "label"])
    
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    
    if dataset_name == 'sst2':
        test_split_name = 'validation'
    else:
        test_split_name = 'test' if 'test' in dataset else 'validation'
    
    test_loader = DataLoader(dataset[test_split_name], batch_size=batch_size)
    
    return train_loader, test_loader, d_config["num_classes"]


# --- 3. Model Architecture (Zarvan) --- #
class MultiHeadLinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mode='full'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // num_heads
        self.mode = mode

        if mode == 'full' or mode == 'no_filter':
            num_local = num_heads // 2
            num_global = num_heads - num_local
        elif mode == 'local_only':
            num_local = num_heads
            num_global = 0
        elif mode == 'global_only':
            num_local = 0
            num_global = num_heads

        self.local_proj = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_local)])
        self.local_score = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_local)])
        self.global_proj = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_global)])
        self.global_score = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_global)])
        total_head_dim = (num_local + num_global) * self.head_dim
        self.fc_out = nn.Linear(total_head_dim, embed_dim)

    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        head_outputs = []
        for proj, score_fn in zip(self.local_proj, self.local_score):
            scores = score_fn(x).squeeze(-1)
            attn_weights = F.softmax(scores, dim=0).unsqueeze(-1)
            values = proj(x)
            head_outputs.append((values * attn_weights).sum(dim=0))
        if self.global_proj:
            g = x.mean(dim=0)
            for proj, score_fn in zip(self.global_proj, self.global_score):
                scores = score_fn(g)
                attn_weights = F.softmax(scores, dim=0)
                attn_weights_expanded = attn_weights.unsqueeze(0).expand(seq_len, -1, -1)
                values = proj(x)
                head_outputs.append((values * attn_weights_expanded).sum(dim=0))
        concatenated_heads = torch.cat(head_outputs, dim=-1)
        return self.fc_out(concatenated_heads)

class Zarvan(nn.Module):
    def __init__(self, seq_len, embed_dim, hidden_dim, num_heads, mode='full'):
        super().__init__()
        self.use_filter = mode == 'full'
        self.attention = MultiHeadLinearAttention(embed_dim, num_heads, mode=mode)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, embed_dim)
        )
        if self.use_filter:
            self.filter_net = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, 1, embed_dim))

    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        x = x + self.pos_encoding[:seq_len]
        residual = x
        q = self.attention(x)
        if self.use_filter:
            filter_input = torch.cat([x.mean(dim=0), q], dim=-1)
            f = self.filter_net(filter_input).unsqueeze(0).expand(seq_len, -1, -1)
            x = residual * f
        else:
            x = q.unsqueeze(0)
        x = self.norm1(residual + x)
        residual = x
        x_ffn = self.ffn(x)
        z = self.norm2(residual + x_ffn)
        return z

class ZarvanClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, hidden_dim, num_classes, num_heads, mode='full'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id)
        self.zarvan = Zarvan(seq_len, embed_dim, hidden_dim, num_heads, mode=mode)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)
        z = self.zarvan(x)
        z_pooled = z.mean(dim=0)
        return self.fc(z_pooled)

# --- 4. Training & Evaluation Loop --- #
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for batch in tqdm(loader, desc="Training"):
        x, y = batch["input_ids"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (out.argmax(1) == y).sum().item()
        total_samples += y.size(0)
    return total_loss / len(loader), total_correct / total_samples

def evaluate(model, loader, device):
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            x, y = batch["input_ids"].to(device), batch["label"].to(device)
            out = model(x)
            total_correct += (out.argmax(1) == y).sum().item()
            total_samples += y.size(0)
    return total_correct / total_samples

# --- 5. Main Execution Logic --- #
results = {}
vocab_size = tokenizer.vocab_size

for dataset_name, d_config in DATASETS_CONFIG.items():
    results[dataset_name] = {}
    train_loader, test_loader, num_classes = get_data_loaders(
        dataset_name, d_config, tokenizer, CONFIG["batch_size"], CONFIG["seq_len"]
    )

    for mode in ABLATION_MODES:
        print(f"\n🚀 Training Zarvan (mode: {mode}) on {dataset_name}...")
        results[dataset_name][mode] = {
            'train_loss': [], 'train_acc': [], 'test_acc': []
        }
        model = ZarvanClassifier(
            vocab_size, CONFIG["seq_len"], CONFIG["embed_dim"], CONFIG["hidden_dim"],
            num_classes, CONFIG["num_heads"], mode=mode
        ).to(CONFIG["device"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(CONFIG["num_epochs"]):
            start_time = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, CONFIG["device"])
            test_acc = evaluate(model, test_loader, CONFIG["device"])
            end_time = time.time()
            results[dataset_name][mode]['train_loss'].append(train_loss)
            results[dataset_name][mode]['train_acc'].append(train_acc)
            results[dataset_name][mode]['test_acc'].append(test_acc)
            print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} | "
                  f"Time: {end_time - start_time:.2f}s | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f}")

# --- 6. Plotting Results --- #

def plot_final_accuracy_bar_chart(results):
    """Plots a bar chart comparing the final test accuracy of each mode."""
    print("\n--- Generating Final Accuracy Comparison plots ---")
    for dataset_name, dataset_results in results.items():
        modes = list(dataset_results.keys())
        final_accuracies = [dataset_results[mode]['test_acc'][-1] * 100 for mode in modes]
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(modes, final_accuracies, color=plt.cm.viridis(np.linspace(0.4, 0.9, len(modes))))
        ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
        ax.set_xlabel('Ablation Mode', fontsize=12)
        ax.set_title(f'Zarvan Ablation Study on {dataset_name.upper()}', fontsize=14, weight='bold')
        ax.set_ylim(0, max(final_accuracies) * 1.15 if max(final_accuracies) > 0 else 10)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        filename = f"bar_chart_final_accuracy_{dataset_name}.png"
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.close(fig)

def plot_learning_curves(results, num_epochs):
    """Plots test accuracy and train loss curves for all modes over epochs."""
    print("\n--- Generating Learning Curve plots ---")
    epochs_range = range(1, num_epochs + 1)
    
    for dataset_name, dataset_results in results.items():
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Plot 1: Test Accuracy vs. Epochs
        for mode, history in dataset_results.items():
            ax1.plot(epochs_range, [acc * 100 for acc in history['test_acc']], marker='o', linestyle='-', label=mode)
        ax1.set_title(f'Test Accuracy vs. Epochs on {dataset_name.upper()}', fontsize=14, weight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--')
        
        # Plot 2: Training Loss vs. Epochs
        for mode, history in dataset_results.items():
            ax2.plot(epochs_range, history['train_loss'], marker='o', linestyle='-', label=mode)
        ax2.set_title(f'Training Loss vs. Epochs on {dataset_name.upper()}', fontsize=14, weight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Training Loss', fontsize=12)
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--')
        
        fig.suptitle(f'Learning Curves for {dataset_name.upper()}', fontsize=16, weight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        
        filename = f"learning_curves_{dataset_name}.png"
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.close(fig)

# Call both plotting functions at the end
plot_final_accuracy_bar_chart(results)
plot_learning_curves(results, CONFIG["num_epochs"])

print("\n--- All experiments finished! ---")