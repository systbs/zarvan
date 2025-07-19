# ===================================================================
#
#   MNIST Benchmark: Definitive Zarvan vs. Transformer
#
# This script runs a benchmark comparing the final, hybrid Zarvan
# architecture against a standard Transformer baseline on the
# MNIST image classification task, treating images as sequences of pixels.
#
# ===================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import math
from tqdm.auto import tqdm
import os

# --- 1. Environment Detection and Universal Setup ---
try:
    # Attempt to import PyTorch/XLA libraries for TPU support
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    IS_TPU = True
    print("TPU environment detected. Using PyTorch/XLA.")
except ImportError:
    IS_TPU = False
    print("TPU environment not detected. Running on CPU/GPU.")

# --- 2. Configuration (FLAGS) ---
FLAGS = {
    "seq_len": 28 * 28,
    "embed_dim": 128,
    "hidden_dim": 256,
    "ff_dim": 512, # Feed-forward dim for Transformer
    "total_batch_size": 128,
    "num_epochs": 10,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "num_heads": 4,
    "num_layers": 2, # Number of layers for both models
    "num_workers": 0, # Set to 0 to avoid multiprocessing issues
}

# ============================================================================
# Part 3: Definitive Zarvan Architecture
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

# --- 4. Classifier Models for MNIST ---
class ZarvanMNISTClassifier(nn.Module):
    def __init__(self, seq_len, embed_dim, hidden_dim, num_classes, num_heads, num_layers):
        super().__init__()
        self.pixel_embedding = nn.Linear(1, embed_dim) # Embed each pixel's intensity
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        self.layers = nn.ModuleList([
            ZarvanBlock(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, -1, 1) # Flatten image into a sequence
        x = self.pixel_embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        z = x.mean(dim=1) # Global average pooling
        return self.fc(z)

class TransformerMNISTClassifier(nn.Module):
    def __init__(self, seq_len, embed_dim, ff_dim, num_classes, num_heads, num_layers):
        super().__init__()
        self.pixel_embedding = nn.Linear(1, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, -1, 1)
        x = self.pixel_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        z = x.mean(dim=1)
        return self.fc(z)

# --- 5. Universal Training and Evaluation Functions ---
def run_epoch(is_training, model, loader, optimizer, criterion, device):
    model.train(is_training)
    total_loss, total_correct, total_samples = 0, 0, 0
    
    progress_bar = tqdm(loader, desc="Training" if is_training else "Evaluating", leave=False)
    
    with torch.set_grad_enabled(is_training):
        for data, target in progress_bar:
            # ✅ FIX: Move data and targets to the correct device
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                if IS_TPU:
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_correct += (output.argmax(1) == target).sum().item()
            total_samples += target.size(0)

    if IS_TPU and not is_training:
        total_correct = xm.mesh_reduce('val_correct', total_correct, sum)
        total_samples = xm.mesh_reduce('val_samples', total_samples, sum)

    avg_loss = total_loss / total_samples
    accuracy = 100. * total_correct / total_samples
    return avg_loss, accuracy

# --- 6. Main Execution Logic ---
def main():
    if IS_TPU:
        rank = xm.get_ordinal()
        device = xm.xla_device()
        batch_size = FLAGS['total_batch_size'] // xm.xrt_world_size()
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = FLAGS['total_batch_size']
    
    if rank == 0:
        print(f"Running on Device: {str(device).upper()} | Total Batch Size: {FLAGS['total_batch_size']}")

    if rank == 0 and not os.path.exists("./data/MNIST"):
        print("Downloading MNIST dataset...")
        datasets.MNIST("./data", train=True, download=True)
    
    if IS_TPU:
        xm.rendezvous('download_complete')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("./data", train=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_sampler = None
    if IS_TPU:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=xm.xrt_world_size(), rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=FLAGS['num_workers'], pin_memory=not IS_TPU)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=FLAGS['num_workers'], pin_memory=not IS_TPU)

    if IS_TPU:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        test_loader = pl.MpDeviceLoader(test_loader, device)
        
    models_to_test = {
        "Zarvan": ZarvanMNISTClassifier(
            seq_len=FLAGS['seq_len'], embed_dim=FLAGS['embed_dim'], hidden_dim=FLAGS['hidden_dim'],
            num_classes=10, num_heads=FLAGS['num_heads'], num_layers=FLAGS['num_layers']),
        "Transformer": TransformerMNISTClassifier(
            seq_len=FLAGS['seq_len'], embed_dim=FLAGS['embed_dim'], ff_dim=FLAGS['ff_dim'],
            num_classes=10, num_heads=FLAGS['num_heads'], num_layers=FLAGS['num_layers'])
    }
    
    criterion = nn.CrossEntropyLoss()
    history = {name: {'train_acc': [], 'test_acc': []} for name in models_to_test.keys()}

    for name, model in models_to_test.items():
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
        
        if rank == 0:
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            print(f"\n🚀 Training {name} ({param_count:.2f}M params) on MNIST...")
            
        for epoch in range(1, FLAGS['num_epochs'] + 1):
            train_loss, train_acc = run_epoch(True, model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = run_epoch(False, model, test_loader, None, criterion, device)
            
            if rank == 0:
                history[name]['train_acc'].append(train_acc)
                history[name]['test_acc'].append(test_acc)
                print(f"Epoch {epoch}/{FLAGS['num_epochs']} -> Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
                
    if rank == 0:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 8))
        plt.title(f'Zarvan vs. Transformer on MNIST ({str(device).upper()})', fontsize=16)
        epochs_range = range(1, FLAGS['num_epochs'] + 1)
        for name, hist in history.items():
            plt.plot(epochs_range, hist['train_acc'], marker='o', linestyle='-', label=f'{name} Train Acc')
            plt.plot(epochs_range, hist['test_acc'], marker='s', linestyle='--', label=f'{name} Test Acc')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.xticks(epochs_range)
        plt.tight_layout()
        plt.savefig("mnist_universal_results.png")
        print("\nSaved final plot to 'mnist_universal_results.png'")

# --- Universal Entry Point ---
if __name__ == '__main__':
    if IS_TPU:
        def _mp_fn_wrapper(rank, flags):
            main()
        xmp.spawn(_mp_fn_wrapper, args=(FLAGS,), nprocs=None, start_method='fork')
    else:
        main()
