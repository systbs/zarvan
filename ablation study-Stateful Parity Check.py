# ============================================================================
#
#    ADVANCED Ablation Study for Zarvan's Gating Mechanism
#
# This script implements the "Stateful Parity Check" task, a much more
# challenging experiment designed to isolate and prove the necessity of
# the gating mechanism by forcing the model to rely on non-local,
# stateful context.
#
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import random
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Part 1: Experiment Configuration
# ============================================================================

# --- Model & Task Hyperparameters ---
# Vocabulary: 0=PAD, 1=DATA_1, 2=DATA_2, 3=FLIP
VOCAB_SIZE = 4
SEQ_LEN = 150
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4 # Increased layers for more complexity

# --- Special Tokens ---
PAD_TOKEN = 0
DATA_1_TOKEN = 1
DATA_2_TOKEN = 2
FLIP_TOKEN = 3

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3
NUM_EPOCHS = 60
STEPS_PER_EPOCH = 50
BATCH_SIZE = 64

# ============================================================================
# Part 2: Stateful Data Generator
# ============================================================================

def create_stateful_batch(batch_size, seq_len):
    """
    Generates a batch for the "Stateful Parity Check" task.
    """
    x = torch.full((batch_size, seq_len), PAD_TOKEN, dtype=torch.long)
    y = torch.full((batch_size, seq_len), PAD_TOKEN, dtype=torch.long)

    for i in range(batch_size):
        is_flipped = False
        # Place 4-8 FLIP tokens to ensure state changes
        num_flips = random.randint(4, 8)
        flip_indices = sorted(random.sample(range(seq_len), num_flips))
        
        flip_ptr = 0
        for j in range(seq_len):
            if flip_ptr < len(flip_indices) and j == flip_indices[flip_ptr]:
                x[i, j] = FLIP_TOKEN
                y[i, j] = PAD_TOKEN  # The FLIP token itself has no output
                is_flipped = not is_flipped
                flip_ptr += 1
            else:
                data_token = random.choice([DATA_1_TOKEN, DATA_2_TOKEN])
                x[i, j] = data_token
                
                if not is_flipped:
                    y[i, j] = data_token # Normal state: copy
                else:
                    # Flipped state: invert
                    y[i, j] = DATA_2_TOKEN if data_token == DATA_1_TOKEN else DATA_1_TOKEN
    return x, y


# ============================================================================
# Part 3: Model Architectures (Full, NoGate, NoContext) - UNCHANGED
# The model definitions from the previous script are used here without change.
# ============================================================================

# --- Base Components ---
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 512):
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
        weights = F.softmax(s, dim=-2) # Softmax over sequence
        head_outputs = torch.sum(weights * v, dim=2)
        concatenated_heads = head_outputs.reshape(B, self.embed_dim)
        return self.combine(concatenated_heads)

class AssociativeContextExtractor(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.s_proj = nn.Linear(embed_dim, 1)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, v = self.s_proj(x), self.v_proj(x)
        weights = F.softmax(s, dim=1)
        context = torch.sum(weights * v, dim=1)
        return context

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- 1. Full Zarvan Block ---
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
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        
        q_holistic = self.holistic_ctx(x_norm).unsqueeze(1).expand(-1, x.size(1), -1)
        q_associative = self.associative_ctx(x_norm).unsqueeze(1).expand(-1, x.size(1), -1)
        
        gate_input = torch.cat([x_norm, q_holistic, q_associative], dim=-1)
        input_gate, forget_gate = self.gate_net(gate_input).chunk(2, dim=-1)
        
        gated_x = torch.sigmoid(input_gate) * x_norm + \
                  torch.sigmoid(forget_gate) * self.update_proj(x_norm)
        
        x = residual + gated_x
        x = x + self.ffn(self.norm2(x))
        return x

# --- 2. Zarvan-NoGate Block ---
class ZarvanBlockNoGate(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.holistic_ctx = HolisticContextExtractor(embed_dim, num_heads)
        self.associative_ctx = AssociativeContextExtractor(embed_dim)
        self.combiner = nn.Linear(embed_dim * 3, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        
        q_holistic = self.holistic_ctx(x_norm).unsqueeze(1).expand(-1, x.size(1), -1)
        q_associative = self.associative_ctx(x_norm).unsqueeze(1).expand(-1, x.size(1), -1)
        
        combined_input = torch.cat([x_norm, q_holistic, q_associative], dim=-1)
        update = self.combiner(combined_input)
        
        gated_x = x_norm + update
        
        x = residual + gated_x
        x = x + self.ffn(self.norm2(x))
        return x

# --- 3. Zarvan-NoContext Block ---
class ZarvanBlockNoContext(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim * 2)
        )
        self.update_proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        
        input_gate, forget_gate = self.gate_net(x_norm).chunk(2, dim=-1)
        gated_x = torch.sigmoid(input_gate) * x_norm + \
                  torch.sigmoid(forget_gate) * self.update_proj(x_norm)
        
        x = residual + gated_x
        x = x + self.ffn(self.norm2(x))
        return x

# --- Universal Model Wrapper ---
class AblationModel(nn.Module):
    def __init__(self, block_class, vocab_size, num_classes, seq_len,
                 embed_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        self.layers = nn.ModuleList([
            block_class(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_head(x)

# ============================================================================
# Part 4: Training and Evaluation
# ============================================================================

def calculate_accuracy(logits, targets, ignore_index=PAD_TOKEN):
    predictions = torch.argmax(logits, dim=-1)
    mask = (targets != ignore_index)
    if mask.sum() == 0:
        return 1.0 # Avoid division by zero if all are pads
    correct = (predictions == targets) & mask
    return correct.sum().float() / mask.sum().float()

def train_model(model, model_name, device):
    print(f"\n--- 🏋️ Training {model_name} ---")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    accuracy_history = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        for step in range(STEPS_PER_EPOCH):
            x, y = create_stateful_batch(BATCH_SIZE, SEQ_LEN)
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += calculate_accuracy(logits, y).item()
        
        avg_acc = epoch_acc / STEPS_PER_EPOCH
        accuracy_history.append(avg_acc)
        
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | Loss: {epoch_loss/STEPS_PER_EPOCH:.4f} | Accuracy: {avg_acc:.4f}")
            
    return accuracy_history

def plot_results(results):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model_name, history in results.items():
        ax.plot(history, label=model_name, lw=2.5, marker='o', markersize=4, markevery=10)
        
    ax.set_title("Stateful Parity Check: Proving the Need for Gating", fontsize=18)
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Token-level Accuracy", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_ylim(0.45, 1.05) # Start y-axis from near 50%
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random Guess Baseline (50%)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    fig_path = "stateful_gating_ablation_results.png"
    plt.savefig(fig_path)
    print(f"\n✅ Results plot saved to {fig_path}")

# ============================================================================
# Part 5: Main Execution
# ============================================================================

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device.type.upper()}")

    # Added LayerNorms and residual connections to stabilize training for this harder task
    models_to_test = {
        "Zarvan (Full)": AblationModel(ZarvanBlock, VOCAB_SIZE, VOCAB_SIZE, SEQ_LEN, EMBED_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS),
        "Zarvan-NoGate": AblationModel(ZarvanBlockNoGate, VOCAB_SIZE, VOCAB_SIZE, SEQ_LEN, EMBED_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS),
        "Zarvan-NoContext": AblationModel(ZarvanBlockNoContext, VOCAB_SIZE, VOCAB_SIZE, SEQ_LEN, EMBED_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS)
    }
    
    results = {}
    
    for name, model in models_to_test.items():
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {name} | Parameters: {param_count/1e6:.2f}M")
        accuracy_history = train_model(model, name, device)
        results[name] = accuracy_history
        
    plot_results(results)