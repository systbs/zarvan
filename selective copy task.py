# ============================================================================
#
#       Selective Copy Task with the Definitive Zarvan Architecture
#
# This script validates that the final, hybrid Zarvan architecture retains its
# strong performance on simpler tasks like selective copy, ensuring that our
# improvements for long-range dependencies did not cause a regression.
#
# Author: Yasser Sajjadi, in collaboration with Google's Gemini
#
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import time
from tqdm.auto import tqdm

# ============================================================================
# Part 1: Core Context Kernels (Definitive Version)
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Injects sinusoidal positional information into the input sequence.
    """
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
    """
    Extracts a holistic context vector from the entire sequence.
    """
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
    """
    Extracts an associative context vector by learning importance scores.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.importance_scorer = nn.Sequential(nn.Linear(embed_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.importance_scorer(x)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

# ============================================================================
# Part 2: The Definitive Zarvan Block
# ============================================================================

class ZarvanBlock(nn.Module):
    """
    The definitive, hybrid Zarvan Block.
    """
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

# ============================================================================
# Part 3: The Complete Model for Sequence-to-Sequence Tasks
# ============================================================================

class ZarvanForSequenceModeling(nn.Module):
    """
    Complete Zarvan model optimized for sequence modeling tasks like selective copy.
    """
    def __init__(self, vocab_size, seq_len, embed_dim, hidden_dim, num_heads, num_layers, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        # ✅ Using the definitive, hybrid Zarvan Block
        self.layers = nn.ModuleList([
            ZarvanBlock(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        # For this task, we need the logits for the entire sequence
        logits = self.lm_head(x)
        return logits

# ============================================================================
# Part 4: Test Harness for Selective Copy
# ============================================================================

class SelectiveCopyDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000):
        self.vocab_size, self.seq_len, self.num_samples = vocab_size, seq_len, num_samples
        self.PAD_TOKEN, self.NOISE_TOKEN, self.GO_TOKEN, self.COPY_TOKEN = 0, 1, 2, 3
        self.first_regular_token = 4

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_seq = torch.full((self.seq_len,), self.NOISE_TOKEN, dtype=torch.long)
        target_seq = torch.full((self.seq_len,), -100, dtype=torch.long) # -100 is CrossEntropy's ignore_index
        value_to_copy = random.randint(self.first_regular_token, self.vocab_size - 1)
        go_pos = random.randint(0, self.seq_len // 10)
        copy_pos = random.randint(self.seq_len - (self.seq_len // 10), self.seq_len - 1)
        input_seq[go_pos] = self.GO_TOKEN
        if go_pos + 1 < self.seq_len:
            input_seq[go_pos + 1] = value_to_copy
        input_seq[copy_pos] = self.COPY_TOKEN
        target_seq[copy_pos] = value_to_copy
        return input_seq, target_seq

if __name__ == '__main__':
    print("--- Testing Definitive Zarvan on Selective Copy Task ---")

    # Hyperparameters
    SEQ_LEN = 256
    VOCAB_SIZE = 20
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    NUM_HEADS = 4
    NUM_LAYERS = 2
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 15
    TRAIN_SAMPLES = 10000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on device: {device}")

    train_dataset = SelectiveCopyDataset(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, num_samples=TRAIN_SAMPLES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = ZarvanForSequenceModeling(
        vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        padding_idx=train_dataset.PAD_TOKEN
    ).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"\nStarting training for {EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_correct, total_predictions = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                target_mask = targets != -100
                if target_mask.sum() > 0:
                    predicted_indices = outputs[target_mask].argmax(dim=-1)
                    actual_values = targets[target_mask]
                    total_correct += (predicted_indices == actual_values).sum().item()
                    total_predictions += target_mask.sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = (total_correct / total_predictions) * 100 if total_predictions > 0 else 0
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")

    print("\n--- Evaluating model on a new sample ---")
    model.eval()
    with torch.no_grad():
        test_dataset = SelectiveCopyDataset(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, num_samples=1)
        input_sample, _ = test_dataset[0]
        
        go_pos = (input_sample == test_dataset.GO_TOKEN).nonzero(as_tuple=True)[0].item()
        copy_pos = (input_sample == test_dataset.COPY_TOKEN).nonzero(as_tuple=True)[0].item()
        value_to_copy = input_sample[go_pos + 1].item()
        
        print(f"Input: GO at pos {go_pos}, VALUE '{value_to_copy}' at pos {go_pos + 1}, COPY at pos {copy_pos}.")
        
        output_logits = model(input_sample.unsqueeze(0).to(device))
        predicted_token = output_logits.squeeze(0)[copy_pos].argmax().item()
        
        print(f"\nExpected Prediction: {value_to_copy}")
        print(f"Model's Prediction:  {predicted_token}")
        
        if predicted_token == value_to_copy:
            print("\n✅ Success! The definitive model correctly handled the copy task.")
        else:
            print("\n❌ Failure. The definitive model regressed on the copy task.")
