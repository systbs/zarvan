# zarvan_test_single_file_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import time
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ============================================================================
# SECTION 1: CORE ZARVAN MODEL ARCHITECTURE (With improvements)
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class LinearQueryExtractor(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.s_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        s = self.s_proj(x)
        v = self.v_proj(x)
        s = s.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        weights = F.softmax(s, dim=-1) # Softmax over feature dimension
        head_outputs = (weights * v).sum(dim=2)
        concatenated_heads = head_outputs.reshape(B, self.embed_dim)
        q = self.combine(concatenated_heads)
        return q

class ZarvanBlock(nn.Module):
    """ The core Zarvan block with ENHANCED information flow. """
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.query_extractor = LinearQueryExtractor(embed_dim, num_heads)
        self.interactive_context = nn.Linear(embed_dim, embed_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim * 2)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.gated_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        q = self.query_extractor(x)
        i_ctx = F.gelu(self.interactive_context(q))
        q_exp = q.unsqueeze(1).expand(-1, S, -1)
        i_ctx_exp = i_ctx.unsqueeze(1).expand(-1, S, -1)
        gate_input = torch.cat([x, q_exp, i_ctx_exp], dim=-1)
        gates = self.gate_net(gate_input)
        input_gate, forget_gate = gates.chunk(2, dim=-1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        gated_x = input_gate * x + forget_gate * self.gated_proj(x)
        
        # 🚀 KEY CHANGE: Directly inject global context before the FFN
        ffn_input = gated_x + q_exp + i_ctx_exp
        output = self.ffn(ffn_input)
        
        return self.norm(x + output)

# ============================================================================
# SECTION 2: MODEL VARIANT AND DATASET (With multiple layers)
# ============================================================================

class ZarvanForSequenceModeling(nn.Module):
    """ Zarvan model adapted for sequence modeling with multiple layers. """
    def __init__(self, vocab_size: int, seq_len: int, embed_dim: int, 
                 hidden_dim: int, num_heads: int, num_layers: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=seq_len)
        self.layers = nn.ModuleList([
            ZarvanBlock(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return logits

class SelectiveCopyDataset(Dataset):
    """ Generates data for the selective copy task. """
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.PAD_TOKEN, self.NOISE_TOKEN, self.GO_TOKEN, self.COPY_TOKEN = 0, 1, 2, 3
        self.first_regular_token = 4

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_seq = torch.full((self.seq_len,), self.NOISE_TOKEN, dtype=torch.long)
        target_seq = torch.full((self.seq_len,), -100, dtype=torch.long)
        value_to_copy = random.randint(self.first_regular_token, self.vocab_size - 1)
        go_pos = random.randint(0, self.seq_len // 10)
        copy_pos = random.randint(self.seq_len - (self.seq_len // 10), self.seq_len - 1)
        input_seq[go_pos] = self.GO_TOKEN
        if go_pos + 1 < self.seq_len:
            input_seq[go_pos + 1] = value_to_copy
        input_seq[copy_pos] = self.COPY_TOKEN
        target_seq[copy_pos] = value_to_copy
        return input_seq, target_seq

# ============================================================================
# SECTION 3: TRAINING AND EVALUATION SCRIPT
# ============================================================================

if __name__ == '__main__':
    print("--- Setting up Selective Copy Task for Zarvan (v2 - Improved) ---")

    # ✅ Tuned Hyperparameters
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

    # Create dataset and dataloader
    train_dataset = SelectiveCopyDataset(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, num_samples=TRAIN_SAMPLES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Instantiate the model with new parameters
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

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_correct, total_predictions = 0, 0, 0

        for inputs, targets in train_loader:
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
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")

    # --- Evaluation ---
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
            print("\n✅ Success! The model correctly copied the value.")
        else:
            print("\n❌ Failure. The model did not copy the correct value.")