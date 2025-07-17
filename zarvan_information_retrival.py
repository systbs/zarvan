import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import time
import pandas as pd
import random
import math

# --- 1. Configuration --- #
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seq_len": 1024,
    "embed_dim": 128,
    "hidden_dim": 256, # For Zarvan's filter net
    "ff_dim": 512,     # For Transformer's FFN, a common ratio is 4*embed_dim
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 5e-5,
    "num_heads": 4,
    "margin": 0.5,
}
print(f"Using device: {CONFIG['device']}")

# --- 2. Building Blocks (The SUCCESSFUL Zarvan Architecture) --- #

class SinusoidalPositionalEncoding(nn.Module):
    # Using the fixed, non-learned positional encoding from the first experiments
    def __init__(self, embed_dim: int, max_len: int = 2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (torch.Tensor): (seq_len, batch_size, embed_dim)"""
        return x + self.pe[:x.size(0)]

class ParallelUnifiedAttention(nn.Module):
    # The FAST, vectorized attention mechanism without Python loops
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_net = nn.Linear(embed_dim, embed_dim * num_heads)
        self.combine = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x (torch.Tensor): (seq_len, batch_size, embed_dim)
        Returns: torch.Tensor: Context vector (batch_size, embed_dim)
        """
        seq_len, batch_size, _ = x.shape
        scores_all_heads = self.query_net(x).view(seq_len, batch_size * self.num_heads, self.embed_dim)
        attn_weights = F.softmax(scores_all_heads, dim=0)
        x_repeated = x.repeat_interleave(self.num_heads, dim=1)
        weighted_values = x_repeated * attn_weights
        head_outputs = weighted_values.sum(dim=0)
        concatenated_heads = head_outputs.view(batch_size, self.num_heads * self.embed_dim)
        return self.combine(concatenated_heads)

class CorrectedZarvanBlock(nn.Module):
    # The original, successful Zarvan block with the filter mechanism
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        self.attention = ParallelUnifiedAttention(embed_dim, num_heads)
        self.interactive_context = nn.Linear(embed_dim, embed_dim)
        self.filter_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(embed_dim)
        self.Linear_xw = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (torch.Tensor): (seq_len, batch_size, embed_dim)"""
        q = self.attention(x)
        i_ctx = F.relu(self.interactive_context(q))
        q_exp = q.unsqueeze(0).expand(x.size(0), -1, -1)
        i_ctx_exp = i_ctx.unsqueeze(0).expand(x.size(0), -1, -1)
        filter_in = torch.cat([x, q_exp, i_ctx_exp], dim=-1)
        f_weights = self.filter_net(filter_in)
        f_weights = f_weights - f_weights.mean(dim=0, keepdim=True) # Centering the filter
        z = x + self.Linear_xw(x * f_weights)
        return self.norm(z)

# --- 3. Encoder Models for the Two-Tower Setup --- #

class ZarvanEncoder(nn.Module):
    # This encoder uses our PROVEN Zarvan block
    def __init__(self, vocab_size, seq_len, embed_dim, hidden_dim, num_heads, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=seq_len)
        self.zarvan_block = CorrectedZarvanBlock(embed_dim, hidden_dim, num_heads)
        
    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)
        x = self.pos_encoder(x)
        processed_seq = self.zarvan_block(x)
        # Average pooling over the sequence to get a single vector representation
        return processed_seq.mean(dim=0)

class TransformerEncoderModel(nn.Module):
    # The standard Transformer baseline
    def __init__(self, vocab_size, seq_len, embed_dim, ff_dim, num_heads, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=False, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)
        x = self.pos_encoder(x)
        return self.transformer(x).mean(dim=0)

class TwoTowerModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, query, pos_doc, neg_doc):
        q_vec = self.encoder(query)
        p_vec = self.encoder(pos_doc)
        n_vec = self.encoder(neg_doc)
        return q_vec, p_vec, n_vec

# --- 4. Data Handling (Identical to previous attempt) --- #
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class TripletDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, seq_len):
        self.hf_dataset, self.tokenizer, self.seq_len = hf_dataset, tokenizer, seq_len
    def __len__(self): return len(self.hf_dataset)
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        passages = item['passages']['passage_text']
        is_selected = item['passages']['is_selected']
        positive_passages = [p for i, p in enumerate(passages) if is_selected[i] == 1]
        negative_passages = [p for i, p in enumerate(passages) if is_selected[i] == 0]
        pos_text = random.choice(positive_passages)
        neg_text = random.choice(negative_passages)
        tokenize_fn = lambda text: self.tokenizer(text, padding="max_length", truncation=True, max_length=self.seq_len, return_tensors='pt')['input_ids'].squeeze(0)
        return tokenize_fn(item['query']), tokenize_fn(pos_text), tokenize_fn(neg_text)

# --- 5. Training and Evaluation (Identical to previous attempt) --- #
def train_ir_epoch(model, loader, optimizer, criterion, device):
    model.train(); total_loss = 0
    for query, pos_doc, neg_doc in tqdm(loader, desc="Training IR Model"):
        query, pos_doc, neg_doc = query.to(device), pos_doc.to(device), neg_doc.to(device)
        optimizer.zero_grad()
        q_vec, p_vec, n_vec = model(query, pos_doc, neg_doc)
        loss = criterion(q_vec, p_vec, n_vec)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_ir(model, loader, device):
    model.eval(); correct_rankings, total_samples = 0, 0
    with torch.no_grad():
        for query, pos_doc, neg_doc in tqdm(loader, desc="Evaluating IR Model"):
            query, pos_doc, neg_doc = query.to(device), pos_doc.to(device), neg_doc.to(device)
            q_vec, p_vec, n_vec = model(query, pos_doc, neg_doc)
            dist_pos = 1 - F.cosine_similarity(q_vec, p_vec)
            dist_neg = 1 - F.cosine_similarity(q_vec, n_vec)
            correct_rankings += (dist_pos < dist_neg).sum().item()
            total_samples += query.size(0)
    return 100 * correct_rankings / total_samples

# --- 6. Main Execution --- #
print("Loading and preparing MS MARCO dataset..."); t = time.time()
dataset = load_dataset("ms_marco", "v2.1", split='train').filter(
    lambda ex: 1 in ex['passages']['is_selected'] and 0 in ex['passages']['is_selected']
).shuffle(seed=42).select(range(20000)) # Using 20k samples for a reasonable runtime
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_loader = DataLoader(TripletDataset(dataset_split['train'], tokenizer, CONFIG['seq_len']), batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
test_loader = DataLoader(TripletDataset(dataset_split['test'], tokenizer, CONFIG['seq_len']), batch_size=CONFIG['batch_size'], num_workers=2)
print(f"Data ready in {time.time() - t:.2f}s")

models_to_test = {
    "Zarvan (Corrected)": TwoTowerModel(ZarvanEncoder(
        vocab_size=tokenizer.vocab_size,
        seq_len=CONFIG['seq_len'],
        embed_dim=CONFIG['embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_heads=CONFIG['num_heads'],
        padding_idx=tokenizer.pad_token_id
    )),
    "Transformer": TwoTowerModel(TransformerEncoderModel(
        vocab_size=tokenizer.vocab_size,
        seq_len=CONFIG['seq_len'],
        embed_dim=CONFIG['embed_dim'],
        ff_dim=CONFIG['ff_dim'],
        num_heads=CONFIG['num_heads'],
        padding_idx=tokenizer.pad_token_id
    ))
}

results_summary = []
for model_name, model in models_to_test.items():
    print(f"\n{'='*20} TESTING {model_name} for IR {'='*20}")
    model.to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.TripletMarginWithDistanceLoss(margin=CONFIG['margin'], distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
    
    start_time = time.time()
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        loss = train_ir_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
        print(f"Epoch {epoch}/{CONFIG['num_epochs']} -> Triplet Loss: {loss:.4f}")
    
    ranking_acc = evaluate_ir(model, test_loader, CONFIG['device'])
    total_time = time.time() - start_time
    
    print(f"\nFinal Results for {model_name}:")
    print(f"Ranking Accuracy: {ranking_acc:.2f}% | Total Time: {total_time:.2f}s")
    results_summary.append({'model': model_name, 'ranking_accuracy': ranking_acc, 'time': total_time})

# --- 7. Final Report --- #
print("\n\n" + "="*50)
print("FINAL CORRECTED INFORMATION RETRIEVAL RESULTS")
print("="*50)
df = pd.DataFrame(results_summary)
print(df.to_string())