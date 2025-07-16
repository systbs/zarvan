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

# --- 1. Configuration --- #
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seq_len": 128,
    "embed_dim": 128,
    "hidden_dim": 256,
    "batch_size": 32,
    "num_epochs": 3,
    "learning_rate": 5e-5,
    "num_heads": 4,
    "margin": 0.5,
}
print(f"Using device: {CONFIG['device']}")

# --- 2. Model Architectures --- #
# Using the final 'no-filter' Zarvan as the encoder
class HybridMultiHeadLinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads; assert embed_dim % num_heads == 0; self.head_dim = embed_dim // num_heads
        num_local, num_global = num_heads // 2, num_heads - num_heads // 2
        self.local_proj = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_local)])
        self.local_score = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_local)])
        self.global_proj = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_global)])
        self.global_score = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_global)])
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)
    def forward(self, x):
        seq_len, batch_size, _ = x.shape; head_outputs = []
        for proj, score_fn in zip(self.local_proj, self.local_score):
            scores = score_fn(x).squeeze(-1)
            attn_weights = F.softmax(scores, dim=0).unsqueeze(-1)
            values = proj(x); head_outputs.append((values * attn_weights).sum(dim=0))
        g = x.mean(dim=0)
        for proj, score_fn in zip(self.global_proj, self.global_score):
            scores = score_fn(g)
            attn_weights = F.softmax(scores, dim=0).unsqueeze(0).expand(seq_len, -1, -1)
            values = proj(x); head_outputs.append((values * attn_weights).sum(dim=0))
        return self.fc_out(torch.cat(head_outputs, dim=-1))

class ZarvanEncoder(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, hidden_dim, num_heads, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, 1, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attention = HybridMultiHeadLinearAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim))
    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)
        x = x + self.pos_encoding[:x.size(0)]
        residual = x; attn_out = self.attention(self.norm1(x)); x = residual + attn_out.unsqueeze(0)
        residual = x; ffn_out = self.ffn(self.norm2(x)); x = residual + ffn_out
        return x.mean(dim=0)

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, hidden_dim, num_heads, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=0.1, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)
        x = x + self.pos_encoding[:x.size(0)]
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

# --- 3. Data Handling --- #
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class TripletDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, seq_len):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    def __len__(self):
        return len(self.hf_dataset)
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        query_text = item['query']
        
        # Separate positive and negative passages
        passages = item['passages']['passage_text']
        is_selected = item['passages']['is_selected']
        
        positive_passages = [p for i, p in enumerate(passages) if is_selected[i] == 1]
        negative_passages = [p for i, p in enumerate(passages) if is_selected[i] == 0]
        
        # Randomly select one from each list
        pos_text = random.choice(positive_passages)
        neg_text = random.choice(negative_passages)
        
        # Tokenize the chosen triplet
        query = self.tokenizer(query_text, padding="max_length", truncation=True, max_length=self.seq_len, return_tensors='pt')['input_ids'].squeeze(0)
        pos_doc = self.tokenizer(pos_text, padding="max_length", truncation=True, max_length=self.seq_len, return_tensors='pt')['input_ids'].squeeze(0)
        neg_doc = self.tokenizer(neg_text, padding="max_length", truncation=True, max_length=self.seq_len, return_tensors='pt')['input_ids'].squeeze(0)
        
        return query, pos_doc, neg_doc

# --- 4. Training and Evaluation --- #
def train_ir_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for query, pos_doc, neg_doc in tqdm(loader, desc="Training IR Model"):
        query, pos_doc, neg_doc = query.to(device), pos_doc.to(device), neg_doc.to(device)
        optimizer.zero_grad()
        q_vec, p_vec, n_vec = model(query, pos_doc, neg_doc)
        loss = criterion(q_vec, p_vec, n_vec)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_ir(model, loader, device):
    model.eval()
    correct_rankings, total_samples = 0, 0
    with torch.no_grad():
        for query, pos_doc, neg_doc in tqdm(loader, desc="Evaluating IR Model"):
            query, pos_doc, neg_doc = query.to(device), pos_doc.to(device), neg_doc.to(device)
            q_vec, p_vec, n_vec = model(query, pos_doc, neg_doc)
            dist_pos = 1 - F.cosine_similarity(q_vec, p_vec)
            dist_neg = 1 - F.cosine_similarity(q_vec, n_vec)
            correct_rankings += (dist_pos < dist_neg).sum().item()
            total_samples += query.size(0)
    return 100 * correct_rankings / total_samples

# --- 5. Main Execution --- #
print("Loading MS MARCO dataset (small subset for speed)...")
# Filter the dataset to ensure each example has at least one positive and one negative passage
def has_pos_and_neg(example):
    is_selected_list = example['passages']['is_selected']
    return 1 in is_selected_list and 0 in is_selected_list

dataset = load_dataset("ms_marco", "v2.1", split='train').filter(has_pos_and_neg)
# Use a small, shuffled subset for a quick experiment
dataset = dataset.shuffle(seed=42).select(range(20000))

dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_data = dataset_split['train']
test_data = dataset_split['test']

train_dataset = TripletDataset(train_data, tokenizer, CONFIG['seq_len'])
test_dataset = TripletDataset(test_data, tokenizer, CONFIG['seq_len'])
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

vocab_size = tokenizer.vocab_size
padding_idx = tokenizer.pad_token_id

encoder_params = {
    'vocab_size': vocab_size, 'seq_len': CONFIG['seq_len'], 'embed_dim': CONFIG['embed_dim'],
    'hidden_dim': CONFIG['hidden_dim'], 'num_heads': CONFIG['num_heads'], 'padding_idx': padding_idx
}

models_to_test = {
    "Zarvan": TwoTowerModel(ZarvanEncoder(**encoder_params)),
    "Transformer": TwoTowerModel(TransformerEncoder(**encoder_params))
}
results_summary = []

for model_name, model in models_to_test.items():
    print(f"\n{'='*20} TESTING {model_name} for IR {'='*20}")
    model.to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.TripletMarginLoss(margin=CONFIG['margin'], p=2)
    
    start_time = time.time()
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        loss = train_ir_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
        print(f"Epoch {epoch}/{CONFIG['num_epochs']} -> Triplet Loss: {loss:.4f}")
    
    ranking_acc = evaluate_ir(model, test_loader, CONFIG['device'])
    total_time = time.time() - start_time
    
    print(f"\nFinal Results for {model_name}:")
    print(f"Ranking Accuracy: {ranking_acc:.2f}% | Total Time: {total_time:.2f}s")
    
    results_summary.append({'model': model_name, 'ranking_accuracy': ranking_acc, 'time': total_time})

# --- 6. Final Report --- #
print("\n\n" + "="*50)
print("FINAL INFORMATION RETRIEVAL RESULTS")
print("="*50)
df = pd.DataFrame(results_summary)
print(df.to_string())