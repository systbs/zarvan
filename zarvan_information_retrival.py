# ===================================================================
# Step 1 (for Kaggle TPU): Install PyTorch/XLA - Run this once in a separate cell
# ===================================================================
# !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# !python pytorch-xla-env-setup.py --version 2.0.0 --apt-packages libomp5

# ===================================================================
# Step 2: Main Universal Script - Run this in the next cell
# ===================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import math
from tqdm.auto import tqdm
import os
import pandas as pd
import random

# Import Hugging Face libraries
from datasets import load_dataset
from transformers import AutoTokenizer

# --- 1. Environment Detection and Universal Setup ---
try:
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
    "seq_len": 128,
    "embed_dim": 128,
    "hidden_dim": 256,
    "ff_dim": 512,
    "total_batch_size": 64, # Will be divided across TPU cores
    "num_epochs": 3,
    "learning_rate": 5e-5,
    "num_heads": 4,
    "margin": 0.5,
    "num_workers": 2,
    "dataset_name": "ms_marco",
    "dataset_version": "v2.1",
    "train_samples": 20000, # Using a subset for faster experimentation
    "test_split_size": 0.1,
    "num_zarvan_blocks": 2, # Added for stacking ZarvanBlocks
}

# --- 3. Refactored Zarvan Model Architecture (B, S, E) ---

class SinusoidalPositionalEncoding(nn.Module):
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

class LinearQueryExtractor(nn.Module):
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
        
        # Applying softmax across sequence dimension for each head to get weights
        weights = F.softmax(s, dim=-1) 
        
        # Weighted sum of values (S) based on weights (S)
        head_outputs = (weights * v).sum(dim=2) # Summing across sequence length dimension (S)
        
        concatenated_heads = head_outputs.reshape(B, self.embed_dim)
        return self.combine(concatenated_heads)

class GatedZarvanBlock(nn.Module): # Renamed to reflect gating mechanism
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.query_extractor = LinearQueryExtractor(embed_dim, num_heads)
        self.interactive_context = nn.Linear(embed_dim, embed_dim)
        
        # Filter net now outputs 2 * embed_dim for input and forget gates
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim), 
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim * 2) # Outputs 2 gates
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential( # Feed-forward network after gating
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.gated_proj = nn.Linear(embed_dim, embed_dim) # Existing gated projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        
        # Extract query and interactive context (global representations)
        q = self.query_extractor(x)
        i_ctx = F.gelu(self.interactive_context(q))
        
        # Expand global representations to match sequence length for per-token interaction
        q_exp = q.unsqueeze(1).expand(-1, S, -1)
        i_ctx_exp = i_ctx.unsqueeze(1).expand(-1, S, -1)
        
        # Concatenate original input, query, and interactive context for gate calculation
        gate_input = torch.cat([x, q_exp, i_ctx_exp], dim=-1)
        
        # Calculate gate values
        gates = self.gate_net(gate_input)
        input_gate, forget_gate = gates.chunk(2, dim=-1) # Split into two gates
        
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        
        # Apply gating mechanism: select which parts of the input to "copy" or update
        # This is the core of the selective copy mechanism
        gated_x = input_gate * x + forget_gate * self.gated_proj(x) # Simplified update
        
        # Apply a feed-forward network to the gated output
        output = self.ffn(gated_x)
        
        # Residual connection and layer normalization
        return self.norm(x + output)


class ZarvanEncoder(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, hidden_dim, num_heads, padding_idx, num_zarvan_blocks=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=seq_len)
        
        # Stack multiple Zarvan blocks
        self.zarvan_blocks = nn.ModuleList([
            GatedZarvanBlock(embed_dim, hidden_dim, num_heads) 
            for _ in range(num_zarvan_blocks)
        ])
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        for block in self.zarvan_blocks:
            x = block(x)
            
        return x.mean(dim=1) # Global average pooling to get sequence representation

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, ff_dim, num_heads, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return x.mean(dim=1)

class TwoTowerModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, query, pos_doc, neg_doc):
        return self.encoder(query), self.encoder(pos_doc), self.encoder(neg_doc)

# --- 5. Data Pipeline ---
class TripletDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, seq_len):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    def __len__(self):
        return len(self.hf_dataset)
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        passages = item['passages']['passage_text']
        is_selected = item['passages']['is_selected']
        pos_passages = [p for i, p in enumerate(passages) if is_selected[i] == 1]
        neg_passages = [p for i, p in enumerate(passages) if is_selected[i] == 0]
        pos_text = random.choice(pos_passages) if pos_passages else "positive sample missing"
        neg_text = random.choice(neg_passages) if neg_passages else "negative sample missing"
        
        tokenize_fn = lambda text: self.tokenizer(
            text, padding="max_length", truncation=True, 
            max_length=self.seq_len, return_tensors='pt')['input_ids'].squeeze(0)
            
        return tokenize_fn(item['query']), tokenize_fn(pos_text), tokenize_fn(neg_text)

# --- 6. Universal Training & Evaluation Functions ---
def run_ir_epoch(is_training, model, loader, optimizer, criterion):
    model.train(is_training)
    total_loss = 0.0
    correct_rankings, total_samples = 0, 0
    
    progress_bar = tqdm(loader, desc="Training" if is_training else "Evaluating", leave=False)
    
    with torch.set_grad_enabled(is_training):
        for query, pos_doc, neg_doc in progress_bar:
            # Move data to device
            if IS_TPU:
                # For XLA, data is already on device if using MpDeviceLoader
                pass 
            else:
                device = next(model.parameters()).device # Get current model device
                query, pos_doc, neg_doc = query.to(device), pos_doc.to(device), neg_doc.to(device)

            q_vec, p_vec, n_vec = model(query, pos_doc, neg_doc)
            loss = criterion(q_vec, p_vec, n_vec)
            
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                if IS_TPU: xm.optimizer_step(optimizer)
                else: optimizer.step()
            else:
                dist_pos = 1 - F.cosine_similarity(q_vec, p_vec)
                dist_neg = 1 - F.cosine_similarity(q_vec, n_vec)
                correct_rankings += (dist_pos < dist_neg).sum().item()
                total_samples += query.size(0)

            total_loss += loss.item()

    if not is_training and IS_TPU:
        correct_rankings = xm.mesh_reduce('val_correct', correct_rankings, sum)
        total_samples = xm.mesh_reduce('val_samples', total_samples, sum)
        
    avg_loss = total_loss / len(loader)
    ranking_acc = 100. * correct_rankings / total_samples if not is_training else 0
    return avg_loss, ranking_acc

# --- 7. Main Execution Logic ---
def main():
    # --- Device and Batch Size Setup ---
    if IS_TPU:
        rank = xm.get_ordinal()
        device = xm.xla_device()
        batch_size = FLAGS['total_batch_size'] // xm.xrt_world_size()
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = FLAGS['total_batch_size']
    
    if rank == 0:
        print(f"Running on: {device} | Global Batch Size: {FLAGS['total_batch_size']}")

    # --- Data Loading & Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if rank == 0:
        print("Loading and preparing MS MARCO dataset...")
        dataset = load_dataset(FLAGS['dataset_name'], FLAGS['dataset_version'], split='train').filter(
            lambda ex: 1 in ex['passages']['is_selected'] and 0 in ex['passages']['is_selected']
        ).shuffle(seed=42).select(range(FLAGS['train_samples']))
        # Save to a common location accessible by all TPU cores
        # For Kaggle, this might be /kaggle/working/
        torch.save(dataset, 'msmarco_data.pth') 

    if IS_TPU: xm.rendezvous('download_complete') # Ensure all cores wait for download

    dataset = torch.load('msmarco_data.pth', weights_only=False)
    dataset_split = dataset.train_test_split(test_size=FLAGS['test_split_size'], seed=42)
    train_dataset = TripletDataset(dataset_split['train'], tokenizer, FLAGS['seq_len'])
    test_dataset = TripletDataset(dataset_split['test'], tokenizer, FLAGS['seq_len'])

    train_sampler = None
    if IS_TPU:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=xm.xrt_world_size(), rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=FLAGS['num_workers'])
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, # No shuffling for test loader
        num_workers=FLAGS['num_workers'])

    if IS_TPU:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        test_loader = pl.MpDeviceLoader(test_loader, device)
        
    # --- Model Training Loop ---
    models_to_test = {
        # Using GatedZarvanBlock and num_zarvan_blocks
        "GatedZarvan": TwoTowerModel(ZarvanEncoder(tokenizer.vocab_size, FLAGS['seq_len'], FLAGS['embed_dim'], FLAGS['hidden_dim'], FLAGS['num_heads'], tokenizer.pad_token_id, FLAGS['num_zarvan_blocks'])),
        "Transformer": TwoTowerModel(TransformerEncoder(tokenizer.vocab_size, FLAGS['seq_len'], FLAGS['embed_dim'], FLAGS['ff_dim'], FLAGS['num_heads'], tokenizer.pad_token_id))
    }
    
    criterion = nn.TripletMarginWithDistanceLoss(margin=FLAGS['margin'], distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
    results_summary = []

    for name, model in models_to_test.items():
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=FLAGS['learning_rate'])
        
        if rank == 0: print(f"\n🚀 Training {name} on MS MARCO...")
        start_time = time.time()
        
        for epoch in range(1, FLAGS['num_epochs'] + 1):
            train_loss, _ = run_ir_epoch(True, model, train_loader, optimizer, criterion)
            if rank == 0: print(f"Epoch {epoch}/{FLAGS['num_epochs']} -> Train Loss: {train_loss:.4f}")

        eval_loss, ranking_acc = run_ir_epoch(False, model, test_loader, None, criterion)
        total_time = time.time() - start_time
        
        if rank == 0:
            print(f"Final Results for {name}:")
            print(f"Ranking Accuracy: {ranking_acc:.2f}% | Total Time: {total_time:.2f}s")
            results_summary.append({'model': name, 'ranking_accuracy': ranking_acc, 'time': total_time})
    
    # --- Final Report ---
    if rank == 0:
        print("\n\n" + "="*50); print("FINAL INFORMATION RETRIEVAL RESULTS"); print("="*50)
        df = pd.DataFrame(results_summary)
        print(df.to_string())

# --- Universal Entry Point ---
if __name__ == '__main__':
    if IS_TPU:
        def _mp_fn_wrapper(rank, flags):
            main()
        xmp.spawn(_mp_fn_wrapper, args=(FLAGS,), nprocs=1, start_method='fork')
    else:
        main()
