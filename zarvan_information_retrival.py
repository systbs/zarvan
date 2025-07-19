# ===================================================================
#
#   MS MARCO Benchmark: Definitive Zarvan vs. Transformer
#
# This script runs a benchmark comparing the final, hybrid Zarvan
# architecture against a standard Transformer baseline on the
# MS MARCO information retrieval task using a two-tower model.
#
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

# Use try-except for environments where these might not be pre-installed
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("Installing 'datasets' and 'transformers' libraries...")
    !pip install -q datasets transformers
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
    "total_batch_size": 128,
    "num_epochs": 3,
    "learning_rate": 5e-5,
    "num_heads": 4,
    "margin": 0.5,
    "num_workers": 0, # Set to 0 to avoid multiprocessing issues
    "dataset_name": "ms_marco",
    "dataset_version": "v2.1",
    "train_samples": 20000,
    "test_split_size": 0.1,
    "num_layers": 2,
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

# --- 4. Encoders for Two-Tower Model ---
class ZarvanEncoder(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, hidden_dim, num_heads, num_layers, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        self.layers = nn.ModuleList([
            ZarvanBlock(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim=1)

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, ff_dim, num_heads, num_layers, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return x.mean(dim=1)

class TwoTowerModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, query, pos_doc, neg_doc, query_mask, pos_mask, neg_mask):
        # Pass masks to the encoder if it's a Transformer
        if isinstance(self.encoder, TransformerEncoder):
            q_vec = self.encoder(query, src_key_padding_mask=query_mask)
            p_vec = self.encoder(pos_doc, src_key_padding_mask=pos_mask)
            n_vec = self.encoder(neg_doc, src_key_padding_mask=neg_mask)
        else: # Zarvan doesn't need masks
            q_vec = self.encoder(query)
            p_vec = self.encoder(pos_doc)
            n_vec = self.encoder(neg_doc)
        return q_vec, p_vec, n_vec

# --- 5. Data Pipeline ---
class TripletDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, seq_len):
        self.hf_dataset, self.tokenizer, self.seq_len = hf_dataset, tokenizer, seq_len
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
        return item['query'], pos_text, neg_text

def collate_fn(batch, tokenizer, seq_len):
    queries, pos_docs, neg_docs = zip(*batch)
    
    tokenize_fn = lambda texts: tokenizer(
        list(texts), padding="max_length", truncation=True, 
        max_length=seq_len, return_tensors='pt'
    )
    
    q_tokens = tokenize_fn(queries)
    p_tokens = tokenize_fn(pos_docs)
    n_tokens = tokenize_fn(neg_docs)
    
    return (q_tokens['input_ids'], p_tokens['input_ids'], n_tokens['input_ids'],
            q_tokens['attention_mask']==0, p_tokens['attention_mask']==0, n_tokens['attention_mask']==0)


# --- 6. Universal Training & Evaluation Functions ---
def run_ir_epoch(is_training, model, loader, optimizer, criterion, device):
    model.train(is_training)
    total_loss, correct_rankings, total_samples = 0.0, 0, 0
    progress_bar = tqdm(loader, desc="Training" if is_training else "Evaluating", leave=False)
    
    with torch.set_grad_enabled(is_training):
        for q_ids, p_ids, n_ids, q_mask, p_mask, n_mask in progress_bar:
            q_ids, p_ids, n_ids = q_ids.to(device), p_ids.to(device), n_ids.to(device)
            q_mask, p_mask, n_mask = q_mask.to(device), p_mask.to(device), n_mask.to(device)

            q_vec, p_vec, n_vec = model(q_ids, p_ids, n_ids, q_mask, p_mask, n_mask)
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
                total_samples += q_ids.size(0)

            total_loss += loss.item() * q_ids.size(0)

    if not is_training and IS_TPU:
        correct_rankings = xm.mesh_reduce('val_correct', correct_rankings, sum)
        total_samples = xm.mesh_reduce('val_samples', total_samples, sum)
        
    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0
    ranking_acc = 100. * correct_rankings / total_samples if not is_training and total_samples > 0 else 0
    return avg_loss, ranking_acc

# --- 7. Main Execution Logic ---
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
        print(f"Running on: {device} | Global Batch Size: {FLAGS['total_batch_size']}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_file = 'msmarco_data.pth'
    if rank == 0 and not os.path.exists(data_file):
        print("Loading and preparing MS MARCO dataset...")
        dataset = load_dataset(FLAGS['dataset_name'], FLAGS['dataset_version'], split='train').filter(
            lambda ex: 1 in ex['passages']['is_selected'] and 0 in ex['passages']['is_selected']
        ).shuffle(seed=42).select(range(FLAGS['train_samples']))
        torch.save(dataset, data_file) 

    if IS_TPU: xm.rendezvous('download_complete')

    dataset = torch.load(data_file, map_location='cpu', weights_only=False)
    dataset_split = dataset.train_test_split(test_size=FLAGS['test_split_size'], seed=42)
    train_dataset = TripletDataset(dataset_split['train'], tokenizer, FLAGS['seq_len'])
    test_dataset = TripletDataset(dataset_split['test'], tokenizer, FLAGS['seq_len'])

    train_sampler = None
    if IS_TPU:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=xm.xrt_world_size(), rank=rank, shuffle=True)

    collate_with_padding = lambda batch: collate_fn(batch, tokenizer, FLAGS['seq_len'])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=FLAGS['num_workers'], collate_fn=collate_with_padding)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=FLAGS['num_workers'], collate_fn=collate_with_padding)

    if IS_TPU:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        test_loader = pl.MpDeviceLoader(test_loader, device)
        
    models_to_test = {
        "Zarvan": TwoTowerModel(ZarvanEncoder(tokenizer.vocab_size, FLAGS['seq_len'], FLAGS['embed_dim'], FLAGS['hidden_dim'], FLAGS['num_heads'], FLAGS['num_layers'], tokenizer.pad_token_id)),
        "Transformer": TwoTowerModel(TransformerEncoder(tokenizer.vocab_size, FLAGS['seq_len'], FLAGS['embed_dim'], FLAGS['ff_dim'], FLAGS['num_heads'], FLAGS['num_layers'], tokenizer.pad_token_id))
    }
    
    criterion = nn.TripletMarginWithDistanceLoss(margin=FLAGS['margin'], distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
    results_summary = []

    for name, model in models_to_test.items():
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=FLAGS['learning_rate'])
        
        if rank == 0: print(f"\n🚀 Training {name} on MS MARCO...")
        start_time = time.time()
        
        for epoch in range(1, FLAGS['num_epochs'] + 1):
            train_loss, _ = run_ir_epoch(True, model, train_loader, optimizer, criterion, device)
            if rank == 0: print(f"Epoch {epoch}/{FLAGS['num_epochs']} -> Train Loss: {train_loss:.4f}")

        eval_loss, ranking_acc = run_ir_epoch(False, model, test_loader, None, criterion, device)
        total_time = time.time() - start_time
        
        if rank == 0:
            print(f"Final Results for {name}:")
            print(f"Ranking Accuracy: {ranking_acc:.2f}% | Total Time: {total_time:.2f}s")
            results_summary.append({'model': name, 'ranking_accuracy': ranking_acc, 'time': total_time})
    
    if rank == 0:
        print("\n\n" + "="*50); print("FINAL INFORMATION RETRIEVAL RESULTS"); print("="*50)
        df = pd.DataFrame(results_summary)
        print(df.to_string())

# --- Universal Entry Point ---
if __name__ == '__main__':
    if IS_TPU:
        def _mp_fn_wrapper(rank, flags):
            main()
        xmp.spawn(_mp_fn_wrapper, args=(FLAGS,), nprocs=None, start_method='fork')
    else:
        main()
