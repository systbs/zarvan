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
import matplotlib.pyplot as plt
import time
import math
from tqdm.auto import tqdm
import os
import re
from collections import Counter
import pandas as pd
# Use try-except for environments where datasets might not be pre-installed
try:
    from datasets import load_dataset
except ImportError:
    print("Installing 'datasets' library...")
    !pip install -q datasets
    from datasets import load_dataset


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
    "embed_dim": 64,
    "hidden_dim": 128,
    "ff_dim": 256,
    "total_batch_size": 128,
    "num_epochs": 5,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "num_heads": 4,
    # ✅ FIX: Set num_workers to 0 to avoid multiprocessing issues in notebook environments.
    # This is the most robust solution for the "can only test a child process" error.
    "num_workers": 0,
    "sequence_lengths": [128, 256, 512],
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
        weights = F.softmax(s, dim=-1)
        head_outputs = (weights * v).sum(dim=2)
        concatenated_heads = head_outputs.reshape(B, self.embed_dim)
        return self.combine(concatenated_heads)

class ZarvanBlock(nn.Module):
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
        ffn_input = gated_x + i_ctx_exp + q_exp
        output = self.ffn(ffn_input)
        return self.norm(x + output)

# --- 4. Classifier Models for IMDb ---
class ZarvanIMDBClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, hidden_dim, num_classes, num_heads, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=seq_len)
        self.zarvan_block = ZarvanBlock(embed_dim, hidden_dim, num_heads)
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.zarvan_block(x)
        z = x.mean(dim=1)
        return self.fc(z)

class TransformerIMDBClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, ff_dim, num_classes, num_heads, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        # Pass the padding mask to the transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        z = x.mean(dim=1)
        return self.fc(z)

# --- 5. Data Pipeline ---
def simple_tokenizer(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return text.split()

class Vocab:
    def __init__(self, counter, min_freq=10):
        self.stoi = {'<unk>': 0, '<pad>': 1}
        self.itos = {0: '<unk>', 1: '<pad>'}
        for word, count in counter.items():
            if count >= min_freq:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx] = word
    def __len__(self): return len(self.stoi)

class IMDbDataset(Dataset):
    def __init__(self, data, vocab, max_len):
        self.data, self.vocab, self.max_len = data, vocab, max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = simple_tokenizer(item['text'])
        token_ids = [self.vocab.stoi.get(t, 0) for t in tokens][:self.max_len]
        return torch.tensor(token_ids), torch.tensor(item['label'])

def collate_fn(batch, pad_idx):
    texts, labels = zip(*batch)
    texts_padded = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_idx)
    # Create the padding mask where True indicates a padded element
    padding_mask = (texts_padded == pad_idx)
    return texts_padded, torch.tensor(labels), padding_mask

# --- 6. Universal Training & Evaluation Functions ---
def run_epoch(is_training, model_name, model, loader, optimizer, criterion, device):
    model.train(is_training)
    total_loss, total_correct, total_samples = 0, 0, 0
    progress_bar = tqdm(loader, desc=f"{model_name} Training" if is_training else f"{model_name} Evaluating", leave=False)
    
    with torch.set_grad_enabled(is_training):
        for data, target, mask in progress_bar:
            data, target, mask = data.to(device), target.to(device), mask.to(device)
            
            if model_name == "Transformer":
                output = model(data, src_key_padding_mask=mask)
            else:
                output = model(data)
                
            loss = criterion(output, target)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                if IS_TPU: xm.optimizer_step(optimizer)
                else: optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            total_correct += (output.argmax(1) == target).sum().item()
            total_samples += target.size(0)

    if IS_TPU and not is_training:
        total_correct = xm.mesh_reduce('val_correct', total_correct, sum)
        total_samples = xm.mesh_reduce('val_samples', total_samples, sum)

    avg_loss = total_loss / total_samples
    accuracy = 100. * total_correct / total_samples
    return avg_loss, accuracy

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
        print(f"Running on: {str(device).upper()} | Global Batch Size: {FLAGS['total_batch_size']}")

    data_file = 'imdb_data.pth'
    if rank == 0 and not os.path.exists(data_file):
        print("Loading IMDb dataset and building vocabulary...")
        imdb = load_dataset("imdb")
        word_counter = Counter(t for item in tqdm(imdb['train'], desc="Building vocab") for t in simple_tokenizer(item['text']))
        vocab = Vocab(word_counter)
        torch.save((imdb, vocab), data_file)
    
    if IS_TPU: xm.rendezvous('download_complete')

    imdb, vocab = torch.load(data_file, map_location='cpu', weights_only=False)
    if rank == 0: print(f"Vocabulary size: {len(vocab)}")
    
    results_summary = []
    criterion = nn.CrossEntropyLoss()

    for seq_len in FLAGS['sequence_lengths']:
        if rank == 0:
            print(f"\n{'='*20} TESTING SEQ_LEN = {seq_len} {'='*20}")
        
        train_dataset = IMDbDataset(imdb['train'], vocab, seq_len)
        test_dataset = IMDbDataset(imdb['test'], vocab, seq_len)
        
        train_sampler = None
        if IS_TPU:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=xm.xrt_world_size(), rank=rank, shuffle=True)

        collate_with_padding = lambda batch: collate_fn(batch, vocab.stoi['<pad>'])
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            shuffle=(train_sampler is None), num_workers=FLAGS['num_workers'], collate_fn=collate_with_padding, pin_memory=not IS_TPU)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=FLAGS['num_workers'], collate_fn=collate_with_padding, pin_memory=not IS_TPU)

        if IS_TPU:
            train_loader = pl.MpDeviceLoader(train_loader, device)
            test_loader = pl.MpDeviceLoader(test_loader, device)
        
        models_to_test = {
            "Zarvan": ZarvanIMDBClassifier(len(vocab), seq_len, FLAGS['embed_dim'], FLAGS['hidden_dim'], 2, FLAGS['num_heads'], vocab.stoi['<pad>']),
            "Transformer": TransformerIMDBClassifier(len(vocab), seq_len, FLAGS['embed_dim'], FLAGS['ff_dim'], 2, FLAGS['num_heads'], vocab.stoi['<pad>'])
        }

        for name, model in models_to_test.items():
            model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
            
            if rank == 0: print(f"\n🚀 Training {name}...")
            start_time = time.time()
            
            for epoch in range(1, FLAGS['num_epochs'] + 1):
                train_loss, train_acc = run_epoch(True, name, model, train_loader, optimizer, criterion, device)
                if rank == 0: 
                    print(f"Epoch {epoch}/{FLAGS['num_epochs']} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

            eval_loss, eval_acc = run_epoch(False, name, model, test_loader, None, criterion, device)
            total_time = time.time() - start_time
            
            if rank == 0:
                print(f"Final Accuracy for {name} (S={seq_len}): {eval_acc:.2f}% | Total Time: {total_time:.2f}s")
                results_summary.append({'seq_len': seq_len, 'model': name, 'accuracy': eval_acc, 'time': total_time})

    if rank == 0:
        print("\n\n" + "="*50); print("FINAL SCALABILITY RESULTS"); print("="*50)
        df = pd.DataFrame(results_summary)
        print(df.to_string())

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f'Zarvan vs. Transformer on IMDb ({str(device).upper()})', fontsize=16)
        
        for model_name, marker, color in [('Zarvan', 'o', 'blue'), ('Transformer', 's', 'red')]:
            model_df = df[df['model'] == model_name]
            ax1.plot(model_df['seq_len'], model_df['accuracy'], marker=marker, color=color, label=f'{model_name} Accuracy')
            ax2.plot(model_df['seq_len'], model_df['time'], marker=marker, color=color, label=f'{model_name} Time')
        
        ax1.set_title('Accuracy vs. Sequence Length'); ax1.set_xlabel('Sequence Length'); ax1.set_ylabel('Final Test Accuracy (%)')
        ax2.set_title('Training Time vs. Sequence Length'); ax2.set_xlabel('Sequence Length'); ax2.set_ylabel('Time (seconds)')
        for ax in [ax1, ax2]: ax.grid(True); ax.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('scalability_universal_results.png')
        print("\nSaved plot to 'scalability_universal_results.png'")

# --- Universal Entry Point ---
if __name__ == '__main__':
    if IS_TPU:
        def _mp_fn_wrapper(rank, flags):
            main()
        xmp.spawn(_mp_fn_wrapper, args=(FLAGS,), nprocs=None, start_method='fork')
    else:
        main()
