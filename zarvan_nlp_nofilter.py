import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import re
from collections import Counter
import pandas as pd

# --- 1. Configuration --- #
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "embed_dim": 64,
    "hidden_dim": 128,
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "num_heads": 4,
}
SEQUENCE_LENGTHS = [128, 256, 512]
print(f"Using device: {CONFIG['device']}")

# --- 2. Data Pipeline (User's Original Style) --- #
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
                self.stoi[word] = len(self.stoi)
    def __len__(self):
        return len(self.stoi)

class IMDbDataset(Dataset):
    def __init__(self, data, vocab, max_len):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        tokens = simple_tokenizer(text)
        token_ids = [self.vocab.stoi.get(token, 0) for token in tokens]
        # Truncate if longer than max_len
        token_ids = token_ids[:self.max_len]
        return torch.tensor(token_ids), torch.tensor(label)

def create_collate_fn(pad_idx):
    def collate_batch(batch):
        label_list, text_list = [], []
        for (_text, _label) in batch:
            label_list.append(_label)
            text_list.append(_text)
        text_padded = pad_sequence(text_list, batch_first=True, padding_value=pad_idx)
        return text_padded, torch.tensor(label_list, dtype=torch.int64)
    return collate_batch

# --- 3. Model Architectures --- #

# Zarvan (no-filter) Architecture
class MultiHeadLinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        num_local, num_global = num_heads // 2, num_heads - num_heads // 2
        self.local_proj = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_local)])
        self.local_score = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_local)])
        self.global_proj = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_global)])
        self.global_score = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_global)])
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)

    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        head_outputs = []
        for proj, score_fn in zip(self.local_proj, self.local_score):
            scores = score_fn(x).squeeze(-1)
            attn_weights = F.softmax(scores, dim=0).unsqueeze(-1)
            values = proj(x)
            head_outputs.append((values * attn_weights).sum(dim=0))
        g = x.mean(dim=0)
        for proj, score_fn in zip(self.global_proj, self.global_score):
            scores = score_fn(g)
            attn_weights = F.softmax(scores, dim=0).unsqueeze(0).expand(seq_len, -1, -1)
            values = proj(x)
            head_outputs.append((values * attn_weights).sum(dim=0))
        concatenated = torch.cat(head_outputs, dim=-1)
        return self.fc_out(concatenated)

class Zarvan(nn.Module):
    def __init__(self, seq_len, embed_dim, hidden_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadLinearAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, 1, embed_dim))

    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        x = x + self.pos_encoding[:seq_len]
        residual = x
        attn_out = self.attention(x)
        x = self.norm1(residual + attn_out.unsqueeze(0))
        residual = x
        ffn_out = self.ffn(x)
        z = self.norm2(residual + ffn_out)
        return z

class ZarvanClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, hidden_dim, num_classes, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.zarvan = Zarvan(seq_len, embed_dim, hidden_dim, num_heads)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)
        z = self.zarvan(x).mean(dim=0)
        return self.fc(z)

# Transformer Baseline
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, hidden_dim, num_classes, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=0.1, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x).permute(1, 0, 2)
        x = x + self.pos_encoding[:seq_len]
        x = self.transformer(x).mean(dim=0)
        return self.fc(x)

# --- 4. Training & Evaluation Loop --- #
def run_epoch(model, loader, optimizer, criterion, device, is_training):
    if is_training: model.train()
    else: model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    desc = "Training" if is_training else "Evaluating"
    with torch.set_grad_enabled(is_training):
        for text, labels in tqdm(loader, desc=desc):
            text, labels = text.to(device), labels.to(device)
            output = model(text)
            loss = criterion(output, labels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_correct += (output.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / len(loader), 100. * total_correct / total_samples

# --- 5. Main Execution Logic --- #
print("Loading IMDb dataset and building vocabulary...")
imdb = load_dataset("imdb")
word_counter = Counter(token for item in tqdm(imdb['train'], desc="Building vocab") for token in simple_tokenizer(item['text']))
vocab = Vocab(word_counter)
print(f"Vocabulary size: {len(vocab)}")

results_summary = []
collate_fn = create_collate_fn(pad_idx=vocab.stoi['<pad>'])

for seq_len in SEQUENCE_LENGTHS:
    print(f"\n{'='*20} TESTING SEQ_LEN = {seq_len} {'='*20}")
    
    train_ds = IMDbDataset(imdb['train'], vocab, seq_len)
    test_ds = IMDbDataset(imdb['test'], vocab, seq_len)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], collate_fn=collate_fn)

    model_configs = {
        'Zarvan': ZarvanClassifier,
        'Transformer': TransformerClassifier
    }
    
    for model_name, model_class in model_configs.items():
        print(f"\n🚀 Training {model_name}...")
        
        # --- THIS IS THE FIX ---
        # Manually passing required arguments instead of unpacking the whole CONFIG dict
        model = model_class(
            vocab_size=len(vocab),
            seq_len=seq_len,
            num_classes=2,
            embed_dim=CONFIG['embed_dim'],
            hidden_dim=CONFIG['hidden_dim'],
            num_heads=CONFIG['num_heads']
        ).to(CONFIG['device'])
        # ------------------------
        
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        for epoch in range(1, CONFIG['num_epochs'] + 1):
            train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, CONFIG['device'], is_training=True)
            print(f"Epoch {epoch}/{CONFIG['num_epochs']} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
        test_loss, test_acc = run_epoch(model, test_loader, None, criterion, CONFIG['device'], is_training=False)
        total_time = time.time() - start_time
        
        print(f"Final Results for {model_name} (SeqLen {seq_len}):")
        print(f"Test Accuracy: {test_acc:.2f}% | Total Time: {total_time:.2f}s")
        
        results_summary.append({
            'seq_len': seq_len, 'model': model_name, 'accuracy': test_acc, 'time': total_time
        })

# --- 6. Final Report and Plotting --- #
print("\n\n" + "="*50)
print("FINAL SCALABILITY RESULTS (Original Data Pipeline)")
print("="*50)
df = pd.DataFrame(results_summary)
print(df.to_string())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
df_zarvan = df[df['model'] == 'Zarvan']
df_transformer = df[df['model'] == 'Transformer']

ax1.plot(df_zarvan['seq_len'], df_zarvan['accuracy'], 'b-o', label='Zarvan (no-filter) Accuracy')
ax1.plot(df_transformer['seq_len'], df_transformer['accuracy'], 'r-s', label='Transformer Accuracy')
ax1.set_title('Accuracy vs. Sequence Length')
ax1.set_xlabel('Sequence Length'); ax1.set_ylabel('Final Test Accuracy (%)')
ax1.grid(True); ax1.legend()

ax2.plot(df_zarvan['seq_len'], df_zarvan['time'], 'b-o', label='Zarvan (no-filter) Time')
ax2.plot(df_transformer['seq_len'], df_transformer['time'], 'r-s', label='Transformer Time')
ax2.set_title('Total Training Time vs. Sequence Length')
ax2.set_xlabel('Sequence Length'); ax2.set_ylabel('Time (seconds)')
ax2.grid(True); ax2.legend()

plt.tight_layout()
plt.savefig('scalability_test_original_pipeline.png')
print("\nSaved scalability plot to 'scalability_test_original_pipeline.png'")
plt.show()