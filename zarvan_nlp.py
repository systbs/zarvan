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
import math

# --- 1. Configuration --- #
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "embed_dim": 64,
    "hidden_dim": 64,
    "ff_dim": 128,
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "num_heads": 4,
}
SEQUENCE_LENGTHS = [128, 256, 512, 1024]
print(f"Using device: {CONFIG['device']}")

# --- 2. Building Blocks --- #
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class ParallelUnifiedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.query_net = nn.Linear(embed_dim, embed_dim * num_heads)
        self.combine = nn.Linear(embed_dim * num_heads, embed_dim)
    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        scores_all_heads = self.query_net(x).view(seq_len, batch_size * self.num_heads, self.embed_dim)
        attn_weights = F.softmax(scores_all_heads, dim=0)
        x_repeated = x.repeat_interleave(self.num_heads, dim=1)
        weighted_values = x_repeated * attn_weights
        head_outputs = weighted_values.sum(dim=0)
        concatenated_heads = head_outputs.view(batch_size, self.num_heads * self.embed_dim)
        return self.combine(concatenated_heads)

# --- 3. Model Architectures --- #
class FinalZarvanBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        self.attention = ParallelUnifiedAttention(embed_dim, num_heads)
        self.interactive_context = nn.Linear(embed_dim, embed_dim)
        self.filter_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(embed_dim)
        self.Linear_xw = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        q = self.attention(x)
        i_ctx = F.relu(self.interactive_context(q))
        q_exp = q.unsqueeze(0).expand(x.size(0), -1, -1)
        i_ctx_exp = i_ctx.unsqueeze(0).expand(x.size(0), -1, -1)
        filter_in = torch.cat([x, q_exp, i_ctx_exp], dim=-1)
        f_weights = self.filter_net(filter_in) - self.filter_net(filter_in).mean(dim=0, keepdim=True)
        z = x + self.Linear_xw(x * f_weights)
        return self.norm(z)

class ZarvanClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, hidden_dim, num_classes, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=seq_len)
        self.zarvan = FinalZarvanBlock(embed_dim, hidden_dim, num_heads)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)
        x = self.pos_encoder(x)
        z = self.zarvan(x).mean(dim=0)
        return self.fc(z)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, ff_dim, num_classes, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)
        x = self.pos_encoder(x)
        return self.fc(self.transformer(x).mean(dim=0))

# --- 4. Data Pipeline (with fixed padding) --- #
def simple_tokenizer(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return text.split()

class Vocab:
    def __init__(self, counter, min_freq=10):
        self.stoi = {'<unk>': 0, '<pad>': 1}
        for word, count in counter.items():
            if count >= min_freq: self.stoi[word] = len(self.stoi)
    def __len__(self): return len(self.stoi)

class IMDbDataset(Dataset):
    def __init__(self, data, vocab, max_len):
        self.data, self.vocab, self.max_len = data, vocab, max_len
        self.pad_idx = vocab.stoi['<pad>']
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = simple_tokenizer(item['text'])
        token_ids = [self.vocab.stoi.get(t, 0) for t in tokens]
        token_ids = token_ids[:self.max_len]
        padding_needed = self.max_len - len(token_ids)
        token_ids.extend([self.pad_idx] * padding_needed)
        return torch.tensor(token_ids), torch.tensor(item['label'])

def collate_fn(batch):
    texts, labels = zip(*batch) 
    return torch.stack(texts), torch.stack(labels)

# --- 5. Training & Evaluation Loop --- #
def run_epoch(model, loader, optimizer, criterion, device, is_training):
    model.train(is_training)
    with torch.set_grad_enabled(is_training):
        for text, labels in tqdm(loader, desc="Training" if is_training else "Evaluating"):
            text, labels = text.to(device), labels.to(device)
            output = model(text)
            if is_training:
                loss = criterion(output, labels)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for text, labels in tqdm(loader, desc="Evaluating"):
            text, labels = text.to(device), labels.to(device)
            predicted = model(text).argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# --- 6. Main Execution Logic --- #
print("Loading IMDb dataset and building vocabulary...")
imdb = load_dataset("imdb")
word_counter = Counter(t for item in tqdm(imdb['train'], desc="Building vocab") for t in simple_tokenizer(item['text']))
vocab = Vocab(word_counter)
print(f"Vocabulary size: {len(vocab)}")

results_summary = []
# NOTE: The collate_fn is not strictly needed anymore because padding is done in __getitem__, 
# but we keep it for consistency.
collate_fn = collate_fn 

for seq_len in SEQUENCE_LENGTHS:
    print(f"\n{'='*20} TESTING SEQ_LEN = {seq_len} {'='*20}")
    train_loader = DataLoader(IMDbDataset(imdb['train'], vocab, seq_len), batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(IMDbDataset(imdb['test'], vocab, seq_len), batch_size=CONFIG['batch_size'], collate_fn=collate_fn)

    # ===================================================================
    # FIX: Manually passing required arguments to prevent the TypeError.
    # ===================================================================
    models_to_test = {
        'Zarvan': ZarvanClassifier(
            vocab_size=len(vocab),
            seq_len=seq_len,
            num_classes=2,
            embed_dim=CONFIG['embed_dim'],
            hidden_dim=CONFIG['hidden_dim'],
            num_heads=CONFIG['num_heads']
        ),
        'Transformer': TransformerClassifier(
            vocab_size=len(vocab),
            seq_len=seq_len,
            num_classes=2,
            embed_dim=CONFIG['embed_dim'],
            ff_dim=CONFIG['ff_dim'],
            num_heads=CONFIG['num_heads']
        )
    }
    # ===================================================================
    
    for name, model in models_to_test.items():
        print(f"\n🚀 Training {name}...")
        model.to(CONFIG['device'])
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        start_time = time.time()
        for epoch in range(1, CONFIG['num_epochs'] + 1):
            print(f"Epoch {epoch}/{CONFIG['num_epochs']}")
            run_epoch(model, train_loader, optimizer, criterion, CONFIG['device'], is_training=True)
        eval_acc = evaluate(model, test_loader, CONFIG['device'])
        total_time = time.time() - start_time
        print(f"Final Accuracy for {name}: {eval_acc:.2f}% | Total Time: {total_time:.2f}s")
        results_summary.append({'seq_len': seq_len, 'model': name, 'accuracy': eval_acc, 'time': total_time})
        
# --- 7. Final Report and Plotting --- #
print("\n\n" + "="*50); print("FINAL SCALABILITY RESULTS"); print("="*50)
df = pd.DataFrame(results_summary)
print(df.to_string())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
for model_name, marker, color in [('Zarvan', 'o', 'blue'), ('Transformer', 's', 'red')]:
    model_df = df[df['model'] == model_name]
    ax1.plot(model_df['seq_len'], model_df['accuracy'], marker=marker, color=color, label=f'{model_name} Accuracy')
    ax2.plot(model_df['seq_len'], model_df['time'], marker=marker, color=color, label=f'{model_name} Time')
ax1.set_title('Accuracy vs. Sequence Length'); ax1.set_xlabel('Sequence Length'); ax1.set_ylabel('Final Test Accuracy (%)')
ax2.set_title('Training Time vs. Sequence Length'); ax2.set_xlabel('Sequence Length'); ax2.set_ylabel('Time (seconds)')
for ax in [ax1, ax2]: ax.grid(True); ax.legend()
plt.tight_layout()
plt.savefig('final_scalability_test_zarvan_vs_transformer.png')
print("\nSaved plot to 'final_scalability_test_zarvan_vs_transformer.png'")
plt.show()