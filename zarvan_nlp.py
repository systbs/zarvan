import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import time
import re

# --- 1. تنظیمات و پارامترها ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# پارامترهای مدل
EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_HEADS = 4
MAX_LEN = 256  # حداکثر طول دنباله برای تحلیل
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

# --- 2. آماده‌سازی داده‌ها و واژگان ---

def simple_tokenizer(text):
    text = re.sub(r'<[^>]+>', ' ', text) # حذف تگ‌های HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower() # حذف کاراکترهای غیرحرفی
    return text.split()

class Vocab:
    def __init__(self, counter, min_freq=5):
        self.stoi = {'<unk>': 0, '<pad>': 1}
        self.itos = {0: '<unk>', 1: '<pad>'}
        for word, count in counter.items():
            if count >= min_freq:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx] = word
    
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
        token_ids = [self.vocab.stoi.get(token, self.vocab.stoi['<unk>']) for token in tokens]
        # کوتاه کردن یا پدینگ
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        
        return torch.tensor(token_ids), torch.tensor(label)

def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        text_list.append(_text)
    
    # پدینگ توالی‌ها
    text_list_padded = pad_sequence(text_list, batch_first=True, padding_value=1) # 1 is pad_token_id
    labels = torch.tensor(label_list, dtype=torch.int64)
    return text_list_padded, labels

# بارگیری دیتاست و ساخت واژگان
print("Loading IMDb dataset and building vocabulary...")
imdb = load_dataset("imdb")
train_data = imdb['train']
test_data = imdb['test']

word_counter = Counter()
for item in tqdm(train_data, desc="Building vocab"):
    word_counter.update(simple_tokenizer(item['text']))

vocab = Vocab(word_counter, min_freq=10)
VOCAB_SIZE = len(vocab)
print(f"Vocabulary size: {VOCAB_SIZE}")

# ساخت دیتاست و دیتا لودر
train_dataset = IMDbDataset(train_data, vocab, MAX_LEN)
test_dataset = IMDbDataset(test_data, vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)


# --- 3. تعریف مدل‌ها (Zarvan و Transformer) ---

# تعریف مجدد Zarvan (بدون تغییر در منطق اصلی)
class MultiHeadLinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(MultiHeadLinearAttention, self).__init__()
        self.num_heads = num_heads
        self.local_query_nets = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_heads // 2)])
        self.global_query_nets = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_heads - num_heads // 2)])
        self.combine = nn.Linear(embed_dim * num_heads, embed_dim)
    
    def forward(self, x):
        heads = []
        # توجه محلی
        for query_net in self.local_query_nets:
            attention_scores = query_net(x)
            attention_weights = F.softmax(attention_scores, dim=0)
            head = (x * attention_weights).sum(dim=0)
            heads.append(head)
        # توجه کلان
        global_mean = x.mean(dim=0, keepdim=True).expand_as(x)
        for query_net in self.global_query_nets:
            attention_scores = query_net(global_mean)
            attention_weights = F.softmax(attention_scores, dim=0)
            head = (x * attention_weights).sum(dim=0)
            heads.append(head)
        combined = self.combine(torch.cat(heads, dim=-1))
        return combined

class Zarvan(nn.Module):
    def __init__(self, seq_len, embed_dim, hidden_dim=64, num_heads=4):
        super(Zarvan, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.attention = MultiHeadLinearAttention(embed_dim, num_heads)
        self.filter_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, 1, embed_dim))
        self.interactive_context = nn.Linear(embed_dim * 2, embed_dim)
        self.global_context = nn.Linear(embed_dim, embed_dim)
        self.Linear_xw = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size = x.size(1)
        pos_encoding = self.pos_encoding[:x.size(0)].expand(-1, batch_size, -1)
        x = x + pos_encoding
        q = self.attention(x)
        q_expanded = q.unsqueeze(0).repeat(x.size(0), 1, 1)
        global_ctx = self.global_context(x.mean(dim=0))
        interactive_ctx = torch.cat([q, global_ctx], dim=-1)
        interactive_ctx = F.relu(self.interactive_context(interactive_ctx))
        interactive_ctx_expanded = interactive_ctx.unsqueeze(0).repeat(x.size(0), 1, 1)
        inputs = torch.cat([x, q_expanded, interactive_ctx_expanded], dim=-1)
        filter_weights = self.filter_net(inputs)
        filter_weights = filter_weights - filter_weights.mean(dim=0, keepdim=True)
        weighted_x = x * filter_weights
        z = x + self.Linear_xw(weighted_x)
        z = self.norm(z)
        return z

# مدل طبقه‌بند Zarvan برای NLP
class ZarvanClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_classes=2):
        super(ZarvanClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.zarvan = Zarvan(seq_len, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len) -> (batch, seq_len, embed_dim)
        x = x.permute(1, 0, 2) # (seq_len, batch, embed_dim)
        z = self.zarvan(x)
        z = z.mean(dim=0) # Global Average Pooling
        out = self.fc(z)
        return out

# مدل طبقه‌بند Transformer برای NLP
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_classes=2, num_heads=4, ff_dim=128):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(1, 0, 2) # (seq_len, batch, embed_dim)
        pos_encoding = self.pos_encoding[:seq_len].expand(-1, batch_size, -1)
        x = x + pos_encoding
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=0) # Global Average Pooling
        out = self.fc(x)
        return out


# --- 4. حلقه آموزش و ارزیابی ---

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    start_time = time.time()
    
    for text, labels in tqdm(loader, desc="Training"):
        text, labels = text.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_correct += (output.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)
        
    end_time = time.time()
    epoch_time = end_time - start_time
    return total_loss / len(loader), 100. * total_correct / total_samples, epoch_time

def test(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for text, labels in tqdm(loader, desc="Evaluating"):
            text, labels = text.to(device), labels.to(device)
            output = model(text)
            loss = criterion(output, labels)
            total_loss += loss.item()
            total_correct += (output.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
            
    return total_loss / len(loader), 100. * total_correct / total_samples

# --- 5. اجرای آزمایش ---

# مقداردهی مدل‌ها
zarvan_model = ZarvanClassifier(VOCAB_SIZE, MAX_LEN, EMBED_DIM).to(DEVICE)
transformer_model = TransformerClassifier(VOCAB_SIZE, MAX_LEN, EMBED_DIM).to(DEVICE)

zarvan_optimizer = optim.AdamW(zarvan_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
transformer_optimizer = optim.AdamW(transformer_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# ذخیره معیارها
history = {
    'zarvan_train_loss': [], 'zarvan_train_acc': [], 'zarvan_test_loss': [], 'zarvan_test_acc': [], 'zarvan_time': [],
    'transformer_train_loss': [], 'transformer_train_acc': [], 'transformer_test_loss': [], 'transformer_test_acc': [], 'transformer_time': []
}

for epoch in range(1, EPOCHS + 1):
    print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
    
    # آموزش و ارزیابی Zarvan
    print("--- Training Zarvan ---")
    z_train_loss, z_train_acc, z_time = train(zarvan_model, train_loader, zarvan_optimizer, criterion, DEVICE)
    z_test_loss, z_test_acc = test(zarvan_model, test_loader, criterion, DEVICE)
    history['zarvan_train_loss'].append(z_train_loss)
    history['zarvan_train_acc'].append(z_train_acc)
    history['zarvan_test_loss'].append(z_test_loss)
    history['zarvan_test_acc'].append(z_test_acc)
    history['zarvan_time'].append(z_time)
    
    # آموزش و ارزیابی Transformer
    print("--- Training Transformer ---")
    t_train_loss, t_train_acc, t_time = train(transformer_model, train_loader, transformer_optimizer, criterion, DEVICE)
    t_test_loss, t_test_acc = test(transformer_model, test_loader, criterion, DEVICE)
    history['transformer_train_loss'].append(t_train_loss)
    history['transformer_train_acc'].append(t_train_acc)
    history['transformer_test_loss'].append(t_test_loss)
    history['transformer_test_acc'].append(t_test_acc)
    history['transformer_time'].append(t_time)
    
    print(f"\nEpoch {epoch} Results:")
    print(f"Zarvan      -> Train Loss: {z_train_loss:.4f}, Train Acc: {z_train_acc:.2f}%, Test Acc: {z_test_acc:.2f}%, Time: {z_time:.2f}s")
    print(f"Transformer -> Train Loss: {t_train_loss:.4f}, Train Acc: {t_train_acc:.2f}%, Test Acc: {t_test_acc:.2f}%, Time: {t_time:.2f}s")

# --- 6. رسم نمودارها ---
epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(18, 6))

# نمودار دقت
plt.subplot(1, 3, 1)
plt.plot(epochs_range, history['zarvan_test_acc'], 'bo-', label='Zarvan Test Accuracy')
plt.plot(epochs_range, history['transformer_test_acc'], 'rs-', label='Transformer Test Accuracy')
plt.title('Accuracy Comparison (Test)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# نمودار خطا
plt.subplot(1, 3, 2)
plt.plot(epochs_range, history['zarvan_test_loss'], 'bo-', label='Zarvan Test Loss')
plt.plot(epochs_range, history['transformer_test_loss'], 'rs-', label='Transformer Test Loss')
plt.title('Loss Comparison (Test)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# نمودار زمان
plt.subplot(1, 3, 3)
plt.plot(epochs_range, history['zarvan_time'], 'bo-', label='Zarvan Train Time')
plt.plot(epochs_range, history['transformer_time'], 'rs-', label='Transformer Train Time')
plt.title('Epoch Time Comparison')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('nlp_comparison_zarvan_vs_transformer.png')
plt.show()
print("Comparison chart saved as 'nlp_comparison_zarvan_vs_transformer.png'")