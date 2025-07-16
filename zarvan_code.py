import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# تعریف شبکه Zarvan با بهینه‌سازی مفهومی
class MultiHeadLinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(MultiHeadLinearAttention, self).__init__()
        self.num_heads = num_heads
        self.local_query_nets = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_heads // 2)])
        self.global_query_nets = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_heads - num_heads // 2)])
        self.combine = nn.Linear(embed_dim * num_heads, embed_dim)
    
    def forward(self, x):
        heads = []
        # توجه محلی (پنجره کوچک)
        for query_net in self.local_query_nets:
            attention_scores = query_net(x)
            attention_weights = F.softmax(attention_scores, dim=0)
            head = (x * attention_weights).sum(dim=0)
            heads.append(head)
        # توجه کلان (میانگین‌گیری)
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
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, 1, embed_dim))
        self.interactive_context = nn.Linear(embed_dim * 2, embed_dim)
        self.global_context = nn.Linear(embed_dim, embed_dim)
        # لایه خطی برای ترکیب بهبودیافته
        self.Linear_xw = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size = x.size(1)
        pos_encoding = self.pos_encoding.expand(-1, batch_size, -1)
        x = x + pos_encoding
        q = self.attention(x)
        q_expanded = q.unsqueeze(0).repeat(x.size(0), 1, 1)
        # نمایه تعاملی
        global_ctx = self.global_context(x.mean(dim=0))
        interactive_ctx = torch.cat([q, global_ctx], dim=-1)
        interactive_ctx = F.relu(self.interactive_context(interactive_ctx))
        interactive_ctx_expanded = interactive_ctx.unsqueeze(0).repeat(x.size(0), 1, 1)
        # ترکیب ویژگی‌ها
        inputs = torch.cat([x, q_expanded, interactive_ctx_expanded], dim=-1)
        filter_weights = self.filter_net(inputs)
        filter_weights = filter_weights - filter_weights.mean(dim=0, keepdim=True)
        # ترکیب خطی بهبودیافته
        weighted_x = x * filter_weights
        z = x + self.Linear_xw(weighted_x)  # استفاده از لایه خطی برای تبدیل
        z = self.norm(z)
        return z

class ZarvanClassifier(nn.Module):
    def __init__(self, seq_len, embed_dim, num_classes=10):
        super(ZarvanClassifier, self).__init__()
        self.embedding = nn.Linear(1, embed_dim)
        self.zarvan = Zarvan(seq_len, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 1)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        z = self.zarvan(x)
        z = z.mean(dim=0)
        out = self.fc(z)
        return out

# تعریف مدل ترنسفورمر ساده
class TransformerClassifier(nn.Module):
    def __init__(self, seq_len, embed_dim, num_classes=10, num_heads=4, ff_dim=128):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(1, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 1)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = x + self.pos_encoding.expand(-1, batch_size, -1)
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=0)
        out = self.fc(x)
        return out

# آماده‌سازی داده‌ها
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# تنظیمات مدل و آموزش
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# مدل Zarvan
zarvan_model = ZarvanClassifier(seq_len=28*28, embed_dim=64, num_classes=10).to(device)
zarvan_optimizer = optim.AdamW(zarvan_model.parameters(), lr=1e-3, weight_decay=1e-2)
zarvan_scheduler = optim.lr_scheduler.CosineAnnealingLR(zarvan_optimizer, T_max=10)

# مدل ترنسفورمر
transformer_model = TransformerClassifier(seq_len=28*28, embed_dim=64, num_classes=10).to(device)
transformer_optimizer = optim.AdamW(transformer_model.parameters(), lr=1e-3, weight_decay=1e-2)
transformer_scheduler = optim.lr_scheduler.CosineAnnealingLR(transformer_optimizer, T_max=10)

criterion = nn.CrossEntropyLoss()

# تابع آموزش
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(train_loader.dataset)
    return total_loss / len(train_loader), accuracy

# تابع ارزیابی
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# ذخیره معیارها
zarvan_train_losses, zarvan_train_accs = [], []
zarvan_test_losses, zarvan_test_accs = [], []
transformer_train_losses, transformer_train_accs = [], []
transformer_test_losses, transformer_test_accs = [], []

# حلقه اصلی آموزش
epochs = 10
for epoch in range(1, epochs + 1):
    # آموزش و ارزیابی Zarvan
    zarvan_train_loss, zarvan_train_acc = train(zarvan_model, train_loader, zarvan_optimizer, criterion, device)
    zarvan_test_loss, zarvan_test_acc = test(zarvan_model, test_loader, criterion, device)
    zarvan_scheduler.step()
    
    # آموزش و ارزیابی ترنسفورمر
    transformer_train_loss, transformer_train_acc = train(transformer_model, train_loader, transformer_optimizer, criterion, device)
    transformer_test_loss, transformer_test_acc = test(transformer_model, test_loader, criterion, device)
    transformer_scheduler.step()
    
    # ذخیره معیارها
    zarvan_train_losses.append(zarvan_train_loss)
    zarvan_train_accs.append(zarvan_train_acc)
    zarvan_test_losses.append(zarvan_test_loss)
    zarvan_test_accs.append(zarvan_test_acc)
    transformer_train_losses.append(transformer_train_loss)
    transformer_train_accs.append(transformer_train_acc)
    transformer_test_losses.append(transformer_test_loss)
    transformer_test_accs.append(transformer_test_acc)
    
    # چاپ نتایج
    print(f'Epoch {epoch}:')
    print(f'Zarvan - Train Loss: {zarvan_train_loss:.4f}, Train Accuracy: {zarvan_train_acc:.2f}%')
    print(f'Zarvan - Test Loss: {zarvan_test_loss:.4f}, Test Accuracy: {zarvan_test_acc:.2f}%')
    print(f'Transformer - Train Loss: {transformer_train_loss:.4f}, Train Accuracy: {transformer_train_acc:.2f}%')
    print(f'Transformer - Test Loss: {transformer_test_loss:.4f}, Test Accuracy: {transformer_test_acc:.2f}%')
    print('-' * 50)

# رسم و ذخیره نمودارها
plt.figure(figsize=(12, 5))

# نمودار دقت
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), zarvan_train_accs, label='Zarvan Train Accuracy', marker='o')
plt.plot(range(1, epochs + 1), zarvan_test_accs, label='Zarvan Test Accuracy', marker='o')
plt.plot(range(1, epochs + 1), transformer_train_accs, label='Transformer Train Accuracy', marker='s')
plt.plot(range(1, epochs + 1), transformer_test_accs, label='Transformer Test Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison: Zarvan vs Transformer')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_comparison_zarvan_vs_transformer_linear.png')

# نمودار خطا
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), zarvan_train_losses, label='Zarvan Train Loss', marker='o')
plt.plot(range(1, epochs + 1), zarvan_test_losses, label='Zarvan Test Loss', marker='o')
plt.plot(range(1, epochs + 1), transformer_train_losses, label='Transformer Train Loss', marker='s')
plt.plot(range(1, epochs + 1), transformer_test_losses, label='Transformer Test Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Comparison: Zarvan vs Transformer')
plt.legend()
plt.grid(True)
plt.savefig('loss_comparison_zarvan_vs_transformer_linear.png')

plt.tight_layout()
plt.show()