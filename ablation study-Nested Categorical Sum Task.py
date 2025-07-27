import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import time
from tqdm.auto import tqdm
import copy 
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Configuration ---
FLAGS = {
    "seq_len": 400, # Increased sequence length
    "max_val_for_sum": 7, # Input numbers will be 0 to 6 (values 0-6)
    "num_categories": 3,  # A, B, C (for conditional rules)
    "embed_dim": 128,
    "hidden_dim": 256,
    "ff_dim": 256, # Feed-forward dim for Transformer
    "batch_size": 32, # Adjusted batch size for increased seq_len
    "num_epochs": 20, # Increased epochs
    "learning_rate": 1e-4,
    "num_heads": 4,
    "num_layers": 2, 
    "train_samples": 20000, # Increased samples
    "val_samples": 4000 
}

# ============================================================================
# Part 2: Core Context Kernels (Definitive Version - Modified for Ablation in ZarvanBlock)
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Injects sinusoidal positional information into the input sequence.
    """
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
    """
    Extracts a holistic context vector from the entire sequence.
    """
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
    """
    Extracts an associative context vector by learning importance scores.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.importance_scorer = nn.Sequential(nn.Linear(embed_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.importance_scorer(x)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

class ZarvanBlock(nn.Module):
    """
    The definitive, hybrid Zarvan Block. Modified to allow ablation.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int,
                 use_holistic: bool = True, use_associative: bool = True,
                 use_gating: bool = True):
        super().__init__()
        self.use_holistic = use_holistic
        self.use_associative = use_associative
        self.use_gating = use_gating

        # --- Dual Context Kernels ---
        if self.use_holistic:
            self.holistic_ctx = HolisticContextExtractor(embed_dim, num_heads)
        if self.use_associative:
            self.associative_ctx = AssociativeContextExtractor(embed_dim)
        
        # Determine input dimension for gate_net/simple_combiner
        gate_input_dim = embed_dim # For x
        if self.use_holistic:
            gate_input_dim += embed_dim
        if self.use_associative:
            gate_input_dim += embed_dim

        # --- Gating Mechanism or Simple Combiner ---
        if self.use_gating:
            self.gate_net = nn.Sequential(
                nn.Linear(gate_input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim * 2)
            )
            self.update_proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.simple_combiner = nn.Linear(gate_input_dim, embed_dim) # Used when gating is off
        
        # --- Final Processing ---
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        
        # 1. Extract dual context vectors in parallel (or skip if ablated)
        q_holistic_exp = torch.zeros(B, S, E, device=x.device)
        q_associative_exp = torch.zeros(B, S, E, device=x.device)

        if self.use_holistic:
            q_holistic = self.holistic_ctx(x)
            q_holistic_exp = q_holistic.unsqueeze(1).expand(-1, S, -1)
        
        if self.use_associative:
            q_associative = self.associative_ctx(x)
            q_associative_exp = q_associative.unsqueeze(1).expand(-1, S, -1)
        
        # 2. Prepare the input for the gate mechanism
        gate_input_components = [x]
        if self.use_holistic:
            gate_input_components.append(q_holistic_exp)
        if self.use_associative:
            gate_input_components.append(q_associative_exp)
            
        gate_input = torch.cat(gate_input_components, dim=-1)
        
        # 3. Compute gates and apply the update mechanism (or simplify if ablated)
        if self.use_gating:
            input_gate, forget_gate = self.gate_net(gate_input).chunk(2, dim=-1)
            gated_x = torch.sigmoid(input_gate) * x + torch.sigmoid(forget_gate) * self.update_proj(x)
        else:
            gated_x = self.simple_combiner(gate_input)
            
        # 4. Final processing and residual connection
        output = self.ffn(gated_x)
        return self.norm(x + output)

# ============================================================================
# Part 3: The Complete Model for Categorical Sum Task (Zarvan & Transformer)
# ============================================================================

class ZarvanForCategoricalSum(nn.Module):
    """
    Complete Zarvan model optimized for Categorical Sum task.
    Uses 'last' token output for classification of sum.
    """
    def __init__(self, vocab_size, num_classes, seq_len, embed_dim, hidden_dim, num_heads, num_layers, padding_idx=0,
                 use_holistic: bool = True, use_associative: bool = True, use_gating: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        self.layers = nn.ModuleList([
            ZarvanBlock(embed_dim, hidden_dim, num_heads,
                        use_holistic=use_holistic, use_associative=use_associative, use_gating=use_gating)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, num_classes) # Output for sum classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        # Use the last token's vector for the final prediction of the sum
        z = x[:, -1, :] 
        logits = self.fc_out(z)
        return logits

class TransformerForCategoricalSum(nn.Module):
    """
    Standard Transformer Encoder model for Categorical Sum task.
    """
    def __init__(self, vocab_size, num_classes, seq_len, embed_dim, ff_dim, num_heads, num_layers, padding_idx=0, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Use the last token's vector for the final prediction of the sum
        z = x[:, -1, :]
        logits = self.fc_out(z)
        return logits


# ============================================================================
# Part 4: Categorical Sum Dataset (Modified for more complexity)
# ============================================================================

class CategoricalSumDataset(Dataset):
    def __init__(self, seq_len: int, num_samples: int, max_val_for_sum: int, num_categories: int):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.max_val_for_sum = max_val_for_sum # e.g., 7 means values 0, 1, ..., 6
        self.num_categories = num_categories # Number of distinct condition tokens/rules

        # Token definitions
        self.PAD_TOKEN = 0
        self.NOISE_TOKEN = 1
        self.MARKER_TOKEN = 2
        # Category tokens will be 3, 4, 5 (for A, B, C)
        self.CAT_TOKEN_START = 3 
        # Numerical tokens start after these fixed special tokens
        self.FIRST_NUM_TOKEN = self.CAT_TOKEN_START + self.num_categories # e.g., 3 + 3 = 6 (for NUM_0)

        self.vocab_size = self.FIRST_NUM_TOKEN + self.max_val_for_sum # e.g., 6 + 7 = 13
        
        # Define rules (arbitrary examples for now, can be adjusted)
        # Rule 1 (Category A): Sum * 2
        # Rule 2 (Category B): Sum + 5
        # Rule 3 (Category C): If sum is even, Sum + 10; else Sum - 5
        self.category_rules_multipliers = {
            self.CAT_TOKEN_START + 0: lambda s: s * 2,  # CAT_A (token 3)
            self.CAT_TOKEN_START + 1: lambda s: s + 5,  # CAT_B (token 4)
            self.CAT_TOKEN_START + 2: lambda s: s + 10 if s % 2 == 0 else s - 5 # CAT_C (token 5)
        }
        
        # Calculate max and min possible sums for output classes
        MAX_RAW_SUM = 2 * (self.max_val_for_sum - 1) # e.g., 6+6=12
        
        # Max/Min based on rules:
        max_possible_sum = MAX_RAW_SUM
        min_possible_sum = 0
        
        # Rule 1: 12 * 2 = 24
        max_possible_sum = max(max_possible_sum, MAX_RAW_SUM * 2)
        
        # Rule 2: 12 + 5 = 17
        max_possible_sum = max(max_possible_sum, MAX_RAW_SUM + 5)
        
        # Rule 3: 
        # Max for even: 12 + 10 = 22
        # Min for odd (if smallest odd raw sum is 1): 1 - 5 = -4
        # Smallest even raw sum 0: 0 + 10 = 10
        min_possible_sum = min(min_possible_sum, MAX_RAW_SUM - 5) # E.g., for raw sum 1, 1-5=-4
        
        # Range of possible results: e.g., -4 to 24.
        # To map to 0-indexed classes: Shift by absolute min value.
        self.output_sum_offset = abs(min_possible_sum) # e.g., 4
        self.num_classes = max_possible_sum + self.output_sum_offset + 1 # e.g., 24 + 4 + 1 = 29

        # Adjust range for placing numbers and markers to avoid conflicts
        self.min_token_pos = 10
        self.max_token_pos = self.seq_len - 10
        assert self.max_token_pos - self.min_token_pos >= (2*2 + 1), "Sequence too short for token placement." # 2 numbers + 2 markers + 1 category

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = torch.full((self.seq_len,), self.NOISE_TOKEN, dtype=torch.long)
        
        # 1. Place two marked numbers
        # Ensure numbers and markers don't overlap with special tokens at end/start
        marked_indices = random.sample(range(self.min_token_pos, self.max_token_pos), 2) 
        
        val1 = random.randint(0, self.max_val_for_sum - 1)
        val2 = random.randint(0, self.max_val_for_sum - 1)
        
        sequence[marked_indices[0]] = self.MARKER_TOKEN
        sequence[marked_indices[0] + 1] = val1 + self.FIRST_NUM_TOKEN 
        
        sequence[marked_indices[1]] = self.MARKER_TOKEN
        sequence[marked_indices[1] + 1] = val2 + self.FIRST_NUM_TOKEN

        raw_sum = val1 + val2

        # 2. Place a random Category Token
        category_token_val = random.randint(0, self.num_categories - 1)
        category_token = self.CAT_TOKEN_START + category_token_val
        
        # Ensure category token is not placed on marker or value tokens
        all_occupied_pos = set(marked_indices) | set([idx+1 for idx in marked_indices])
        available_pos_for_cat = list(set(range(self.min_token_pos, self.max_token_pos)) - all_occupied_pos)
        
        if not available_pos_for_cat: # Fallback if no ideal position found (should be rare)
            available_pos_for_cat = list(set(range(self.seq_len)) - all_occupied_pos) # Any empty spot
        
        cat_pos = random.choice(available_pos_for_cat)
        sequence[cat_pos] = category_token

        # 3. Calculate the final categorized sum based on the assigned category token
        # The logic here directly uses the chosen category_token_val
        # The model has to *discover* this category_token and its rule.
        
        rule_function = self.category_rules_multipliers[category_token]
        final_sum = rule_function(raw_sum)
        
        # Map final_sum to class index (0-indexed and handle negative values)
        target_class = final_sum + self.output_sum_offset
        
        # Ensure target_class is within bounds. This acts as clipping for extreme sums.
        target_class = max(0, min(target_class, self.num_classes - 1)) 

        target = torch.tensor(target_class, dtype=torch.long)
        return sequence, target

# ============================================================================
# Part 5: Training and Evaluation Helper (Generalized - No Change)
# ============================================================================

def run_experiment(model_name: str, model: nn.Module, 
                   train_loader: DataLoader, val_loader: DataLoader,
                   device: torch.device, config: dict):
    
    model.to(device)
    criterion = nn.CrossEntropyLoss() # For sum classification
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    print(f"\n--- Training {model_name} ---")
    
    # Store metrics for plotting
    train_losses_history = []
    val_losses_history = []
    val_accuracies_history = []

    start_time = time.time()
    best_accuracy = 0.0

    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']} [Train]", leave=False)
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs) # outputs: (B, num_classes)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=total_train_loss / (progress_bar.n + 1)) 

        # Validation phase
        model.eval()
        total_correct, total_predictions = 0, 0
        total_val_loss = 0
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{config['num_epochs']} [Val]", leave=False)

        with torch.no_grad():
            for inputs, targets in progress_bar_val:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                loss_val = criterion(outputs, targets)
                total_val_loss += loss_val.item()

                predicted_indices = outputs.argmax(dim=-1)
                total_correct += (predicted_indices == targets).sum().item()
                total_predictions += targets.size(0)
                
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = (total_correct / total_predictions) * 100 if total_predictions > 0 else 0
        
        print(f"Epoch {epoch}/{config['num_epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {accuracy:.2f}%")
        
        # Store for plotting
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        val_accuracies_history.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy

    total_time = time.time() - start_time
    print(f"Finished Training {model_name}. Best Val Accuracy: {best_accuracy:.2f}%. Total Time: {total_time:.2f}s")
    
    return {
        'best_accuracy': best_accuracy,
        'train_losses': train_losses_history,
        'val_losses': val_losses_history,
        'val_accuracies': val_accuracies_history
    }


# ============================================================================
# Part 6: Ablation Study Runner for Categorical Sum Task
# ============================================================================

def run_ablation_study_categorical_sum(config: dict, device: torch.device):
    print("\n" + "="*50)
    print("Starting Categorical Sum Ablation Study")
    print("="*50)

    # Use a dummy dataset instance to get vocab_size and num_classes
    dummy_dataset = CategoricalSumDataset(
        seq_len=config['seq_len'], 
        num_samples=1, # Just for getting properties
        max_val_for_sum=config['max_val_for_sum'],
        num_categories=config['num_categories']
    )
    vocab_size = dummy_dataset.vocab_size
    num_classes = dummy_dataset.num_classes
    PAD_TOKEN = dummy_dataset.PAD_TOKEN # Retrieve PAD_TOKEN from dataset instance

    print(f"Task Vocab Size: {vocab_size}, Output Classes (Sums): {num_classes}")

    train_dataset = CategoricalSumDataset(
        seq_len=config['seq_len'], 
        num_samples=config['train_samples'], 
        max_val_for_sum=config['max_val_for_sum'],
        num_categories=config['num_categories']
    )
    val_dataset = CategoricalSumDataset(
        seq_len=config['seq_len'], 
        num_samples=config['val_samples'], 
        max_val_for_sum=config['max_val_for_sum'],
        num_categories=config['num_categories']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    ablation_results = {}
    ablation_histories = {} 

    # 1. Full Zarvan (Baseline)
    print("\n--- Running Full Zarvan (Baseline) ---")
    full_zarvan = ZarvanForCategoricalSum(
        vocab_size=vocab_size, num_classes=num_classes, seq_len=config['seq_len'],
        embed_dim=config['embed_dim'], hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'], num_layers=config['num_layers'],
        padding_idx=PAD_TOKEN, use_holistic=True, use_associative=True, use_gating=True
    )
    result = run_experiment("Full Zarvan", full_zarvan, train_loader, val_loader, device, config)
    ablation_results['Full Zarvan'] = result['best_accuracy']
    ablation_histories['Full Zarvan'] = result

    # 2. Zarvan without Holistic Context
    print("\n--- Running Zarvan -No Holistic Context ---")
    no_holistic_zarvan = ZarvanForCategoricalSum(
        vocab_size=vocab_size, num_classes=num_classes, seq_len=config['seq_len'],
        embed_dim=config['embed_dim'], hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'], num_layers=config['num_layers'],
        padding_idx=PAD_TOKEN, use_holistic=False, use_associative=True, use_gating=True
    )
    result = run_experiment("Zarvan -No Holistic", no_holistic_zarvan, train_loader, val_loader, device, config)
    ablation_results['Zarvan -No Holistic'] = result['best_accuracy']
    ablation_histories['Zarvan -No Holistic'] = result

    # 3. Zarvan without Associative Context
    print("\n--- Running Zarvan -No Associative Context ---")
    no_associative_zarvan = ZarvanForCategoricalSum(
        vocab_size=vocab_size, num_classes=num_classes, seq_len=config['seq_len'],
        embed_dim=config['embed_dim'], hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'], num_layers=config['num_layers'],
        padding_idx=PAD_TOKEN, use_holistic=True, use_associative=False, use_gating=True
    )
    result = run_experiment("Zarvan -No Associative", no_associative_zarvan, train_loader, val_loader, device, config)
    ablation_results['Zarvan -No Associative'] = result['best_accuracy']
    ablation_histories['Zarvan -No Associative'] = result

    # 4. Zarvan without Gating Mechanism
    print("\n--- Running Zarvan -No Gating Mechanism ---")
    no_gating_zarvan = ZarvanForCategoricalSum(
        vocab_size=vocab_size, num_classes=num_classes, seq_len=config['seq_len'],
        embed_dim=config['embed_dim'], hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'], num_layers=config['num_layers'],
        padding_idx=PAD_TOKEN, use_holistic=True, use_associative=True, use_gating=False
    )
    result = run_experiment("Zarvan -No Gating", no_gating_zarvan, train_loader, val_loader, device, config)
    ablation_results['Zarvan -No Gating'] = result['best_accuracy']
    ablation_histories['Zarvan -No Gating'] = result

    # 5. Transformer Baseline
    print("\n--- Running Transformer Baseline ---")
    transformer_model = TransformerForCategoricalSum(
        vocab_size=vocab_size, num_classes=num_classes, seq_len=config['seq_len'],
        embed_dim=config['embed_dim'], ff_dim=config['ff_dim'], num_heads=config['num_heads'], 
        num_layers=config['num_layers'], padding_idx=PAD_TOKEN
    )
    result = run_experiment("Transformer", transformer_model, train_loader, val_loader, device, config)
    ablation_results['Transformer'] = result['best_accuracy']
    ablation_histories['Transformer'] = result

    print("\n" + "="*50)
    print("Categorical Sum Ablation Study Results:")
    print("="*50)
    for model_type, acc in ablation_results.items():
        print(f"{model_type}: {acc:.2f}% Accuracy")
    print("="*50)

    # --- Plotting Ablation Study Trends ---
    epochs_range = range(1, config['num_epochs'] + 1)

    plt.figure(figsize=(12, 6))
    for model_name, history in ablation_histories.items():
        plt.plot(epochs_range, history['val_accuracies'], label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Categorical Sum Ablation Study: Validation Accuracy vs. Epoch")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ablation_cs_accuracy_trend.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    for model_name, history in ablation_histories.items():
        plt.plot(epochs_range, history['val_losses'], label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Categorical Sum Ablation Study: Validation Loss vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ablation_cs_loss_trend.png")
    plt.close()

    return ablation_results


# ============================================================================
# Part 7: Hyperparameter Sensitivity Study Runner for Nested Categorical Sum Task
# ============================================================================

def run_hyperparameter_sensitivity_study_categorical_sum(config: dict, device: torch.device):
    print("\n" + "="*50)
    print("Starting Categorical Sum Hyperparameter Sensitivity Study (Zarvan)")
    print("="*50)

    # Use a dummy dataset instance to get vocab_size and num_classes
    dummy_dataset = CategoricalSumDataset(
        seq_len=config['seq_len'], 
        num_samples=1, 
        max_val_for_sum=config['max_val_for_sum'],
        num_categories=config['num_categories']
    )
    vocab_size = dummy_dataset.vocab_size
    num_classes = dummy_dataset.num_classes
    PAD_TOKEN = dummy_dataset.PAD_TOKEN

    train_dataset = CategoricalSumDataset(
        seq_len=config['seq_len'], 
        num_samples=config['train_samples'], 
        max_val_for_sum=config['max_val_for_sum'],
        num_categories=config['num_categories']
    )
    val_dataset = CategoricalSumDataset(
        seq_len=config['seq_len'], 
        num_samples=config['val_samples'], 
        max_val_for_sum=config['max_val_for_sum'],
        num_categories=config['num_categories']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    sensitivity_results = {} 
    sensitivity_histories = {} 

    # Parameters to vary and their values
    embed_dims = [64, 128, 256]
    num_heads_options = [2, 4, 8]
    num_layers_options = [1, 2, 4] 

    # Sensitivity to embed_dim
    print("\n--- Sensitivity to Embedding Dimension (embed_dim) ---")
    sensitivity_histories['embed_dim'] = {}
    for dim in embed_dims:
        print(f"Testing with embed_dim = {dim}")
        current_zarvan = ZarvanForCategoricalSum(
            vocab_size=vocab_size, num_classes=num_classes, seq_len=config['seq_len'], embed_dim=dim,
            hidden_dim=config['hidden_dim'], num_heads=config['num_heads'], num_layers=config['num_layers'],
            padding_idx=PAD_TOKEN
        )
        result = run_experiment(f"Zarvan (embed_dim={dim})", current_zarvan, train_loader, val_loader, device, config)
        sensitivity_results[f"embed_dim={dim}"] = result['best_accuracy']
        sensitivity_histories['embed_dim'][dim] = result

    # Sensitivity to num_heads
    print("\n--- Sensitivity to Number of Heads (num_heads) ---")
    sensitivity_histories['num_heads'] = {}
    for heads in num_heads_options:
        print(f"Testing with num_heads = {heads}")
        current_embed_dim = config['embed_dim'] 
        if current_embed_dim % heads != 0:
            current_embed_dim = heads * (current_embed_dim // heads + 1)
            print(f"Adjusted embed_dim to {current_embed_dim} for num_heads={heads}")
        
        current_zarvan = ZarvanForCategoricalSum(
            vocab_size=vocab_size, num_classes=num_classes, seq_len=config['seq_len'], embed_dim=current_embed_dim,
            hidden_dim=config['hidden_dim'], num_heads=heads, num_layers=config['num_layers'],
            padding_idx=PAD_TOKEN
        )
        result = run_experiment(f"Zarvan (num_heads={heads})", current_zarvan, train_loader, val_loader, device, config)
        sensitivity_results[f"num_heads={heads}"] = result['best_accuracy']
        sensitivity_histories['num_heads'][heads] = result

    # Sensitivity to num_layers
    print("\n--- Sensitivity to Number of Layers (num_layers) ---")
    sensitivity_histories['num_layers'] = {}
    for layers in num_layers_options:
        print(f"Testing with num_layers = {layers}")
        current_zarvan = ZarvanForCategoricalSum(
            vocab_size=vocab_size, num_classes=num_classes, seq_len=config['seq_len'], embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'], num_heads=config['num_heads'], num_layers=layers,
            padding_idx=PAD_TOKEN
        )
        result = run_experiment(f"Zarvan (num_layers={layers})", current_zarvan, train_loader, val_loader, device, config)
        sensitivity_results[f"num_layers={layers}"] = result['best_accuracy']
        sensitivity_histories['num_layers'][layers] = result

    print("\n" + "="*50)
    print("Hyperparameter Sensitivity Study Results on Categorical Sum (Zarvan):")
    print("="*50)
    for param_val, acc in sensitivity_results.items():
        print(f"{param_val}: {acc:.2f}% Accuracy")
    print("="*50)

    # --- Plotting Sensitivity Study Trends ---
    epochs_range = range(1, config['num_epochs'] + 1)

    # Plot for embed_dim
    plt.figure(figsize=(12, 6))
    for dim, history in sensitivity_histories['embed_dim'].items():
        plt.plot(epochs_range, history['val_accuracies'], label=f"embed_dim={dim}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Zarvan Sensitivity: Validation Accuracy vs. Epoch (Embed Dim)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sensitivity_cs_embed_dim_trend.png")
    plt.close()

    # Plot for num_heads
    plt.figure(figsize=(12, 6))
    for heads, history in sensitivity_histories['num_heads'].items():
        plt.plot(epochs_range, history['val_accuracies'], label=f"num_heads={heads}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Zarvan Sensitivity: Validation Accuracy vs. Epoch (Num Heads)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sensitivity_cs_num_heads_trend.png")
    plt.close()

    # Plot for num_layers
    plt.figure(figsize=(12, 6))
    for layers, history in sensitivity_histories['num_layers'].items():
        plt.plot(epochs_range, history['val_accuracies'], label=f"num_layers={layers}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Zarvan Sensitivity: Validation Accuracy vs. Epoch (Num Layers)")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sensitivity_cs_num_layers_trend.png")
    plt.close()

    return sensitivity_results

# --- Main Execution Block ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Run Ablation Study for Nested Categorical Sum Task
    ablation_final_results_cs = run_ablation_study_categorical_sum(FLAGS, device)

    # Run Hyperparameter Sensitivity Study for Zarvan on Nested Categorical Sum Task
    sensitivity_flags_cs = FLAGS.copy() 
    sensitivity_final_results_cs = run_hyperparameter_sensitivity_study_categorical_sum(sensitivity_flags_cs, device)