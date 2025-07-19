# zarvan_model.py
#
# Official PyTorch implementation of the Zarvan architecture.
# VERSION 2: Includes key improvements for enhanced information flow and depth.
#
# Zarvan is an efficient, high-performance architecture for sequence modeling,
# demonstrating strong results on vision and language tasks.
#
# Author: Yasser Sajjadi, in collaboration with Google's Gemini.
# Architectural Improvements Contributor: Google's Gemini & User Feedback.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time # Added for the example usage block

class SinusoidalPositionalEncoding(nn.Module):
    """
    Injects sinusoidal positional information into the input embeddings.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

class LinearQueryExtractor(nn.Module):
    """
    Extracts a global query vector from a sequence with linear complexity.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.s_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        s = self.s_proj(x)
        v = self.v_proj(x)
        s = s.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        weights = F.softmax(s, dim=-1)
        head_outputs = (weights * v).sum(dim=2)
        concatenated_heads = head_outputs.reshape(B, self.embed_dim)
        q = self.combine(concatenated_heads)
        return q

class ZarvanBlock(nn.Module):
    """
    The core Zarvan architecture block with ENHANCED information flow.
    """
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
        
        # 1. Extract the global query vector 'q'.
        q = self.query_extractor(x)
        
        # 2. Transform 'q' into an 'interactive context' vector.
        i_ctx = F.gelu(self.interactive_context(q))
        
        # 3. Expand global vectors to match sequence length.
        q_exp = q.unsqueeze(1).expand(-1, S, -1)
        i_ctx_exp = i_ctx.unsqueeze(1).expand(-1, S, -1)
        
        # 4. Create input for the gate network.
        gate_input = torch.cat([x, q_exp, i_ctx_exp], dim=-1)
        
        # 5. Calculate gates.
        gates = self.gate_net(gate_input)
        input_gate, forget_gate = gates.chunk(2, dim=-1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        
        # 6. Apply selective copy mechanism.
        gated_x = input_gate * x + forget_gate * self.gated_proj(x)
        
        # 7. 🚀 KEY IMPROVEMENT: Inject global context for enhanced information flow.
        # This provides the FFN with the raw query (q_exp), its non-linear 
        # transformation (i_ctx_exp), and the locally gated token representation (gated_x).
        ffn_input = gated_x + i_ctx_exp + q_exp
        output = self.ffn(ffn_input)
        
        # 8. Apply residual connection and layer normalization.
        return self.norm(x + output)

class ZarvanForClassification(nn.Module):
    """
    A complete classifier model built using a stack of ZarvanBlocks.
    """
    def __init__(self, vocab_size: int, num_classes: int, seq_len: int,
                 embed_dim: int, hidden_dim: int, num_heads: int,
                 num_layers: int = 1, # ✅ Added num_layers parameter
                 padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=seq_len)
        
        # ✅ Use a ModuleList to stack multiple ZarvanBlocks
        self.zarvan_layers = nn.ModuleList([
            ZarvanBlock(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Embed tokens and add positional info.
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # 2. ✅ Pass through the stacked Zarvan blocks.
        for layer in self.zarvan_layers:
            x = layer(x)
        
        # 3. Pool the output sequence for classification.
        z = x.mean(dim=1)
        
        # 4. Apply the final classification layer.
        return self.fc(z)

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Demonstrating the Improved ZarvanForClassification model ---")

    # Define some dummy parameters for a demonstration
    BATCH_SIZE = 4
    SEQ_LEN = 512
    VOCAB_SIZE = 10000
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    NUM_HEADS = 4
    NUM_LAYERS = 2  # ✅ Using a 2-layer model for demonstration
    NUM_CLASSES = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning example on: {device}")
    print(f"Model configured with {NUM_LAYERS} Zarvan layers.")

    # 1. Create the model instance
    model = ZarvanForClassification(
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, # ✅ Pass num_layers
        padding_idx=0
    ).to(device)

    # 2. Create a dummy input batch
    dummy_input = torch.randint(low=1, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN)).to(device)
    
    print(f"\nModel Architecture:\n{model}")
    print(f"\nInput shape: {dummy_input.shape} (Batch, Sequence)")
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # 3. Perform a forward pass
    try:
        start_time = time.time()
        output = model(dummy_input)
        end_time = time.time()
        
        print(f"\nOutput shape: {output.shape} (Batch, NumClasses)")
        print(f"Inference time for {BATCH_SIZE} samples: {end_time - start_time:.4f} seconds")
        
        assert output.shape == (BATCH_SIZE, NUM_CLASSES)
        print("\n✅ Model forward pass successful and output shape is correct!")
    except Exception as e:
        print(f"\n❌ An error occurred during the forward pass: {e}")

    print("\n--- End of ZarvanForClassification demonstration ---")
