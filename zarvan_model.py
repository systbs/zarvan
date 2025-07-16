# zarvan_model.py
#
# PyTorch implementation of the Zarvan architecture, an efficient alternative
# to the Transformer for sequence modeling.
#
# Author: Yasser Sajjadi, in collaboration with Google's Gemini and Grok.
# Date: July 2025

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridMultiHeadLinearAttention(nn.Module):
    """
    The core attention mechanism of Zarvan.
    It combines two types of linear attention heads:
    1. Local Heads: Compute attention scores based on each token's representation.
    2. Global Heads: Compute attention scores based on the mean representation of the entire sequence.
    This mechanism operates in O(n) time complexity with respect to sequence length n.
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.head_dim = embed_dim // num_heads

        # Split heads into local and global
        num_local_heads = num_heads // 2
        num_global_heads = num_heads - num_local_heads

        # Projections and scoring layers for local heads
        self.local_proj = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_local_heads)])
        self.local_score = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_local_heads)])

        # Projections and scoring layers for global heads
        self.global_proj = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_global_heads)])
        self.global_score = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_global_heads)])
        
        # Final output projection
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, embed_dim)
        Returns:
            torch.Tensor: Output context vector of shape (batch_size, embed_dim)
        """
        seq_len, batch_size, _ = x.shape
        head_outputs = []

        # --- Local Attention Path ---
        for proj, score_fn in zip(self.local_proj, self.local_score):
            # Each token gets a score
            scores = score_fn(x).squeeze(-1)  # Shape: (seq_len, batch_size)
            # Softmax over the sequence dimension to find important tokens
            attn_weights = F.softmax(scores, dim=0).unsqueeze(-1)  # Shape: (seq_len, batch_size, 1)
            # Project input to value and apply attention
            values = proj(x)  # Shape: (seq_len, batch_size, head_dim)
            # Weighted sum of values
            head_outputs.append((values * attn_weights).sum(dim=0)) # Shape: (batch_size, head_dim)

        # --- Global Attention Path ---
        # Get a single global context vector for the whole sequence
        global_context = x.mean(dim=0) # Shape: (batch_size, embed_dim)
        for proj, score_fn in zip(self.global_proj, self.global_score):
            # The global context vector gets a score
            scores = score_fn(global_context) # Shape: (batch_size, 1)
            # Softmax over the batch dimension to create competition between sequences
            attn_weights = F.softmax(scores, dim=0) # Shape: (batch_size, 1)
            # Expand weights to apply to the whole sequence
            attn_weights_expanded = attn_weights.unsqueeze(0).expand(seq_len, -1, -1) # Shape: (seq_len, batch_size, 1)
            # Project input to value and apply attention
            values = proj(x) # Shape: (seq_len, batch_size, head_dim)
            # Weighted sum of values
            head_outputs.append((values * attn_weights_expanded).sum(dim=0)) # Shape: (batch_size, head_dim)
        
        # Concatenate all head outputs and project back to embed_dim
        concatenated = torch.cat(head_outputs, dim=-1) # Shape: (batch_size, embed_dim)
        return self.fc_out(concatenated)


class ZarvanBlock(nn.Module):
    """
    The main building block of the Zarvan model.
    It follows a standard Pre-LN Transformer block structure but uses
    the HybridMultiHeadLinearAttention instead of quadratic self-attention.
    """
    def __init__(self, seq_len, embed_dim, hidden_dim, num_heads):
        super().__init__()
        self.attention = HybridMultiHeadLinearAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, embed_dim)
        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, embed_dim)
        """
        # Add positional encoding only once at the beginning
        x = x + self.pos_encoding[:x.size(0)]
        
        # --- Attention Sub-layer with Pre-Norm and Residual Connection ---
        residual = x
        attn_out = self.attention(self.norm1(x))
        # Add sequence dimension back for the residual connection
        x = residual + attn_out.unsqueeze(0)
        
        # --- FFN Sub-layer with Pre-Norm and Residual Connection ---
        residual = x
        ffn_out = self.ffn(self.norm2(x))
        x = residual + ffn_out
        
        return x


class ZarvanClassifier(nn.Module):
    """
    A wrapper model that uses a ZarvanBlock for classification tasks.
    """
    def __init__(self, vocab_size: int, num_classes: int, seq_len: int,
                 embed_dim: int, hidden_dim: int, num_heads: int,
                 padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.zarvan_block = ZarvanBlock(seq_len, embed_dim, hidden_dim, num_heads)
        # The final classification head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of token IDs, shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # 1. Embed tokens
        # Input: (batch_size, seq_len) -> Output: (batch_size, seq_len, embed_dim)
        x = self.embedding(x)
        
        # 2. PyTorch models often expect (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)
        
        # 3. Pass through the Zarvan block
        x = self.zarvan_block(x)
        
        # 4. Pool the output over the sequence dimension (average pooling)
        # Input: (seq_len, batch_size, embed_dim) -> Output: (batch_size, embed_dim)
        x_pooled = x.mean(dim=0)
        
        # 5. Final classification layer
        # Input: (batch_size, embed_dim) -> Output: (batch_size, num_classes)
        return self.fc(x_pooled)


# --- Example Usage ---
if __name__ == '__main__':
    # Define some dummy parameters for a test run
    BATCH_SIZE = 4
    SEQ_LEN = 128
    VOCAB_SIZE = 1000
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    NUM_HEADS = 4
    NUM_CLASSES = 5 # Example for a 5-class classification problem

    print("--- Testing ZarvanClassifier ---")

    # 1. Create the model instance
    model = ZarvanClassifier(
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        padding_idx=1
    )

    # 2. Create a dummy input batch
    # (A batch of 4 sequences, each 128 tokens long)
    dummy_input = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    
    print(f"\nModel Architecture:\n{model}")
    print(f"\nInput shape: {dummy_input.shape}")

    # 3. Perform a forward pass
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print("Model forward pass successful!")
        assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")