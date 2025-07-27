# ============================================================================
#
#           Zarvan: The Official PyTorch Implementation
#
# This file contains the definitive, polished implementation of the Zarvan
# architecture, a highly efficient, linearly scalable alternative to the
# Transformer for sequence modeling.
#
# As detailed in the paper "Zarvan: An Efficient Gated Architecture for
# Sequence Modeling with Linear Complexity," this architecture is designed
# to be both powerful and computationally frugal.
#
# Author: Yasser Sajjadi, in collaboration with Google's Gemini
# GitHub: https://github.com/systbs/zarvan 
#
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# ============================================================================
# Part 1: Core Context Kernels
# These are the fundamental building blocks for understanding the sequence.
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Injects sinusoidal positional information into the input sequence. This
    allows the model to understand the order of tokens, which is crucial for
    any sequence processing task.
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
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Sequence, Embedding).
        Returns:
            torch.Tensor: Output tensor with added positional information.
        """
        return x + self.pe[:, :x.size(1), :]

class HolisticContextExtractor(nn.Module):
    """
    Extracts a holistic context vector from the entire sequence.
    This module produces a single, comprehensive summary of the sequence
    with linear complexity. This vector represents the overall "gist" or
    "essence" of the sequence, useful for general understanding.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads."
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
        head_outputs = torch.sum(weights * v, dim=2)
        concatenated_heads = head_outputs.reshape(B, self.embed_dim)
        return self.combine(concatenated_heads)

class AssociativeContextExtractor(nn.Module):
    """
    Extracts an associative context vector from the sequence.
    This module learns an importance score for each token, allowing it to
    selectively aggregate information from important parts of the sequence
    while ignoring noise. This vector acts as the model's "focused memory,"
    crucial for tasks requiring long-range information retrieval.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.s_proj = nn.Linear(embed_dim, 1)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, v = self.s_proj(x), self.v_proj(x)
        weights = F.softmax(s, dim=1)
        context = torch.sum(weights * v, dim=1)
        return context

# ============================================================================
# Part 2: The Definitive Zarvan Block
# This is the heart of the architecture, combining multiple context types.
# ============================================================================

class ZarvanBlock(nn.Module):
    """
    The definitive, hybrid Zarvan Block.
    This block operates by leveraging two parallel context types:
    1. Holistic Context: For a general understanding of the content.
    2. Associative Context: To focus on important, sparse details.
    Together, these contexts inform a powerful and intelligent gating mechanism,
    allowing the model to be both a strong generalist and a precise specialist.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        # --- Dual Context Kernels ---
        self.holistic_ctx = HolisticContextExtractor(embed_dim, num_heads)
        self.associative_ctx = AssociativeContextExtractor(embed_dim)
        
        # --- Gating Mechanism ---
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim), # Input: [x, holistic_ctx, associative_ctx]
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim * 2)  # For input_gate and forget_gate
        )
        self.update_proj = nn.Linear(embed_dim, embed_dim)
        
        # --- Final Processing ---
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        
        # 1. Extract dual context vectors in parallel
        q_holistic = self.holistic_ctx(x)
        q_associative = self.associative_ctx(x)
        
        # 2. Prepare the input for the gate network
        q_holistic_exp = q_holistic.unsqueeze(1).expand(-1, S, -1)
        q_associative_exp = q_associative.unsqueeze(1).expand(-1, S, -1)
        
        gate_input = torch.cat([x, q_holistic_exp, q_associative_exp], dim=-1)
        
        # 3. Compute gates and apply the update mechanism
        input_gate, forget_gate = self.gate_net(gate_input).chunk(2, dim=-1)
        
        gated_x = torch.sigmoid(input_gate) * x + \
                  torch.sigmoid(forget_gate) * self.update_proj(x)
        
        # 4. Final processing and residual connection
        output = self.ffn(gated_x)
        return self.norm(x + output)

# ============================================================================
# Part 3: The Complete Model Interface
# A high-level wrapper to easily build Zarvan models.
# ============================================================================

class ZarvanModel(nn.Module):
    """
    A complete, general-purpose Zarvan model.
    This class provides a simple interface to build a stacked Zarvan model for
    various sequence processing tasks.

    Args:
        vocab_size (int): The size of the input vocabulary.
        num_classes (int): The number of output classes for classification tasks.
                           Set to vocab_size for language modeling.
        seq_len (int): The maximum sequence length the model will handle.
        embed_dim (int): The core dimensionality of the model's embeddings.
        hidden_dim (int): The dimensionality of the hidden layers in FFNs.
        num_heads (int): The number of heads for the HolisticContextExtractor.
        num_layers (int): The number of ZarvanBlocks to stack.
        padding_idx (int, optional): The index of the padding token. Defaults to 0.
    """
    def __init__(self, vocab_size: int, num_classes: int, seq_len: int,
                 embed_dim: int, hidden_dim: int, num_heads: int,
                 num_layers: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        self.layers = nn.ModuleList([
            ZarvanBlock(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor, pool: str = 'mean') -> torch.Tensor:
        """
        Performs the forward pass of the Zarvan model.

        Args:
            x (torch.Tensor): Input tensor of token IDs, shape (Batch, Sequence).
            pool (str, optional): The pooling strategy for classification.
                                  'mean': Average pooling over the sequence.
                                  'last': Use only the last token's output.
                                  'none': Return the full sequence output.
                                  Defaults to 'mean'.

        Returns:
            torch.Tensor: The model's output. Shape depends on the pooling strategy.
        """
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        
        if pool == 'mean':
            # Global average pooling for classification
            z = x.mean(dim=1)
            return self.output_head(z)
        elif pool == 'last':
            # Use the last hidden state for tasks like the Adding Problem
            z = x[:, -1, :]
            return self.output_head(z)
        elif pool == 'none':
            # Return the full sequence for token-level tasks (e.g., language modeling)
            return self.output_head(x)
        else:
            raise ValueError(f"Unknown pooling strategy: {pool}")

# ============================================================================
# Part 4: Example Usage
# A simple demonstration of how to instantiate and use the ZarvanModel.
# ============================================================================

if __name__ == '__main__':
    print("--- Demonstrating the Definitive Zarvan Architecture ---")

    # --- Model Configuration ---
    VOCAB_SIZE = 10000
    NUM_CLASSES = 10
    SEQ_LEN = 512
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    NUM_HEADS = 4
    NUM_LAYERS = 4
    BATCH_SIZE = 16

    # --- Instantiation ---
    print("\n1. Instantiating the ZarvanModel for a classification task...")
    model = ZarvanModel(
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Model created successfully on {str(device).upper()}.")
    print(f"   - Total trainable parameters: {param_count / 1e6:.2f}M")

    # --- Dummy Data and Forward Pass ---
    print("\n2. Performing a dummy forward pass...")
    dummy_input = torch.randint(low=1, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN)).to(device)
    
    try:
        start_time = time.time()
        # Use mean pooling for classification
        output = model(dummy_input, pool='mean')
        end_time = time.time()

        print(f"   - Input shape:  {list(dummy_input.shape)}")
        print(f"   - Output shape: {list(output.shape)}")
        assert list(output.shape) == [BATCH_SIZE, NUM_CLASSES]
        print(f"   - Inference time for {BATCH_SIZE} samples: {(end_time - start_time) * 1000:.2f} ms")
        print("\n✅ Zarvan model is ready for use!")

    except Exception as e:
        print(f"\n❌ An error occurred during the forward pass: {e}")
