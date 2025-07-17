# zarvan_model.py
#
# Official PyTorch implementation of the Zarvan architecture.
# Zarvan is an efficient, high-performance architecture for sequence modeling,
# demonstrating strong results on vision and language tasks.
#
# Author: Yasser Sajjadi, in collaboration with Google's Gemini.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    This implementation is based on the original Transformer paper.
    It is a fixed, non-learned encoding.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, embed_dim)
        Returns:
            torch.Tensor: Input tensor with added positional information.
        """
        x = x + self.pe[:x.size(0)]
        return x

class ParallelUnifiedAttention(nn.Module):
    """
    A parallelized version of the UnifiedAttention mechanism.
    It removes the Python for-loop to process all heads simultaneously for efficiency.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # A single linear layer projects the input for all heads at once.
        self.query_net = nn.Linear(embed_dim, embed_dim * num_heads)
        # A single linear layer to combine the head outputs at the end.
        self.combine = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, embed_dim)
        Returns:
            torch.Tensor: Context vector of shape (batch_size, embed_dim)
        """
        seq_len, batch_size, _ = x.shape

        # 1. Project input for all heads simultaneously.
        # (seq_len, batch, dim) -> (seq_len, batch, dim * heads)
        scores_all_heads = self.query_net(x)

        # 2. Reshape for parallel computation.
        # (seq_len, batch, dim * heads) -> (seq_len, batch * heads, dim)
        scores_all_heads = scores_all_heads.view(seq_len, batch_size * self.num_heads, self.embed_dim)

        # 3. Apply softmax along the sequence dimension for all heads in parallel.
        attn_weights = F.softmax(scores_all_heads, dim=0)

        # 4. Prepare input 'x' for element-wise multiplication by repeating it for each head.
        # (seq_len, batch, dim) -> (seq_len, batch * heads, dim)
        x_repeated = x.repeat_interleave(self.num_heads, dim=1)
        
        # 5. Apply attention weights.
        weighted_values = x_repeated * attn_weights

        # 6. Sum over the sequence dimension to get the context vector for each head.
        # (seq_len, batch * heads, dim) -> (batch * heads, dim)
        head_outputs = weighted_values.sum(dim=0)

        # 7. Reshape to concatenate heads.
        # (batch * heads, dim) -> (batch, heads * dim)
        concatenated_heads = head_outputs.view(batch_size, self.num_heads * self.embed_dim)
        
        # 8. Apply the final projection layer.
        # (batch, heads * dim) -> (batch, dim)
        return self.combine(concatenated_heads)

class ZarvanBlock(nn.Module):
    """ 
    The final, successful Zarvan architecture block.
    It uses the filter mechanism and expects external positional encoding.
    """
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with positional encoding, shape (seq_len, batch_size, embed_dim)
        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        # 1. Get the primary context vector 'q' from the attention mechanism.
        q = self.attention(x)
        
        # 2. Create an "interactive context" based on 'q'.
        i_ctx = F.relu(self.interactive_context(q))
        
        # 3. Expand context vectors to match the sequence length for the filter.
        q_exp = q.unsqueeze(0).expand(x.size(0), -1, -1)
        i_ctx_exp = i_ctx.unsqueeze(0).expand(x.size(0), -1, -1)
        
        # 4. Concatenate all information to create the input for the filter network.
        filter_in = torch.cat([x, q_exp, i_ctx_exp], dim=-1)
        
        # 5. Calculate and center the filter weights.
        f_weights = self.filter_net(filter_in)
        f_weights = f_weights - f_weights.mean(dim=0, keepdim=True)
        
        # 6. Apply the gated residual connection.
        z = x + self.Linear_xw(x * f_weights)
        
        # 7. Apply final layer normalization.
        return self.norm(z)

class ZarvanForClassification(nn.Module):
    """
    A complete, user-facing classifier model using the final Zarvan architecture.
    Handles embedding, positional encoding, and final classification.
    """
    def __init__(self, vocab_size: int, num_classes: int, seq_len: int,
                 embed_dim: int, hidden_dim: int, num_heads: int,
                 padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=seq_len)
        self.zarvan_block = ZarvanBlock(embed_dim, hidden_dim, num_heads)
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of token IDs, shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # 1. Embed tokens and set tensor shape for sequence models
        # Input: (batch, seq) -> Output: (seq, batch, dim)
        x = self.embedding(x).permute(1, 0, 2)
        
        # 2. Add positional information
        x = self.pos_encoder(x)
        
        # 3. Pass through the main Zarvan block
        x = self.zarvan_block(x)
        
        # 4. Pool the output sequence into a single vector (average pooling)
        z = x.mean(dim=0)
        
        # 5. Final classification layer
        return self.fc(z)

# --- Example Usage ---
if __name__ == '__main__':
    # Define some dummy parameters for a demonstration
    BATCH_SIZE = 4
    SEQ_LEN = 512
    VOCAB_SIZE = 10000
    EMBED_DIM = 128
    HIDDEN_DIM = 128
    NUM_HEADS = 4
    NUM_CLASSES = 5

    print("--- Demonstrating the ZarvanForClassification model ---")

    # 1. Create the model instance
    model = ZarvanForClassification(
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        padding_idx=0
    )

    # 2. Create a dummy input batch of token IDs
    dummy_input = torch.randint(low=1, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    
    print(f"\nModel Architecture:\n{model}")
    print(f"\nInput shape: {dummy_input.shape}")

    # 3. Perform a forward pass
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print("\n✅ Model forward pass successful!")
        assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    except Exception as e:
        print(f"\n❌ An error occurred during the forward pass: {e}")