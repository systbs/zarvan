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
    Injects sinusoidal positional information into the input embeddings.
    This implementation is based on the original Transformer paper
    (Vaswani et al., 2017) and is adapted for the (Batch, Sequence, Embedding)
    input tensor shape. Positional encodings help the model understand
    the order of elements in a sequence.

    Args:
        embed_dim (int): The dimension of the input and output embeddings.
        max_len (int, optional): The maximum expected sequence length.
                                 This determines the size of the pre-computed
                                 positional encoding table. Defaults to 5000.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        # Create a tensor for positions (0, 1, ..., max_len-1)
        position = torch.arange(max_len).unsqueeze(1) # Shape: (max_len, 1)

        # Calculate the division term for the sinusoidal functions.
        # This ensures that wavelengths form a geometric progression from 2*pi to 10000*2*pi.
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        # Initialize positional encoding table
        pe = torch.zeros(max_len, embed_dim)

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register 'pe' as a buffer. Buffers are tensors that are part of the module's state
        # but are not parameters (i.e., not updated by optimizers).
        # Unsqueeze(0) makes it (1, max_len, embed_dim) for easy broadcasting with (B, S, E) input.
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Sequence, Embedding).

        Returns:
            torch.Tensor: Input tensor with added positional information,
                          of the same shape (Batch, Sequence, Embedding).
        """
        # Add positional encoding to the input tensor.
        # The slicing `[:, :x.size(1), :]` ensures that we only use
        # positional encodings up to the current sequence length of x.
        x = x + self.pe[:, :x.size(1), :]
        return x

class LinearQueryExtractor(nn.Module):
    """
    Extracts a global query vector from a sequence with linear complexity.
    This module processes the entire sequence to derive a single,
    contextualized representation (query vector) that summarizes the sequence.
    It achieves this by calculating importance weights for each token and
    applying them to a learned 'value' representation, similar to attention
    but without quadratic complexity. It is fully vectorized to process
    all heads in parallel.

    Args:
        embed_dim (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads. `embed_dim` must be
                         divisible by `num_heads`.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        # Ensure that the embedding dimension can be evenly split across heads.
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # Dimension of each head's output
        
        # Linear projection layers for 'scores' (s) and 'values' (v).
        # These transform the input features into representations suitable for weighting.
        self.s_proj = nn.Linear(embed_dim, embed_dim) # Projects to (B, S, E)
        self.v_proj = nn.Linear(embed_dim, embed_dim) # Projects to (B, S, E)
        
        # Final linear layer to combine the outputs from all heads into the final query vector.
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass to extract the global query vector.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Sequence, Embedding).

        Returns:
            torch.Tensor: Global query vector of shape (Batch, Embedding),
                          representing a summary of the input sequence.
        """
        B, S, _ = x.shape # B: Batch size, S: Sequence length, E: Embedding dimension

        # 1. Project input to scores (s) and values (v).
        # These projections are independent for each token.
        s = self.s_proj(x)  # Shape: (B, S, E)
        v = self.v_proj(x)  # Shape: (B, S, E)
        
        # 2. Reshape for multi-head processing.
        # The tensor is reshaped to separate heads and then permuted
        # to bring the head dimension forward for parallel computation.
        # (B, S, E) -> (B, S, num_heads, head_dim) -> (B, num_heads, S, head_dim)
        s = s.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3. Calculate weights using softmax along the sequence dimension (-1).
        # This assigns an importance score to each token's 'score' representation
        # within each head. The sum of weights for each head and batch item is 1.
        # Shape: (B, num_heads, S, head_dim)
        weights = F.softmax(s, dim=-1) 

        # 4. Apply weights to values and sum over the sequence dimension (dim=2)
        # to get the output from each head. This effectively performs a weighted
        # average of the 'value' representations, summarizing the sequence for each head.
        # (B, num_heads, S, head_dim) * (B, num_heads, S, head_dim) -> sum(dim=2) -> (B, num_heads, head_dim)
        head_outputs = (weights * v).sum(dim=2)
        
        # 5. Reshape and concatenate the outputs from all heads.
        # This combines the individual head summaries back into a single embedding dimension.
        # (B, num_heads, head_dim) -> (B, num_heads * head_dim) -> (B, E)
        concatenated_heads = head_outputs.reshape(B, self.embed_dim)
        
        # 6. Apply a final linear projection to produce the global query vector.
        # This allows for further transformation and integration of the combined head outputs.
        # Shape: (B, E)
        q = self.combine(concatenated_heads)
        return q

class ZarvanBlock(nn.Module):
    """
    The core Zarvan architecture block, implementing a linear-complexity
    selective copy mechanism. It uses a global query vector (derived from
    the entire sequence) to dynamically filter and update each token's
    representation through learned gating mechanisms.

    This block aims to achieve efficient sequence modeling without the
    quadratic complexity of traditional attention mechanisms, making it
    suitable for long sequences and resource-constrained environments.

    Args:
        embed_dim (int): The dimension of the input and output embeddings.
        hidden_dim (int): The dimension of the hidden layers within the FFN and gate networks.
        num_heads (int): The number of heads for the LinearQueryExtractor.
                         `embed_dim` must be divisible by `num_heads`.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        # Extracts a global query vector from the sequence, summarizing its content.
        self.query_extractor = LinearQueryExtractor(embed_dim, num_heads)
        
        # Transforms the global query into an 'interactive context' vector.
        # This context provides additional global information to the gating mechanism.
        self.interactive_context = nn.Linear(embed_dim, embed_dim)
        
        # Gate network: Takes concatenated input (original token embedding,
        # expanded global query, expanded interactive context) and outputs
        # two gate values (input_gate and forget_gate) for each token.
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim), # Input: [x_token, global_query_expanded, interactive_context_expanded]
            nn.GELU(), # GELU activation for non-linearity
            nn.Linear(hidden_dim, embed_dim * 2) # Output: 2 * embed_dim (one for input_gate, one for forget_gate)
        )
        
        # Layer normalization applied after the residual connection and FFN.
        self.norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward network (FFN) applied after the gating mechanism.
        # This allows for further non-linear transformation of the gated output.
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Linear projection applied to the input `x` before being scaled by the forget gate.
        # This provides a transformed version of `x` for the forget gate to act upon.
        self.gated_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the Zarvan block.

        Args:
            x (torch.Tensor): Input tensor with positional encoding,
                              shape (Batch, Sequence, Embedding).

        Returns:
            torch.Tensor: Output tensor of the same shape (Batch, Sequence, Embedding),
                          after applying the selective copy through gating and FFN.
        """
        B, S, E = x.shape # B: Batch size, S: Sequence length, E: Embedding dimension
        
        # 1. Extract the global query vector 'q' from the input sequence.
        # This 'q' summarizes the entire sequence for the current block.
        # Shape: (B, E)
        q = self.query_extractor(x)
        
        # 2. Transform 'q' into an 'interactive context' vector.
        # This context provides a global, non-linear transformation of the query,
        # which can be used to modulate token-level gates.
        # Shape: (B, E)
        i_ctx = F.gelu(self.interactive_context(q))
        
        # 3. Expand the global query and interactive context vectors to match the sequence length.
        # This allows these global representations to interact with each token individually
        # within the gate network.
        # Shapes: (B, S, E) for both q_exp and i_ctx_exp
        q_exp = q.unsqueeze(1).expand(-1, S, -1)
        i_ctx_exp = i_ctx.unsqueeze(1).expand(-1, S, -1)
        
        # 4. Concatenate the original input token (x), the expanded global query (q_exp),
        # and the expanded interactive context (i_ctx_exp) along the embedding dimension.
        # This combined tensor provides all necessary information for the gate network
        # to make token-specific gating decisions.
        # Shape: (B, S, 3 * E)
        gate_input = torch.cat([x, q_exp, i_ctx_exp], dim=-1)
        
        # 5. Calculate the raw gate values from the gate network.
        # These raw values will be transformed into probabilities by sigmoid.
        # Shape: (B, S, 2 * E)
        gates = self.gate_net(gate_input)
        
        # 6. Split the raw gate values into 'input_gate' and 'forget_gate'.
        # Apply sigmoid activation to constrain values between 0 and 1.
        # The `input_gate` controls how much of the original input `x` is passed through.
        # The `forget_gate` controls how much of a transformed version of `x` is retained/updated.
        # Shapes: (B, S, E) for each gate
        input_gate, forget_gate = gates.chunk(2, dim=-1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        
        # 7. Apply the selective copy mechanism:
        # This is the core of Zarvan's token-wise update.
        # - `input_gate * x`: Selectively "copies" or retains parts of the original input token `x`.
        #   If `input_gate` is close to 1, the original `x` passes through. If close to 0, it's blocked.
        # - `forget_gate * self.gated_proj(x)`: Selectively "forgets" or updates parts
        #   of a linearly transformed version of `x`. This allows the block to learn
        #   what information from the previous layer's output to carry forward or modify.
        # The sum combines these two modulated streams.
        # Shape: (B, S, E)
        gated_x = input_gate * x + forget_gate * self.gated_proj(x)
        
        # 8. Pass the gated output through a Feed-Forward Network.
        # This FFN provides additional non-linear processing capacity after the gating.
        # Shape: (B, S, E)
        output = self.ffn(gated_x)
        
        # 9. Apply a residual connection (x + output) and layer normalization.
        # The residual connection helps in training deeper networks by allowing
        # gradients to flow more easily. Layer normalization stabilizes training.
        # Shape: (B, S, E)
        return self.norm(x + output)

class ZarvanForClassification(nn.Module):
    """
    A complete classifier model built using the Zarvan architecture.
    This model takes token IDs as input, embeds them, adds positional
    information, processes them through a ZarvanBlock, and then
    performs classification based on the aggregated sequence representation.

    Args:
        vocab_size (int): The size of the vocabulary (number of unique tokens).
        num_classes (int): The number of output classes for classification.
        seq_len (int): The maximum sequence length the model is designed to handle.
        embed_dim (int): The dimension of the token embeddings and internal representations.
        hidden_dim (int): The dimension of hidden layers within ZarvanBlock's FFN and gate networks.
        num_heads (int): The number of heads for the LinearQueryExtractor.
        padding_idx (int, optional): The index of the padding token in the vocabulary.
                                     Embeddings for this index will be zeroed out. Defaults to 0.
    """
    def __init__(self, vocab_size: int, num_classes: int, seq_len: int,
                 embed_dim: int, hidden_dim: int, num_heads: int,
                 padding_idx: int = 0):
        super().__init__()
        # Token embedding layer: Converts token IDs into dense vectors.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # Positional encoding: Adds information about token positions in the sequence.
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=seq_len)
        
        # The core Zarvan processing block.
        self.zarvan_block = ZarvanBlock(embed_dim, hidden_dim, num_heads)
        
        # Final linear layer for classification, mapping the aggregated sequence
        # representation to class logits.
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the Zarvan classifier.

        Args:
            x (torch.Tensor): Input tensor of token IDs, shape (Batch, Sequence).

        Returns:
            torch.Tensor: Output logits for each class, shape (Batch, NumClasses).
        """
        # 1. Embed tokens.
        # Input shape: (B, S) -> Output shape: (B, S, E)
        x = self.embedding(x)
        
        # 2. Add positional information to the embeddings.
        # Shape remains: (B, S, E)
        x = self.pos_encoder(x)
        
        # 3. Pass through the main Zarvan block.
        # This block processes the sequence and updates token representations.
        # Shape remains: (B, S, E)
        x = self.zarvan_block(x)
        
        # 4. Pool the output sequence into a single vector for classification.
        # Here, global average pooling is used, taking the mean across the sequence dimension.
        # (B, S, E) -> mean(dim=1) -> (B, E)
        z = x.mean(dim=1)
        
        # 5. Apply the final classification layer to get the logits.
        # (B, E) -> (B, C)
        return self.fc(z)

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Demonstrating the ZarvanForClassification model ---")

    # Define some dummy parameters for a demonstration
    BATCH_SIZE = 4
    SEQ_LEN = 512
    VOCAB_SIZE = 10000
    EMBED_DIM = 128
    HIDDEN_DIM = 256 # Hidden dimension for FFN and gate networks within ZarvanBlock
    NUM_HEADS = 4
    NUM_CLASSES = 5

    # Determine the device to run the example on (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning example on: {device}")

    # 1. Create the model instance
    model = ZarvanForClassification(
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        padding_idx=0 # Assuming 0 is the padding token ID
    ).to(device) # Move model to the selected device

    # 2. Create a dummy input batch of token IDs with shape (B, S)
    # Ensure token IDs are within the vocabulary range (1 to VOCAB_SIZE-1, excluding padding_idx)
    dummy_input = torch.randint(low=1, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN)).to(device)
    
    print(f"\nModel Architecture:\n{model}")
    print(f"\nInput shape: {dummy_input.shape} (Batch, Sequence)")

    # 3. Perform a forward pass
    try:
        # Measure inference time
        start_time = time.time()
        output = model(dummy_input)
        end_time = time.time()
        
        print(f"Output shape: {output.shape} (Batch, NumClasses)")
        print(f"Inference time for {BATCH_SIZE} samples: {end_time - start_time:.4f} seconds")
        
        # Assert the output shape is as expected
        assert output.shape == (BATCH_SIZE, NUM_CLASSES)
        print("\n✅ Model forward pass successful and output shape is correct!")
    except Exception as e:
        print(f"\n❌ An error occurred during the forward pass: {e}")

    print("\n--- End of ZarvanForClassification demonstration ---")
