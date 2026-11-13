
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length
    ):
        super().__init__()

        # Ensure equal-sized attention heads
        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.head_dim = embedding_dim // num_heads

        # Query, Key and Value projections
        self.W_query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.W_key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.W_value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        # Combine information from all heads
        self.out_proj = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # Causal mask: block future tokens
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(
                    context_length,
                    context_length,
                    dtype=torch.bool
                ),
                diagonal=1
            )
        )


    def forward(self, x, return_weights=False):

        # B = batch size
        # T = number of tokens
        # C = embedding dimension
        B, T, C = x.shape

        # ===================================
        # 1. Calculate Q, K and V
        # ===================================

        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        # ===================================
        # 2. Split into multiple heads
        # ===================================

        Q = Q.reshape(
            B, T, self.num_heads, self.head_dim
        )

        K = K.reshape(
            B, T, self.num_heads, self.head_dim
        )

        V = V.reshape(
            B, T, self.num_heads, self.head_dim
        )

        # Shape: (B, num_heads, T, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ===================================
        # 3. Calculate attention scores
        # ===================================

        scores = Q @ K.transpose(-2, -1)

        # Scale attention scores
        scores = scores / (self.head_dim ** 0.5)

        # ===================================
        # 4. Apply causal mask
        # ===================================

        mask = self.mask[:T, :T]

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        # ===================================
        # 5. Convert scores to weights
        # ===================================

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # ===================================
        # 6. Calculate context vectors
        # ===================================

        context = attention_weights @ V

        # ===================================
        # 7. Combine all attention heads
        # ===================================

        context = context.transpose(1, 2)

        context = context.contiguous().reshape(
            B, T, C
        )

        # ===================================
        # 8. Final output projection
        # ===================================

        output = self.out_proj(context)

        if return_weights:
            return output, attention_weights

        return output


# =========================================
# TEST OUR ATTENTION LAYER
# =========================================

if __name__ == "__main__":

    torch.manual_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # Our model configuration
    embedding_dim = 64
    num_heads = 4
    context_length = 4
    batch_size = 2

    # Example input embeddings
    x = torch.randn(
        batch_size,
        context_length,
        embedding_dim,
        device=device
    )

    print("\nInput shape:", x.shape)

    # Create attention layer
    attention = MultiHeadAttention(
        embedding_dim,
        num_heads,
        context_length
    ).to(device)

    # Run the attention mechanism
    output, weights = attention(
        x,
        return_weights=True
    )

    print("\nNumber of heads:", num_heads)

    print("Features per head:", attention.head_dim)

    print("\nAttention weights shape:")
    print(weights.shape)

    print("\nAttention weights - Batch 1, Head 1:")
    print(weights[0, 0])

    print("\nAttention output shape:")
    print(output.shape)

    print("\nSum of attention rows:")
    print(weights[0, 0].sum(dim=-1))
