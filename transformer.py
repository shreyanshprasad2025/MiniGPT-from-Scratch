
import torch
import torch.nn as nn

from attention import MultiHeadAttention


# =========================================
# 1. Feed-Forward Network
# =========================================

class FeedForward(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.layers = nn.Sequential(

            # Expand the features
            nn.Linear(
                embedding_dim,
                4 * embedding_dim
            ),

            # Nonlinear activation
            nn.GELU(),

            # Return to original dimension
            nn.Linear(
                4 * embedding_dim,
                embedding_dim
            )
        )

    def forward(self, x):

        return self.layers(x)


# =========================================
# 2. Transformer Block
# =========================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length,
        dropout=0.1
    ):

        super().__init__()

        # Our attention layer from Step 24
        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            context_length
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            embedding_dim
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        # ===================================
        # PART 1: Attention
        # ===================================

        # Save original input
        shortcut = x

        # Normalize
        x = self.norm1(x)

        # Apply causal multi-head attention
        x = self.attention(x)

        # Apply dropout
        x = self.dropout(x)

        # Residual connection
        x = x + shortcut


        # ===================================
        # PART 2: Feed-Forward Network
        # ===================================

        # Save attention output
        shortcut = x

        # Normalize
        x = self.norm2(x)

        # Process each token
        x = self.feed_forward(x)

        # Apply dropout
        x = self.dropout(x)

        # Residual connection
        x = x + shortcut

        return x


# =========================================
# TEST OUR TRANSFORMER BLOCK
# =========================================

if __name__ == "__main__":

    torch.manual_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # Model configuration
    embedding_dim = 64
    num_heads = 4
    context_length = 4
    batch_size = 2

    # Simulate input embeddings
    x = torch.randn(
        batch_size,
        context_length,
        embedding_dim,
        device=device
    )

    print("\nInput shape:", x.shape)

    # Create our transformer block
    block = TransformerBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        context_length=context_length,
        dropout=0.1
    ).to(device)

    # Disable dropout for testing
    block.eval()

    # Forward pass
    with torch.no_grad():

        output = block(x)

    print("\nOutput shape:", output.shape)

    # Count trainable parameters
    total_params = sum(
        p.numel()
        for p in block.parameters()
        if p.requires_grad
    )

    print(
        "\nTrainable parameters:",
        total_params
    )

    # Check that shape remains the same
    assert x.shape == output.shape

    print("\nShape check: PASSED")

    print(
        "Input and output are different:",
        not torch.allclose(x, output)
    )

    print("\nTransformer block works!")
