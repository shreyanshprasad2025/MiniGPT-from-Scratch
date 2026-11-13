
import torch
import torch.nn as nn

from transformer import TransformerBlock


# =========================================
# 1. MODEL CONFIGURATION
# =========================================

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 128,
    "embedding_dim": 64,
    "num_heads": 4,
    "num_layers": 4,
    "dropout": 0.1
}


# =========================================
# 2. COMPLETE GPT MODEL
# =========================================

class MiniGPT(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config

        # ---------------------------------
        # Token embeddings
        # ---------------------------------

        self.token_embedding = nn.Embedding(
            config["vocab_size"],
            config["embedding_dim"]
        )

        # ---------------------------------
        # Positional embeddings
        # ---------------------------------

        self.position_embedding = nn.Embedding(
            config["context_length"],
            config["embedding_dim"]
        )

        # Embedding dropout
        self.embedding_dropout = nn.Dropout(
            config["dropout"]
        )

        # ---------------------------------
        # Stack Transformer blocks
        # ---------------------------------

        self.transformer_blocks = nn.Sequential(

            *[
                TransformerBlock(
                    embedding_dim=config["embedding_dim"],
                    num_heads=config["num_heads"],
                    context_length=config["context_length"],
                    dropout=config["dropout"]
                )
                for _ in range(config["num_layers"])
            ]
        )

        # ---------------------------------
        # Final normalization
        # ---------------------------------

        self.final_norm = nn.LayerNorm(
            config["embedding_dim"]
        )

        # ---------------------------------
        # Output prediction layer
        # ---------------------------------

        self.output_head = nn.Linear(
            config["embedding_dim"],
            config["vocab_size"],
            bias=False
        )

        # Weight tying:
        # Share embedding and output weights
        self.output_head.weight = (
            self.token_embedding.weight
        )

        # Initialize model parameters
        self.apply(self._init_weights)


    # =====================================
    # 3. INITIALIZE MODEL WEIGHTS
    # =====================================

    def _init_weights(self, module):

        if isinstance(
            module,
            (nn.Linear, nn.Embedding)
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if (
                isinstance(module, nn.Linear)
                and module.bias is not None
            ):

                nn.init.zeros_(module.bias)


    # =====================================
    # 4. FORWARD PASS
    # =====================================

    def forward(self, input_ids):

        # Input shape: (batch_size, seq_length)
        B, T = input_ids.shape

        if T > self.config["context_length"]:

            raise ValueError(
                "Input exceeds model context length"
            )

        # ---------------------------------
        # Token embeddings
        # ---------------------------------

        token_embeddings = self.token_embedding(
            input_ids
        )

        # ---------------------------------
        # Positional embeddings
        # ---------------------------------

        positions = torch.arange(
            T,
            device=input_ids.device
        )

        position_embeddings = (
            self.position_embedding(positions)
        )

        # Combine embeddings
        x = token_embeddings + position_embeddings

        x = self.embedding_dropout(x)

        # ---------------------------------
        # Transformer blocks
        # ---------------------------------

        x = self.transformer_blocks(x)

        # ---------------------------------
        # Final normalization
        # ---------------------------------

        x = self.final_norm(x)

        # ---------------------------------
        # Predict vocabulary logits
        # ---------------------------------

        logits = self.output_head(x)

        return logits


# =========================================
# 5. TEST OUR COMPLETE GPT MODEL
# =========================================

if __name__ == "__main__":

    torch.manual_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # Create our GPT model
    model = MiniGPT(GPT_CONFIG).to(device)

    # Print configuration
    print("\nModel configuration:")

    for key, value in GPT_CONFIG.items():
        print(f"{key}: {value}")

    # Create example token IDs
    input_ids = torch.tensor(
        [
            [8001, 9542, 4430, 318],
            [9542, 4430, 318, 5609]
        ],
        device=device
    )

    print("\nInput shape:", input_ids.shape)

    # Forward pass
    model.eval()

    with torch.no_grad():

        logits = model(input_ids)

    print("Output shape:", logits.shape)

    # Count trainable parameters
    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTotal trainable parameters:",
        f"{total_params:,}"
    )

    print(
        "Number of transformer blocks:",
        len(model.transformer_blocks)
    )

    print("\nMiniGPT successfully created!")
