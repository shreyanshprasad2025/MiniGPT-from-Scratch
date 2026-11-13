
import torch
import torch.nn.functional as F
import tiktoken

from model import MiniGPT


# =========================================
# 1. LOAD OUR TRAINED GPT
# =========================================

def load_model(checkpoint_path, device):

    # Load the saved checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    # Recreate the model architecture
    model = MiniGPT(checkpoint["config"])

    # Load learned parameters
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Move model to our device
    model = model.to(device)

    # Disable dropout
    model.eval()

    return model


# =========================================
# 2. TEXT GENERATION FUNCTION
# =========================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40
):

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if temperature < 0:
        raise ValueError(
            "Temperature cannot be negative"
        )

    device = next(model.parameters()).device

    context_length = model.config["context_length"]

    # Convert prompt into token IDs
    token_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    # Generate tokens one at a time
    for _ in range(max_new_tokens):

        # Keep only the latest context window
        input_context = input_ids[
            :, -context_length:
        ]

        # Forward pass
        logits = model(input_context)

        # Get predictions for the LAST token
        logits = logits[:, -1, :]

        # ---------------------------------
        # Greedy decoding
        # ---------------------------------

        if temperature == 0:

            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

        else:

            # -----------------------------
            # Temperature scaling
            # -----------------------------

            logits = logits / temperature

            # -----------------------------
            # Top-k filtering
            # -----------------------------

            if top_k is not None:

                k = min(
                    max(1, top_k),
                    logits.size(-1)
                )

                top_values, _ = torch.topk(
                    logits, k
                )

                threshold = top_values[:, -1:]

                logits = logits.masked_fill(
                    logits < threshold,
                    float("-inf")
                )

            # Convert scores to probabilities
            probabilities = F.softmax(
                logits,
                dim=-1
            )

            # Randomly select a token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

        # Append the new token
        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )

        # Stop if model generates end-of-text
        if next_token.item() == tokenizer.eot_token:
            break

    # Convert generated IDs back to text
    generated_text = tokenizer.decode(
        input_ids[0].tolist()
    )

    return generated_text


# =========================================
# 3. INTERACT WITH OUR OWN GPT
# =========================================

if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    tokenizer = tiktoken.get_encoding("gpt2")

    model = load_model(
        "checkpoints/minigpt_best_2000.pt",
        device
    )

    print("\nMiniGPT loaded successfully!")

    print("Enter a prompt to generate text.")

    print("Type 'exit' to stop.\n")

    while True:

        prompt = input("You: ")

        if prompt.lower().strip() == "exit":
            break

        if not prompt.strip():
            continue

        generated_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_k=40
        )

        print("\nMiniGPT:")
        print(generated_text)
        print()
