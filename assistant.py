
import torch
import tiktoken

from pathlib import Path
from tokenizers import Tokenizer
from generate import load_model


# =====================================
# 1. LOAD THE CORRECT TOKENIZER
# =====================================

def load_tokenizer(model):

    vocab_size = model.config["vocab_size"]

    # Our custom V3 tokenizer
    if vocab_size == 8192:

        return Tokenizer.from_file(
            "data/v3/tokenizer.json"
        )

    # Older V2 tokenizer
    return tiktoken.get_encoding("gpt2")


# =====================================
# 2. ENCODE TEXT
# =====================================

def encode(tokenizer, text):

    result = tokenizer.encode(text)

    if hasattr(result, "ids"):
        return result.ids

    return result


# =====================================
# 3. GET END-OF-TEXT TOKEN
# =====================================

def get_eot_id(tokenizer):

    if hasattr(tokenizer, "eot_token"):
        return tokenizer.eot_token

    return tokenizer.token_to_id(
        "<|endoftext|>"
    )


# =====================================
# 4. GENERATE AN ANSWER
# =====================================

@torch.no_grad()
def ask(
    model,
    tokenizer,
    question,
    max_new_tokens=100
):

    model.eval()

    device = next(model.parameters()).device

    context_length = model.config["context_length"]

    # Same format used during training
    prompt = f"User: {question}\nAssistant:"

    prompt_ids = encode(tokenizer, prompt)

    if len(prompt_ids) >= context_length:
        return "Your question is too long."

    x = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    answer_ids = []

    eot_id = get_eot_id(tokenizer)

    limit = min(
        max_new_tokens,
        context_length - len(prompt_ids)
    )

    # Generate one token at a time
    for _ in range(limit):

        logits = model(x)[:, -1, :]

        # Select the most likely token
        next_token = logits.argmax(
            dim=-1,
            keepdim=True
        )

        token_id = next_token.item()

        # Stop when the answer is complete
        if token_id == eot_id:
            break

        answer_ids.append(token_id)

        # Add new token to context
        x = torch.cat(
            [x, next_token],
            dim=1
        )

    return tokenizer.decode(
        answer_ids
    ).strip()


# =====================================
# 5. TERMINAL TEST
# =====================================

if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = (
        "checkpoints/minigpt_assistant_v3.pt"
    )

    if not Path(checkpoint).exists():

        raise FileNotFoundError(
            "Train Assistant V3 first."
        )

    model = load_model(checkpoint, device)

    tokenizer = load_tokenizer(model)

    print("MiniGPT V3 is ready!")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer = ask(
            model,
            tokenizer,
            question
        )

        print(
            "\nMiniGPT:",
            answer or "[Empty answer]",
            "\n"
        )
