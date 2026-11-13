
import torch
import torch.nn.functional as F
import tiktoken

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from generate import load_model
from instruction_data_v2 import train_v2


# ====================================
# 1. Configuration
# ====================================

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 8
max_steps = 1500
learning_rate = 5e-4

tokenizer = tiktoken.get_encoding("gpt2")

print("Device:", device)

model = load_model(
    "checkpoints/minigpt_best_2000.pt",
    device
)

model.train()


# ====================================
# 2. Prepare instruction dataset
# ====================================

class InstructionDataset(Dataset):

    def __init__(self, examples):

        self.samples = []

        # Avoid inconsistent answer formats
        # for arithmetic questions from v1.
        arithmetic_answers = {
            "2 plus 2 equals 4.": "4",
            "5 plus 3 equals 8.": "8",
            "10 minus 4 equals 6.": "6"
        }

        for example in examples:

            question = example["question"]
            answer = example["answer"]

            answer = arithmetic_answers.get(
                answer, answer
            )

            prompt = f"User: {question}\nAssistant:"

            prompt_ids = tokenizer.encode(prompt)

            answer_ids = tokenizer.encode(
                " " + answer
            )

            answer_ids.append(tokenizer.eot_token)

            ids = prompt_ids + answer_ids

            inputs = ids[:-1]
            targets = ids[1:]

            # Only calculate loss on answers
            targets[:len(prompt_ids) - 1] = (
                [-100] * (len(prompt_ids) - 1)
            )

            if len(inputs) > model.config["context_length"]:
                raise ValueError("Example too long")

            self.samples.append((
                torch.tensor(inputs, dtype=torch.long),
                torch.tensor(targets, dtype=torch.long)
            ))

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        return self.samples[index]


# ====================================
# 3. Dynamic padding
# ====================================

def collate_batch(batch):

    inputs, targets = zip(*batch)

    inputs = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=tokenizer.eot_token
    )

    targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=-100
    )

    return inputs, targets


dataset = InstructionDataset(train_v2)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch
)

print("Training samples:", len(dataset))
print("Batches:", len(loader))


# ====================================
# 4. Optimizer
# ====================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01
)


# ====================================
# 5. Train our assistant
# ====================================

train_iter = iter(loader)

running_loss = 0.0
running_steps = 0

print("\n--- Training Assistant V2 ---")

for step in range(1, max_steps + 1):

    try:

        inputs, targets = next(train_iter)

    except StopIteration:

        train_iter = iter(loader)
        inputs, targets = next(train_iter)

    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad(set_to_none=True)

    # Forward pass
    logits = model(inputs)

    # Answer-only cross-entropy
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100
    )

    # Backpropagation
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    running_loss += loss.item()
    running_steps += 1

    if step % 100 == 0 or step == 1:

        print(
            f"Step {step}/{max_steps} | "
            f"Answer loss: "
            f"{running_loss / running_steps:.4f}"
        )

        running_loss = 0.0
        running_steps = 0

    # Save progress periodically
    if step % 300 == 0 or step == max_steps:

        Path("checkpoints").mkdir(exist_ok=True)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "config": model.config,
                "step": step,
                "training_type": "assistant_v2"
            },
            "checkpoints/minigpt_assistant_v2.pt"
        )

        print("Checkpoint saved!")


print("\n--- Training Complete ---")
print("Assistant V2 saved successfully!")
