
import random

from instruction_data import (
    training_examples,
    test_examples
)

rng = random.Random(42)

# Start with our original questions
knowledge_examples = training_examples.copy()

# Add more conversational and factual knowledge
extra_qa = [
    ("Who created you?",
     "I was built as a MiniGPT project using Python and PyTorch."),

    ("Who built you?",
     "I was built as a MiniGPT project using Python and PyTorch."),

    ("What can you do?",
     "I can answer some basic questions about Python, AI and mathematics."),

    ("Are you ChatGPT?",
     "No, I am MiniGPT, a small language model built from scratch."),

    ("What is a variable?",
     "A variable is a name used to store a value in a program."),

    ("What is a function?",
     "A function is a reusable block of code that performs a task."),

    ("What is a list in Python?",
     "A list is an ordered collection of items in Python."),

    ("What is a dictionary in Python?",
     "A dictionary stores information as key-value pairs."),

    ("What is a loop?",
     "A loop repeats a block of code."),

    ("What is a class?",
     "A class is a blueprint for creating objects."),

    ("What is an algorithm?",
     "An algorithm is a sequence of steps for solving a problem."),

    ("What is a Transformer?",
     "A Transformer is a neural network architecture that uses attention."),

    ("What is attention in AI?",
     "Attention helps a model focus on relevant information."),

    ("What is a loss function?",
     "A loss function measures the error in a model's predictions."),

    ("What is backpropagation?",
     "Backpropagation calculates gradients used to train neural networks."),

    ("What is the capital of Germany?",
     "Berlin is the capital of Germany."),

    ("What is the capital of Italy?",
     "Rome is the capital of Italy."),

    ("What is the capital of China?",
     "Beijing is the capital of China."),

    ("What is the capital of Australia?",
     "Canberra is the capital of Australia."),

    ("How many days are in a week?",
     "There are seven days in a week."),

    ("How many months are in a year?",
     "There are twelve months in a year."),

    ("What is oxygen?",
     "Oxygen is a chemical element needed by humans to breathe.")
]

for question, answer in extra_qa:

    knowledge_examples.append({
        "question": question,
        "answer": answer
    })


# Add a few genuine variations in phrasing

variations = [
    ("Could you introduce yourself?",
     "I am MiniGPT, a small AI language model."),

    ("Good morning",
     "Good morning! How can I help you?"),

    ("Explain Python programming",
     "Python is a popular programming language."),

    ("Define AI",
     "AI stands for artificial intelligence."),

    ("Tell me India's capital",
     "New Delhi is the capital of India."),

    ("What does a GPU do?",
     "A GPU processes many numerical operations in parallel."),

    ("Explain what a function does",
     "A function is a reusable block of code that performs a task.")
]

for question, answer in variations:

    knowledge_examples.append({
        "question": question,
        "answer": answer
    })


# ====================================
# Generate mathematics examples
# ====================================

math_train = []
math_test = []

# Some questions already appeared in v1.
# Keep those operand pairs in training.
known_pairs = {
    (2, 2),
    (3, 5),
    (4, 10)
}

for a in range(30):

    for b in range(30):

        # Hold out complete operand pairs,
        # not just different phrasings.
        pair = (min(a, b), max(a, b))

        held_out = (
            (pair[0] * 31 + pair[1]) % 7 == 0
            and pair not in known_pairs
        )

        destination = math_test if held_out else math_train

        addition_questions = [
            f"What is {a} plus {b}?",
            f"Calculate {a} + {b}",
            f"Add {a} and {b}",
            f"How much is {a} + {b}?"
        ]

        for question in addition_questions:

            destination.append({
                "question": question,
                "answer": str(a + b)
            })

        # Non-negative subtraction examples
        if a >= b:

            subtraction_questions = [
                f"What is {a} minus {b}?",
                f"Calculate {a} - {b}"
            ]

            for question in subtraction_questions:

                destination.append({
                    "question": question,
                    "answer": str(a - b)
                })


# ====================================
# Combine and balance training data
# ====================================

# Repeat knowledge examples to prevent
# mathematics from dominating training.
train_v2 = math_train + knowledge_examples * 35

rng.shuffle(train_v2)

# Keep the original unseen question phrasings
# and add held-out mathematics questions.
test_v2 = test_examples + rng.sample(
    math_test,
    min(100, len(math_test))
)


if __name__ == "__main__":

    print("Knowledge examples:", len(knowledge_examples))
    print("Math training examples:", len(math_train))
    print("Combined training samples:", len(train_v2))
    print("Test samples:", len(test_v2))

    print("\nExample:")

    example = rng.choice(train_v2)

    print("Question:", example["question"])
    print("Answer:", example["answer"])
