import torch
import json
import os
import re
from model import TinyTransformer, save_generation_output

device = torch.device("cpu")

TOKEN_PATTERN = re.compile(r"\b\w+\b")

# -----------------------
# Load vocabulary
# -----------------------

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab_data = json.load(f)

stoi = vocab_data["word2idx"]
itos = {int(k): v for k, v in vocab_data["idx2word"].items()}

vocab_size = vocab_data["size"]

print("Vocabulary size:", vocab_size)

unk_token = stoi.get("<UNK>", None)

# -----------------------
# Load model
# -----------------------

model = TinyTransformer(vocab_size)

checkpoint = torch.load("checkpoints/latest_model.pt", map_location=device)

model.load_state_dict(checkpoint["model_state"])

model.to(device)
model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"] + 1)

# -----------------------
# Sampling settings
# -----------------------

temperature = 0.8
top_k = 20
max_tokens = 60
repetition_penalty = 1.2


# -----------------------
# Top-k sampling
# -----------------------

def sample_top_k(logits, k=20):

    values, indices = torch.topk(logits, k)

    probs = torch.softmax(values, dim=-1)

    sampled = torch.multinomial(probs, 1)

    return indices[sampled].item()


# -----------------------
# Prompt builder
# -----------------------

def build_prompt(child_name, scenario, emotion):

    # keeps name active in generation
    prompt = f"{child_name} feels {emotion} because of {scenario}. {child_name}"

    return prompt.lower()


# -----------------------
# Generate text
# -----------------------

def generate(child_name, scenario, emotion):

    prompt = build_prompt(child_name, scenario, emotion)

    print("\nPrompt used for generation:\n", prompt)

    tokens = TOKEN_PATTERN.findall(prompt)

    token_ids = []

    for t in tokens:
        if t in stoi:
            token_ids.append(stoi[t])
        elif unk_token is not None:
            token_ids.append(unk_token)

    if len(token_ids) == 0:
        raise ValueError("Prompt contains no known tokens.")

    input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

    for _ in range(max_tokens):

        with torch.no_grad():
            logits = model(input_ids)

        logits = logits[:, -1, :] / temperature

        # repetition penalty
        for token in set(input_ids.squeeze().tolist()):
            logits[0, token] /= repetition_penalty

        next_token = sample_top_k(logits.squeeze(), top_k)

        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]], device=device)],
            dim=1
        )

        if itos[next_token] == "<EOS>":
            break

    output_tokens = input_ids.squeeze().tolist()

    words = [itos.get(i, "") for i in output_tokens]

    text = " ".join(words)

    return (
        text.replace("<BOS>", "")
        .replace("<EOS>", "")
        .replace("<PAD>", "")
        .replace("<UNK>", child_name)
        .strip()
    )


# -----------------------
# Run generation
# -----------------------

child_name = input("Child Name: ")
scenario = input("Scenario: ")
emotion = input("Emotion: ")

generated = generate(child_name, scenario, emotion)

print("\nGenerated Story:\n")
print(generated)

# -----------------------
# Save output
# -----------------------

os.makedirs("outputs", exist_ok=True)

save_generation_output(
    f"{child_name} | {scenario} | {emotion}",
    generated
)

