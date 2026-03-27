import torch
import json
from datetime import datetime
import os

from model import TinyTransformer

device = torch.device("cpu")

os.makedirs("outputs", exist_ok=True)

with open("vocab.json") as f:
    vocab = json.load(f)

word2idx = vocab["word2idx"]
idx2word = {int(k): v for k, v in vocab["idx2word"].items()}

model = TinyTransformer(vocab["size"])
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()


def generate(prompt, max_len=60):

    tokens = prompt.split()

    ids = [word2idx[t] for t in tokens if t in word2idx]

    for _ in range(max_len):

        x = torch.tensor(ids).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)

        next_id = torch.argmax(logits[0, -1]).item()

        ids.append(next_id)

        if idx2word[next_id] == "<end>":
            break

    return " ".join(idx2word[i] for i in ids)


def save_log(prompt, output):

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": prompt,
        "generated_text": output
    }

    path = "outputs/generation_log.json"

    try:
        with open(path) as f:
            data = json.load(f)
    except:
        data = []

    data.append(record)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


prompt = "<scenario_doctor_visit> <emotion_scared> <start>"

story = generate(prompt)

print("\nGenerated Story:\n")
print(story)

save_log(prompt, story)
