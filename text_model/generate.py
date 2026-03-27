"""
generate.py — LSTM Story Generation (v3)

Changes from v2:
  - Removed mid-generation section tag injection: it was causing
    the model to restart sentence patterns mid-output, producing
    "Jony can Jony can..." style repetition.
  - Added scenario/emotion validator with fuzzy suggestions so
    users get clear feedback on invalid inputs instead of silent
    <unk> fallback.
  - Repetition penalty now covers all recent tokens with a tiered
    penalty: heavier for filler words, lighter for content words.
  - Added hard sentence deduplication in post-processing to remove
    repeated sentences that slip through sampling.
  - Temperature lowered to 0.8 for more coherent output.
"""

import torch
import json
import re
import os
from datetime import datetime
from difflib import get_close_matches
from model import TinyLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_PATH = "logs/generation_log.json"

# ---------------------------------------------------------------------------
# Load vocabulary
# ---------------------------------------------------------------------------
with open("vocab.json", "r") as f:
    word2idx = json.load(f)

idx2word  = {i: w for w, i in word2idx.items()}
UNK_IDX   = word2idx.get("<unk>", 0)

KNOWN_SCENARIOS = sorted(
    k.replace("<scenario_", "").replace(">", "")
    for k in word2idx if k.startswith("<scenario_")
)
KNOWN_EMOTIONS = sorted(
    k.replace("<emotion_", "").replace(">", "")
    for k in word2idx if k.startswith("<emotion_")
)


def tok(word: str) -> int:
    return word2idx.get(word.lower(), UNK_IDX)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
checkpoint = torch.load("tiny_lstm.pth", map_location=device)
model = TinyLSTM(checkpoint["vocab_size"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_input(scenario: str, emotion: str) -> tuple[bool, str]:
    errors = []

    if scenario not in KNOWN_SCENARIOS:
        suggestions = get_close_matches(scenario, KNOWN_SCENARIOS, n=3, cutoff=0.5)
        msg = f"Unknown scenario: '{scenario}'"
        if suggestions:
            msg += f"\n  Did you mean: {', '.join(suggestions)}?"
        else:
            msg += f"\n  Known scenarios: {', '.join(KNOWN_SCENARIOS)}"
        errors.append(msg)

    if emotion not in KNOWN_EMOTIONS:
        suggestions = get_close_matches(emotion, KNOWN_EMOTIONS, n=3, cutoff=0.5)
        msg = f"Unknown emotion: '{emotion}'"
        if suggestions:
            msg += f"\n  Did you mean: {', '.join(suggestions)}?"
        else:
            msg += f"\n  Known emotions: {', '.join(KNOWN_EMOTIONS)}"
        errors.append(msg)

    if errors:
        return False, "\n".join(errors)
    return True, ""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_generation(name, scenario, emotion, story):
    os.makedirs("logs", exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "inputs": {"name": name, "scenario": scenario, "emotion": emotion},
        "output": story,
    }
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = []
    else:
        log_data = []
    log_data.append(record)
    with open(LOG_PATH, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\n[Log saved] → {LOG_PATH}")


# ---------------------------------------------------------------------------
# Filler words — penalised more aggressively for repetition
# ---------------------------------------------------------------------------
FILLER_WORDS = {
    "the", "a", "an", "is", "it", "in", "on", "to", "and",
    "feels", "feel", "very", "slowly", "softly", "kindly",
    "situation", "ends", "safely", "better", "today", "faces",
}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_with_context(
    scenario: str,
    emotion: str,
    max_words: int = 80,
    temperature: float = 0.8,
    top_k: int = 10,
) -> list[str]:
    """
    Prime the LSTM on scenario + emotion anchor tokens, then generate
    one token at a time with persistent hidden state.
    """
    anchor = [
        f"<scenario_{scenario}>",
        f"<emotion_{emotion}>",
        "<name>",
        "<start>",
    ]

    generated: list[str] = []
    state = None

    # Prime hidden state on anchor tokens
    for t in anchor:
        idx_t = torch.tensor([[tok(t)]]).to(device)
        with torch.no_grad():
            _, state = model(idx_t, state)

    last_token = "<start>"

    for step in range(max_words):
        idx_t = torch.tensor([[tok(last_token)]]).to(device)
        with torch.no_grad():
            output, state = model(idx_t, state)

        logits = output[0, -1].clone()

        # Tiered repetition penalty
        recent_set = set(generated[-20:])
        for word in recent_set:
            if word not in word2idx:
                continue
            penalty = 3.0 if word in FILLER_WORDS else 1.8
            logits[word2idx[word]] /= penalty

        # Encourage <end> after sufficient content
        end_idx = word2idx.get("<end>")
        if end_idx is not None and step > 55:
            logits[end_idx] *= 1.8

        probs = torch.softmax(logits / temperature, dim=0)
        top_probs, top_indices = torch.topk(probs, top_k)
        top_probs = top_probs / top_probs.sum()
        predicted_idx = top_indices[torch.multinomial(top_probs, 1)].item()
        next_word = idx2word[predicted_idx]

        if next_word == "<end>":
            break

        generated.append(next_word)
        last_token = next_word

    return generated


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def clean_story(tokens: list[str], name: str) -> str:
    text = " ".join(tokens)

    # Strip all control tags
    text = re.sub(r"<scenario_[^>]+>", "", text)
    text = re.sub(r"<emotion_[^>]+>",  "", text)
    text = re.sub(r"</?(?:start|feeling|event|coping|ending|end)>", "", text)
    text = re.sub(r"<name>", name.capitalize(), text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    # Capitalise + deduplicate sentences
    raw_sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 5]
    seen = set()
    sentences = []
    for s in raw_sentences:
        if s.lower().strip() not in seen:
            seen.add(s.lower().strip())
            sentences.append(s[0].upper() + s[1:] if s else s)

    return ". ".join(sentences) + "."


def enrich_emotion(story: str, name: str, emotion: str) -> str:
    closings = {
        "excited":     f"{name.capitalize()} feels proud and happy.",
        "nervous":     f"It is okay to feel nervous. {name.capitalize()} was brave.",
        "sad":         f"Feelings can change. {name.capitalize()} feels a little better now.",
        "angry":       f"Big feelings are okay. {name.capitalize()} took a deep breath.",
        "scared":      f"{name.capitalize()} was safe the whole time.",
        "worried":     f"Things worked out okay for {name.capitalize()}.",
        "overwhelmed": f"{name.capitalize()} took it one step at a time.",
        "shy":         f"{name.capitalize()} did it, even though it felt hard.",
    }
    closing = closings.get(emotion, "")
    if closing:
        story = story.rstrip(".") + ". " + closing
    return story


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\nKnown scenarios: {', '.join(KNOWN_SCENARIOS)}")
    print(f"Known emotions:  {', '.join(KNOWN_EMOTIONS)}\n")

    user_name = input("Enter child's name: ").strip()
    scenario  = input("Enter scenario: ").strip().lower().replace(" ", "_")
    emotion   = input("Enter emotion: ").strip().lower().replace(" ", "_")

    valid, error_msg = validate_input(scenario, emotion)
    if not valid:
        print(f"\n[Error]\n{error_msg}")
        exit(1)

    print(f"\nGenerating story for scenario='{scenario}', emotion='{emotion}'…\n")

    raw_tokens = generate_with_context(scenario, emotion)
    story      = clean_story(raw_tokens, user_name)
    story      = enrich_emotion(story, user_name, emotion)

    print("Generated Story:\n")
    print(story)

    log_generation(
        name=user_name,
        scenario=scenario,
        emotion=emotion,
        story=story,
    )