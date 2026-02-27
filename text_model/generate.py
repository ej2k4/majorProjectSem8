# # import torch
# # from model import TinyLSTM
# # from utils import tokenize, build_vocab

# # def generate(model, scenario_token, word2idx, idx2word, length=80):
# #     model.eval()

# #     input_word = torch.tensor([[word2idx[scenario_token]]])
# #     result = [scenario_token]

# #     hidden = None

# #     for _ in range(length):
# #         output, hidden = model(input_word, hidden)
# #         probs = torch.softmax(output[0, -1], dim=0)
# #         next_word_idx = torch.argmax(probs).item()
# #         next_word = idx2word[next_word_idx]

# #         result.append(next_word)
# #         input_word = torch.tensor([[next_word_idx]])

# #     return " ".join(result)


# import torch
# from model import TinyLSTM
# from utils import tokenize, build_vocab

# # Reload dataset for vocab
# with open("dataset/stories.txt", "r") as f:
#     text = f.read()

# tokens = tokenize(text)
# word2idx, idx2word = build_vocab(tokens)

# vocab_size = len(word2idx)

# model = TinyLSTM(vocab_size)
# model.load_state_dict(torch.load("tiny_lstm.pth"))
# model.eval()

# def generate(seed_text, max_words=30):
#     words = seed_text.lower().split()
#     state = None

#     for _ in range(max_words):
#         input_ids = torch.tensor([[word2idx.get(w, 0) for w in words[-5:]]])
#         output, state = model(input_ids, state)
#         last_word_logits = output[0, -1]
#         predicted_idx = torch.argmax(last_word_logits).item()
#         next_word = idx2word[predicted_idx]
#         words.append(next_word)

#     return " ".join(words)

# print(generate("<scenario_dentist> arjun"))


# import torch
# import json
# from model import TinyLSTM

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load vocab
# with open("vocab.json", "r") as f:
#     word2idx = json.load(f)

# idx2word = {i: w for w, i in word2idx.items()}

# # Load model
# checkpoint = torch.load("tiny_lstm.pth", map_location=device)

# model = TinyLSTM(checkpoint["vocab_size"]).to(device)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()

# def generate(seed_text, max_words=60):
#     words = seed_text.lower().split()
#     state = None

#     for _ in range(max_words):
#         input_ids = torch.tensor(
#             [[word2idx.get(w, word2idx.get("<unk>", 0)) for w in words[-10:]]]
#         ).to(device)

#         output, state = model(input_ids, state)

#         logits = output[0, -1]

#         #  Repetition penalty
#         for word in set(words[-15:]):  # penalize recent words
#             if word in word2idx:
#                 logits[word2idx[word]] /= 1.5

#         temperature = 0.75
#         probs = torch.softmax(logits / temperature, dim=0)

#         #  Top-k sampling
#         top_k = 10
#         top_probs, top_indices = torch.topk(probs, top_k)
#         top_probs = top_probs / torch.sum(top_probs)

#         predicted_idx = top_indices[torch.multinomial(top_probs, 1)].item()
#         next_word = idx2word[predicted_idx]

#         if next_word == "<end>":
#             break

#         words.append(next_word)

#     return " ".join(words)



# if __name__ == "__main__":
#     user_name = input("Enter a name: ").strip().lower()
#     scenario = input("Enter a scenario: ").strip().lower().replace(" ", "_")
#     emotion = input("Enter current emotion: ").strip().lower().replace(" ", "_")
#     scenario = scenario.replace(" ", "_")

#     seed_text = f"<SCENARIO_{scenario}> <EMOTION_{emotion}> <NAME>"
#     story = generate(seed_text)

#     story = story.replace("<NAME>", user_name)
#     story = story.replace(f"<SCENARIO_{scenario}>", "")
#     story = story.replace(f"<EMOTION_{emotion}>", "")
#     story = story.replace("<end>", "")
#     story = story.strip()


#     print("\nGenerated Story:\n")
#     print(story)



import torch
import json
import re
from model import TinyLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load vocabulary
# -----------------------
with open("vocab.json", "r") as f:
    word2idx = json.load(f)

idx2word = {i: w for w, i in word2idx.items()}

# -----------------------
# Load model
# -----------------------
checkpoint = torch.load("tiny_lstm.pth", map_location=device)

model = TinyLSTM(checkpoint["vocab_size"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# -----------------------
# Story Generation
# -----------------------
def generate(seed_text, max_words=80):
    words = seed_text.lower().split()
    state = None

    for _ in range(max_words):
        input_ids = torch.tensor(
            [[word2idx.get(w, word2idx.get("<unk>", 0)) for w in words[-10:]]]
        ).to(device)

        output, state = model(input_ids, state)

        logits = output[0, -1]

        # Repetition penalty
        for word in set(words[-15:]):
            if word in word2idx:
                logits[word2idx[word]] /= 1.5

        temperature = 0.8
        probs = torch.softmax(logits / temperature, dim=0)

        # Top-k sampling
        top_k = 10
        top_probs, top_indices = torch.topk(probs, top_k)
        top_probs = top_probs / torch.sum(top_probs)

        predicted_idx = top_indices[torch.multinomial(top_probs, 1)].item()
        next_word = idx2word[predicted_idx]

        if next_word == "<end>":
            break

        words.append(next_word)

    return " ".join(words)


# -----------------------
# Emotional Enrichment Layer
# -----------------------
def enrich_emotion(story, name, emotion):
    emotional_lines = {
        "excited": [
            f"{name.capitalize()} feels butterflies of excitement inside.",
            f"{name.capitalize()} cannot wait to see what happens next."
        ],
        "nervous": [
            f"{name.capitalize()}'s heart beats a little faster.",
            "It is okay to feel nervous sometimes."
        ],
        "sad": [
            f"{name.capitalize()}'s eyes feel a little heavy.",
            "It is okay to feel sad, and feelings can change."
        ],
        "angry": [
            f"{name.capitalize()}'s hands feel tight for a moment.",
            "Taking slow deep breaths can help big feelings."
        ],
        "scared": [
            f"{name.capitalize()} feels a small shiver inside.",
            "It is safe, and grown-ups are there to help."
        ]
    }

    if emotion in emotional_lines:
        extra = " ".join(emotional_lines[emotion])
        story = extra + " " + story

    return story


# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    user_name = input("Enter a name: ").strip().lower()
    scenario = input("Enter a scenario: ").strip().lower().replace(" ", "_")
    emotion = input("Enter current emotion: ").strip().lower().replace(" ", "_")

    # Improved seed structure
    seed_text = f"<scenario_{scenario}> <emotion_{emotion}> <name> </start>"

    story = generate(seed_text)

    # Replace name
    story = story.replace("<name>", user_name)

    # Remove scenario & emotion tags
    story = story.replace(f"<scenario_{scenario}>", "")
    story = story.replace(f"<emotion_{emotion}>", "")

    # Remove ALL remaining tags
    story = re.sub(r"<.*?>", "", story)

    # Clean extra spaces
    story = re.sub(r"\s+", " ", story).strip()

    # Add emotional richness
    story = enrich_emotion(story, user_name, emotion)

    # Improve formatting
    sentences = [s.strip().capitalize() for s in story.split(".") if s.strip()]
    story = ". ".join(sentences) + "."

    print("\nGenerated Story:\n")
    print(story)