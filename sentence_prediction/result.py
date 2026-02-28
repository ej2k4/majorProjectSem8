import torch
import pickle
from model import Encoder, Decoder, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# CONFIGURABLE LISTS
# -------------------
PLACES = ["park", "school", "garden", "shop", "home"]

# -------------------
# Load Vocabulary
# -------------------
with open("vocab.pkl", "rb") as f:
    word2idx = pickle.load(f)

idx2word = {i: w for w, i in word2idx.items()}

def numericalize(sentence):
    return [word2idx.get(word, word2idx["<unk>"]) for word in sentence.lower().split()]

# -------------------
# Load Model
# -------------------
vocab_size = len(word2idx)

encoder = Encoder(vocab_size, 128, 256)
decoder = Decoder(vocab_size, 128, 256)
model = Seq2Seq(encoder, decoder, device).to(device)

model.load_state_dict(torch.load("asd_model.pt", map_location=device))
model.eval()

# -------------------
# Simple Greedy Prediction
# -------------------
def predict(sentence, max_len=15):
    tokens = numericalize(sentence)
    tokens = tokens[:max_len]
    tokens += [word2idx["<pad>"]] * (max_len - len(tokens))

    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden = model.encoder(src_tensor)

        input_token = torch.tensor([word2idx["<sos>"]]).to(device)

        result = []

        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            top1 = output.argmax(1)

            if top1.item() == word2idx["<eos>"]:
                break

            result.append(idx2word.get(top1.item(), ""))

            input_token = top1

    return " ".join(result)

# -------------------
# Controlled Variation Generator
# -------------------
def generate_variations(base_sentence):
    variations = [base_sentence]

    # Only generate variations for "I want to go to the X"
    if "to the" in base_sentence:
        prefix = base_sentence.split("to the")[0]

        for place in PLACES:
            variations.append(prefix + "to the " + place)

    # Remove duplicates
    variations = list(dict.fromkeys(variations))

    return variations[:3]   # show max 3 suggestions

# -------------------
# Main Loop
# -------------------
while True:
    text = input("Enter fragmented sentence: ")

    base_prediction = predict(text)

    suggestions = generate_variations(base_prediction)

    print("\nSuggestions:")
    for i, s in enumerate(suggestions):
        print(f"{i+1}. {s}")