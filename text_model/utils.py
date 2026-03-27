import re


def preprocess_text(file_path):
    """
    Converts dataset tags:
      [NAME]          -> <NAME>
      [SCENARIO] x   -> <SCENARIO_x>
      [EMOTION] x    -> <EMOTION_x>
    """
    processed_text = ""

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().lower()

        line = line.replace("[name]", "<NAME>")

        if line.startswith("[scenario]"):
            scenario_name = line.replace("[scenario]", "").strip()
            processed_text += f"<SCENARIO_{scenario_name}>\n"
            continue

        if line.startswith("[emotion]"):
            emotion_name = line.replace("[emotion]", "").strip()
            processed_text += f"<EMOTION_{emotion_name}>\n"
            continue

        if line in ["[text]", "[end]"]:
            continue

        if line:
            processed_text += line + "\n"

    return processed_text


def tokenize(text):
    text = text.lower()
    text = re.sub(r"\[name\]", "<name>", text)
    text = re.sub(r"\[scenario\]\s*(\w+)", r"<scenario_\1>", text)
    return text.split()


def build_vocab(tokens):
    vocab = sorted(set(tokens))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word