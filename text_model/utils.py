import re

def preprocess_text(file_path):
    """
    Reads dataset and converts:
    - lowercase
    - [NAME] → <NAME>
    - [SCENARIO] dentist → <SCENARIO_dentist>
    """

    processed_text = ""

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_scenario = None

    for line in lines:
        line = line.strip()

        # Convert to lowercase
        line = line.lower()

        # Replace [name]
        line = line.replace("[name]", "<NAME>")

        # Replace scenario line
        if line.startswith("[scenario]"):
            scenario_name = line.replace("[scenario]", "").strip()
            current_scenario = f"<SCENARIO_{scenario_name}>"
            processed_text += current_scenario + "\n"
            continue

        # Skip [text] and [end] tags
        if line in ["[text]", "[end]"]:
            continue

        # Add normal story lines
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
