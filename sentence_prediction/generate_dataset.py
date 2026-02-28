import random
import pandas as pd
subjects = ["I"]
verbs = ["want", "like", "need", "see", "play", "eat", "go"]
places = ["park", "school", "home", "shop", "garden"]
objects = ["water", "toy", "food", "book", "ball"]
emotions = ["happy", "sad", "angry", "scared"]
people = ["friend", "brother", "sister", "teacher", "mom"]

def pattern_want_place():
    place = random.choice(places)
    return f"I want to go to the {place}"

def pattern_feel_emotion():
    emotion = random.choice(emotions)
    place = random.choice(places)
    return f"I feel {emotion} about {place}"

def pattern_friend_action():
    person = random.choice(people)
    obj = random.choice(objects)
    return f"My {person} took my {obj}"

def pattern_angry_reason():
    person = random.choice(people)
    obj = random.choice(objects)
    return f"I am angry because my {person} broke my {obj}"

def pattern_like_play():
    person = random.choice(people)
    return f"I like to play with my {person}"

def pattern_need_object():
    obj = random.choice(objects)
    return f"I need {obj}"

patterns = [
    pattern_want_place,
    pattern_feel_emotion,
    pattern_friend_action,
    pattern_angry_reason,
    pattern_like_play,
    pattern_need_object
]

REMOVE_WORDS = {
    "i", "am", "is", "are", "to", "the", "my",
    "because", "about", "with"
}

def fragment(sentence):
    words = sentence.lower().split()
    fragmented = [w for w in words if w not in REMOVE_WORDS]
    return " ".join(fragmented)

data = []

for _ in range(30000):
    pattern_func = random.choice(patterns)
    correct = pattern_func()
    fragmented = fragment(correct)
    data.append([fragmented, correct])

df = pd.DataFrame(data, columns=["fragmented_input", "corrected_output"])
df.to_csv("asd_dataset.csv", index=False)

print("Dataset generated: 30000 samples")