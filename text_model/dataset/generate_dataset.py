import random

scenarios = {
    "dentist": [
        "The dentist is kind and smiles warmly.",
        "The chair moves slowly and gently.",
        "There is a bright light to see the teeth clearly.",
        "The dentist counts the teeth carefully.",
        "You can raise your hand anytime.",
        "You will get a shiny sticker after the visit."
    ],
    "haircut": [
        "The chair is big and safe.",
        "The scissors make a soft snip sound.",
        "The barber talks in a calm voice.",
        "Hair grows back again.",
        "You can look in the mirror.",
        "You will look neat and fresh."
    ],
    "doctor_visit": [
        "The doctor listens to your heartbeat.",
        "The room smells clean.",
        "The doctor speaks gently.",
        "The checkup will be quick.",
        "You can sit on a soft bed.",
        "You will feel proud afterward."
    ],
}

emotions = [
    "feels a little nervous",
    "feels curious",
    "feels brave",
    "feels unsure",
    "feels excited",
    "takes a deep breath",
    "holds mom's hand",
    "remembers to stay calm",
]

endings = [
    "Everything goes well.",
    "It is not scary at all.",
    "You did a great job.",
    "You are very brave.",
    "It finishes quickly.",
    "You feel proud and happy."
]

names = ["<name>"]

def generate_story(scenario, details):
    emotion = random.choice(emotions)
    chosen_details = random.sample(details, 4)
    ending = random.choice(endings)

    story = f"<scenario_{scenario}> <name> {emotion}. "
    story += " ".join(chosen_details) + " "
    story += ending
    story += " <end>"

    return story


def main():
    entries = []

    for scenario, details in scenarios.items():
        for _ in range(100):  # 100 per scenario
            entries.append(generate_story(scenario, details))

    random.shuffle(entries)

    with open("stories.txt", "w", encoding="utf-8") as f:
        for story in entries:
            f.write(story + "\n")

    print(f"Generated {len(entries)} stories.")


if __name__ == "__main__":
    main()
