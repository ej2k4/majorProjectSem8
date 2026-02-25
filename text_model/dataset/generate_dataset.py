# import random

# scenarios = {
#     "dentist": [
#         "The dentist is kind and smiles warmly.",
#         "The chair moves slowly and gently.",
#         "There is a bright light to see the teeth clearly.",
#         "The dentist counts the teeth carefully.",
#         "You can raise your hand anytime.",
#         "You will get a shiny sticker after the visit."
#     ],
#     "haircut": [
#         "The chair is big and safe.",
#         "The scissors make a soft snip sound.",
#         "The barber talks in a calm voice.",
#         "Hair grows back again.",
#         "You can look in the mirror.",
#         "You will look neat and fresh."
#     ],
#     "doctor_visit": [
#         "The doctor listens to your heartbeat.",
#         "The room smells clean.",
#         "The doctor speaks gently.",
#         "The checkup will be quick.",
#         "You can sit on a soft bed.",
#         "You will feel proud afterward."
#     ],
# }

# emotions = [
#     "feels a little nervous",
#     "feels curious",
#     "feels brave",
#     "feels unsure",
#     "feels excited",
#     "takes a deep breath",
#     "holds mom's hand",
#     "remembers to stay calm",
# ]

# endings = [
#     "Everything goes well.",
#     "It is not scary at all.",
#     "You did a great job.",
#     "You are very brave.",
#     "It finishes quickly.",
#     "You feel proud and happy."
# ]

# names = ["<name>"]

# def generate_story(scenario, details):
#     emotion = random.choice(emotions)
#     chosen_details = random.sample(details, 4)
#     ending = random.choice(endings)

#     story = f"<scenario_{scenario}> <name> {emotion}. "
#     story += " ".join(chosen_details) + " "
#     story += ending
#     story += " <end>"

#     return story


# def main():
#     entries = []

#     for scenario, details in scenarios.items():
#         for _ in range(100):  # 100 per scenario
#             entries.append(generate_story(scenario, details))

#     random.shuffle(entries)

#     with open("stories.txt", "w", encoding="utf-8") as f:
#         for story in entries:
#             f.write(story + "\n")

#     print(f"Generated {len(entries)} stories.")


# if __name__ == "__main__":
#     main()


import random

scenarios = {

    "dentist_visit": {
        "events": [
            "The dentist checks the teeth.",
            "There is a bright light.",
            "The chair moves slowly.",
            "The dentist counts carefully."
        ],
        "coping": [
            "<NAME> can take slow deep breaths.",
            "<NAME> can raise a hand if needed.",
            "<NAME> can hold the chair gently.",
            "<NAME> can listen to the calm voice."
        ]
    },

    "haircut": {
        "events": [
            "The scissors make a soft sound.",
            "Hair falls slowly.",
            "The chair turns gently.",
            "The barber smiles kindly."
        ],
        "coping": [
            "<NAME> can look in the mirror.",
            "<NAME> can hold a small toy.",
            "<NAME> can ask for a short break.",
            "<NAME> can take deep breaths."
        ]
    },

    "school_bus": {
        "events": [
            "The bus is loud.",
            "Children are talking.",
            "The bus moves quickly.",
            "The driver watches carefully."
        ],
        "coping": [
            "<NAME> can sit near the front.",
            "<NAME> can look out the window.",
            "<NAME> can wear headphones.",
            "<NAME> can hold a comfort toy."
        ]
    },

    "birthday_party": {
        "events": [
            "Music plays loudly.",
            "Children laugh and shout.",
            "Balloons fill the room.",
            "Games are being played."
        ],
        "coping": [
            "<NAME> can take a quiet break.",
            "<NAME> can stay near a safe adult.",
            "<NAME> can cover the ears.",
            "<NAME> can take slow breaths."
        ]
    },

    "loud_noise": {
        "events": [
            "A truck makes a big sound.",
            "Thunder is very loud.",
            "A door slams suddenly.",
            "The noise echoes in the room."
        ],
        "coping": [
            "<NAME> can cover the ears.",
            "<NAME> can hug a pillow.",
            "<NAME> can take deep breaths.",
            "<NAME> can stay near a safe adult."
        ]
    },


    "grocery_store": {
        "events": [
            "The store has bright lights.",
            "Carts make loud sounds.",
            "Many people walk around.",
            "There are many items on shelves."
        ],
        "coping": [
            "<NAME> can hold the cart.",
            "<NAME> can wear headphones.",
            "<NAME> can stay close to a parent.",
            "<NAME> can take slow breaths."
        ]
    },

    "waiting_in_line": {
        "events": [
            "The line moves slowly.",
            "Many people are waiting.",
            "It takes time.",
            "Others stand quietly."
        ],
        "coping": [
            "<NAME> can count to ten.",
            "<NAME> can squeeze hands gently.",
            "<NAME> can look at a book.",
            "<NAME> can take deep breaths."
        ]
    },

    "classroom_test": {
        "events": [
            "The teacher gives a paper.",
            "The room is quiet.",
            "Everyone is writing.",
            "The test has simple questions."
        ],
        "coping": [
            "<NAME> can read slowly.",
            "<NAME> can ask for help.",
            "<NAME> can breathe calmly.",
            "<NAME> can try their best."
        ]
    },

    "new_teacher": {
        "events": [
            "The teacher smiles kindly.",
            "The voice sounds different.",
            "The classroom looks the same.",
            "The teacher explains the rules."
        ],
        "coping": [
            "<NAME> can say hello softly.",
            "<NAME> can sit calmly.",
            "<NAME> can raise a hand.",
            "<NAME> can take slow breaths."
        ]
    },

    "substitute_teacher": {
        "events": [
            "The regular teacher is away.",
            "A new teacher stands in class.",
            "Instructions sound different.",
            "The routine changes."
        ],
        "coping": [
            "<NAME> can ask questions.",
            "<NAME> can follow class rules.",
            "<NAME> can look at the schedule.",
            "<NAME> can breathe slowly."
        ]
    },

    "cafeteria_noise": {
        "events": [
            "Children talk loudly.",
            "Chairs scrape the floor.",
            "Plates make clinking sounds.",
            "Many students sit together."
        ],
        "coping": [
            "<NAME> can sit at a quiet table.",
            "<NAME> can wear headphones.",
            "<NAME> can take deep breaths.",
            "<NAME> can stay near a friend."
        ]
    },

    "playground_conflict": {
        "events": [
            "A friend says something unkind.",
            "Two children want the same swing.",
            "Voices get louder.",
            "A teacher walks over."
        ],
        "coping": [
            "<NAME> can use calm words.",
            "<NAME> can ask for help.",
            "<NAME> can take slow breaths.",
            "<NAME> can wait for a turn."
        ]
    },

    "swimming_pool": {
        "events": [
            "The water looks deep.",
            "The pool smells clean.",
            "Children splash loudly.",
            "The instructor stands nearby."
        ],
        "coping": [
            "<NAME> can hold the pool edge.",
            "<NAME> can wear floaties.",
            "<NAME> can move slowly.",
            "<NAME> can take deep breaths."
        ]
    },

    "moving_house": {
        "events": [
            "Boxes are packed.",
            "Rooms look different.",
            "Furniture is moved.",
            "New neighbors live nearby."
        ],
        "coping": [
            "<NAME> can keep favorite toys close.",
            "<NAME> can decorate the new room.",
            "<NAME> can talk about feelings.",
            "<NAME> can take slow breaths."
        ]
    },

    "fire_alarm": {
        "events": [
            "The alarm rings loudly.",
            "Teachers guide students outside.",
            "Everyone walks quickly.",
            "The sound is very loud."
        ],
        "coping": [
            "<NAME> can cover the ears.",
            "<NAME> can walk calmly.",
            "<NAME> can hold a teacher's hand.",
            "<NAME> can take deep breaths."
        ]
    },

    "sharing_toys": {
        "events": [
            "Another child wants the same toy.",
            "Taking turns feels hard.",
            "The teacher watches kindly.",
            "Children wait patiently."
        ],
        "coping": [
            "<NAME> can take turns.",
            "<NAME> can use kind words.",
            "<NAME> can count slowly.",
            "<NAME> can breathe calmly."
        ]
    },

    "sports_day": {
        "events": [
            "Many people are watching.",
            "Children run on the field.",
            "Teachers cheer loudly.",
            "The whistle blows."
        ],
        "coping": [
            "<NAME> can focus on one step.",
            "<NAME> can breathe slowly.",
            "<NAME> can try their best.",
            "<NAME> can ask for help."
        ]
    },

    "bedtime": {
        "events": [
            "The room is dark.",
            "Shadows move on the wall.",
            "The house becomes quiet.",
            "The clock ticks softly."
        ],
        "coping": [
            "<NAME> can turn on a night light.",
            "<NAME> can hug a soft toy.",
            "<NAME> can take slow breaths.",
            "<NAME> can call a parent."
        ]
    },

    "library_visit": {
        "events": [
            "People read quietly.",
            "Books fill the shelves.",
            "Chairs are soft.",
            "Voices are whispers."
        ],
        "coping": [
            "<NAME> can whisper softly.",
            "<NAME> can choose one book.",
            "<NAME> can sit in a cozy spot.",
            "<NAME> can take slow breaths."
        ]
    },

    "broken_routine": {
        "events": [
            "Plans change suddenly.",
            "The schedule looks different.",
            "Things happen in a new way.",
            "The day feels unusual."
        ],
        "coping": [
            "<NAME> can ask what happens next.",
            "<NAME> can look at a schedule.",
            "<NAME> can take deep breaths.",
            "<NAME> can talk about feelings."
        ]
    },

    "vaccination": {
        "events": [
            "The nurse holds a small needle.",
            "The room smells clean.",
            "It feels quick.",
            "The nurse smiles kindly."
        ],
        "coping": [
            "<NAME> can look away.",
            "<NAME> can squeeze a hand.",
            "<NAME> can take deep breaths.",
            "<NAME> can count to five."
        ]
    },

    "art_class": {
        "events": [
            "Paint colors mix together.",
            "The brush feels wet.",
            "Paper lies on the table.",
            "Children create pictures."
        ],
        "coping": [
            "<NAME> can ask for help.",
            "<NAME> can try again slowly.",
            "<NAME> can take a short break.",
            "<NAME> can breathe calmly."
        ]
    },

    "assembly": {
        "events": [
            "Many students sit together.",
            "Speakers talk loudly.",
            "Clapping fills the room.",
            "Teachers stand nearby."
        ],
        "coping": [
            "<NAME> can sit at the side.",
            "<NAME> can cover the ears.",
            "<NAME> can take deep breaths.",
            "<NAME> can focus on a teacher."
        ]
    },

    "losing_game": {
        "events": [
            "Another child wins.",
            "Everyone claps.",
            "The game ends.",
            "Scores are announced."
        ],
        "coping": [
            "<NAME> can say good job.",
            "<NAME> can try again later.",
            "<NAME> can take slow breaths.",
            "<NAME> can remember games are for fun."
        ]
    },

    "group_activity": {
        "events": [
            "Children sit in a circle.",
            "Everyone shares ideas.",
            "Voices overlap sometimes.",
            "The teacher guides the group."
        ],
        "coping": [
            "<NAME> can wait for a turn.",
            "<NAME> can speak slowly.",
            "<NAME> can raise a hand.",
            "<NAME> can breathe calmly."
        ]
    }

}

emotions = [
    "scared",
    "nervous",
    "sad",
    "angry",
    "shy",
    "worried",
    "overwhelmed",
    "excited"
]


def generate_story(scenario_name, data):
    emotion = random.choice(emotions)

    chosen_events = random.sample(data["events"], 2)
    chosen_coping = random.sample(data["coping"], 2)

    return f"""[SCENARIO] {scenario_name}
[EMOTION] {emotion}
[TEXT]
<START>
Today, <NAME> faces {scenario_name.replace('_', ' ')}.
</START>
<FEELING>
<NAME> feels {emotion}.
</FEELING>
<EVENT>
{chosen_events[0]}
{chosen_events[1]}
</EVENT>
<COPING>
{chosen_coping[0]}
{chosen_coping[1]}
</COPING>
<ENDING>
The situation ends safely.
<NAME> feels better.
<END>
"""


def main():
    entries = []

    for scenario_name, data in scenarios.items():
        for _ in range(50):  # 50 variations per scenario
            entries.append(generate_story(scenario_name, data))

    random.shuffle(entries)

    with open("stories.txt", "w", encoding="utf-8") as f:
        for story in entries:
            f.write(story + "\n")

    print(f"Generated {len(entries)} structured stories.")


if __name__ == "__main__":
    main()