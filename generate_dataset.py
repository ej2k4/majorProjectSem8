import json
import random
import os

os.makedirs("data", exist_ok=True)

# -----------------------
# Configuration
# -----------------------

DATASET_SIZE = 20000
MIN_EVENTS = 3
MAX_EVENTS = 6

# -----------------------
# Scenario tags
# -----------------------

scenarios = [
    "art_class","playground_conflict","doctor_visit","bedtime",
    "classroom_test","sharing_toys","loud_cafeteria","school_presentation",
    "new_student","losing_a_game","winning_a_game","group_project",
    "asking_a_question","making_a_mistake","waiting_turn","library_time",
    "sports_practice","music_class","helping_a_friend","first_day_school",
    "buying_snack","saving_money","receiving_allowance","spending_money_wisely"
]

emotions = [
    "happy","excited","nervous","curious","confused","sad",
    "angry","frustrated","proud","scared","shy","worried","surprised"
]

# -----------------------
# Story building blocks
# -----------------------

opening_lines = [
    "Today something interesting is happening.",
    "It is an important moment in the day.",
    "The day begins like any other day.",
    "Something new is about to happen.",
    "The classroom is full of activity.",
]

events = [
    "children are drawing colorful pictures",
    "many students are talking and laughing",
    "the classroom is busy and a little noisy",
    "a teacher is explaining something new",
    "friends are working together",
    "toys are spread across the floor",
    "students are waiting for their turn",
    "books are stacked on the library table",
    "a group of students are solving a problem",
    "everyone is preparing for an activity",
    "students are practicing a presentation",
    "a friend is asking for help",
]

feelings = [
    "<name> feels a little {emotion}.",
    "<name> begins to feel {emotion} inside.",
    "<name> notices the feeling of {emotion}.",
    "<name>'s heart feels {emotion}.",
]

thoughts = [
    "<name> wonders what to do next.",
    "<name> takes a moment to think.",
    "<name> tries to understand the situation.",
    "<name> remembers something helpful.",
]

coping_actions = [
    "<name> takes a deep breath.",
    "<name> counts slowly to five.",
    "<name> asks the teacher for help.",
    "<name> uses calm and kind words.",
    "<name> takes a short break.",
    "<name> listens carefully.",
    "<name> tries again slowly.",
    "<name> asks a friend politely.",
    "<name> waits patiently.",
    "<name> thinks carefully about the problem.",
]

positive_actions = [
    "friends begin helping each other",
    "everyone starts working together",
    "the problem slowly becomes smaller",
    "people start understanding each other",
    "things begin to feel calmer",
]

endings = [
    "everything becomes calm and safe",
    "the situation gets better",
    "everyone understands each other",
    "things work out well",
    "the problem is solved kindly",
    "everyone feels happy again",
    "the day continues peacefully"
]

# -----------------------
# Story generator
# -----------------------

def build_story():

    scenario = random.choice(scenarios)
    emotion = random.choice(emotions)

    story = []

    story.append(f"<scenario_{scenario}> <emotion_{emotion}> <name> </start>")

    story.append(random.choice(opening_lines))

    story.append(random.choice(feelings).format(emotion=emotion))

    # random event sequence
    num_events = random.randint(MIN_EVENTS, MAX_EVENTS)

    for _ in range(num_events):

        story.append("In the room, " + random.choice(events) + ".")
        story.append(random.choice(thoughts))

    # coping phase
    for _ in range(2):
        story.append(random.choice(coping_actions))

    story.append("Soon, " + random.choice(positive_actions) + ".")

    story.append(random.choice(endings) + ".")

    story.append("<name> feels better and learns something important.")

    story.append("<end>")

    return " ".join(story)

# -----------------------
# Build dataset
# -----------------------

dataset = []

for _ in range(DATASET_SIZE):
    dataset.append(build_story())

# -----------------------
# Save
# -----------------------

with open("data/stories.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Dataset generated:", len(dataset))