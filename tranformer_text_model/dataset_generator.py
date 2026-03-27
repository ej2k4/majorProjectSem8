import json
import random
import os

os.makedirs("data", exist_ok=True)

DATASET_SIZE = 20000
MIN_EVENTS = 3
MAX_EVENTS = 6

scenarios = [
"art_class","playground_conflict","doctor_visit","bedtime",
"classroom_test","sharing_toys","loud_cafeteria","school_presentation",
"new_student","losing_a_game","winning_a_game","group_project",
"asking_a_question","making_a_mistake","waiting_turn","library_time",
"sports_practice","music_class","helping_a_friend","first_day_school",
"buying_snack","saving_money","receiving_allowance","spending_money_wisely",
"dental_visit","injury_while_playing"
]

emotions = [
"happy","excited","nervous","curious","confused","sad",
"angry","frustrated","proud","scared","shy","worried","surprised"
]

opening_lines = [
"The day begins like any other day.",
"Something new is about to happen.",
"Today something interesting is happening.",
"It is an important moment.",
"The classroom is full of activity."
]

events = [
"children are drawing colorful pictures",
"students are talking and laughing",
"the room is a little noisy",
"a teacher explains something new",
"friends are working together",
"toys are spread across the floor",
"students wait for their turn",
"books are stacked on the table",
"a group is solving a problem",
"everyone prepares for an activity"
]

feelings = [
"<name> feels {emotion}.",
"<name> begins to feel {emotion}.",
"<name> notices feeling {emotion}.",
"<name>'s heart feels {emotion}."
]

thoughts = [
"<name> wonders what to do.",
"<name> takes a moment to think.",
"<name> tries to understand.",
"<name> remembers something helpful."
]

coping = [
"<name> takes a deep breath.",
"<name> counts slowly to five.",
"<name> asks the teacher for help.",
"<name> uses calm words.",
"<name> listens carefully.",
"<name> tries again slowly."
]

endings = [
"everything becomes calm",
"the situation gets better",
"everyone understands",
"things work out well",
"the problem is solved"
]

def build_story():

    scenario = random.choice(scenarios)
    emotion = random.choice(emotions)

    story = []

    story.append(f"<scenario_{scenario}> <emotion_{emotion}> <start>")

    story.append(random.choice(opening_lines))

    story.append(random.choice(feelings).format(emotion=emotion))

    num_events = random.randint(MIN_EVENTS,MAX_EVENTS)

    for _ in range(num_events):

        story.append("In the room " + random.choice(events) + ".")
        story.append(random.choice(thoughts))

    story.append(random.choice(coping))
    story.append(random.choice(coping))

    story.append(random.choice(endings) + ".")

    story.append("<name> feels better.")

    story.append("<end>")

    return " ".join(story)

dataset = [build_story() for _ in range(DATASET_SIZE)]

with open("data/stories.json","w") as f:
    json.dump(dataset,f,indent=2)

print("Dataset generated:",len(dataset))
