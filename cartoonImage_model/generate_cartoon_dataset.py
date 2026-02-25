# import os
# import random
# from PIL import Image, ImageDraw

# IMG_SIZE = 32
# NUM_IMAGES = 100

# SCENARIOS = ["dentist", "doctor", "haircut"]

# BASE_DIR = "dataset"


# def ensure_folders():
#     for scenario in SCENARIOS:
#         os.makedirs(os.path.join(BASE_DIR, scenario), exist_ok=True)


# def random_color():
#     return tuple(random.randint(50, 255) for _ in range(3))


# def generate_dentist(draw):
#     bg_color = (random.randint(80, 150), random.randint(120, 180), 255)
#     draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=bg_color)

#     x_offset = random.randint(-2, 2)
#     y_offset = random.randint(-2, 2)

#     draw.ellipse(
#         [8 + x_offset, 8 + y_offset, 24 + x_offset, 28 + y_offset],
#         fill=(255, 255, 255)
#     )

#     draw.arc(
#         [10 + x_offset, 15 + y_offset, 22 + x_offset, 25 + y_offset],
#         start=0,
#         end=180,
#         fill=(0, 0, 0)
#     )


# def generate_doctor(draw):
#     bg_color = (random.randint(100, 180), 200, random.randint(100, 180))
#     draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=bg_color)

#     x_offset = random.randint(-3, 3)
#     y_offset = random.randint(-3, 3)

#     draw.polygon(
#         [(16 + x_offset, 10 + y_offset),
#          (22 + x_offset, 16 + y_offset),
#          (16 + x_offset, 24 + y_offset),
#          (10 + x_offset, 16 + y_offset)],
#         fill=(255, random.randint(0, 50), random.randint(0, 50))
#     )

#     draw.ellipse(
#         [12 + x_offset, 4 + y_offset, 20 + x_offset, 12 + y_offset],
#         fill=(255, 220, 180)
#     )



# def generate_haircut(draw):
#     bg_color = (255, random.randint(200, 255), random.randint(80, 150))
#     draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=bg_color)

#     x_offset = random.randint(-2, 2)
#     y_offset = random.randint(-2, 2)

#     draw.line(
#         [8 + x_offset, 8 + y_offset, 24 + x_offset, 24 + y_offset],
#         fill=(0, 0, 0),
#         width=2
#     )

#     draw.line(
#         [24 + x_offset, 8 + y_offset, 8 + x_offset, 24 + y_offset],
#         fill=(0, 0, 0),
#         width=2
#     )

#     draw.ellipse(
#         [6 + x_offset, 6 + y_offset, 12 + x_offset, 12 + y_offset],
#         outline=(0, 0, 0),
#         width=2
#     )

#     draw.ellipse(
#         [20 + x_offset, 6 + y_offset, 26 + x_offset, 12 + y_offset],
#         outline=(0, 0, 0),
#         width=2
#     )



# def generate_image(scenario, index):
#     img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
#     draw = ImageDraw.Draw(img)

#     if scenario == "dentist":
#         generate_dentist(draw)
#     elif scenario == "doctor":
#         generate_doctor(draw)
#     elif scenario == "haircut":
#         generate_haircut(draw)

#     save_path = os.path.join(BASE_DIR, scenario, f"{scenario}_{index}.png")
#     img.save(save_path)


# def main():
#     ensure_folders()

#     for scenario in SCENARIOS:
#         for i in range(NUM_IMAGES):
#             generate_image(scenario, i)

#     print("Dataset generation complete!")


# if __name__ == "__main__":
#     main()


# import os
# import random
# from PIL import Image, ImageDraw

# IMG_SIZE = 64
# NUM_IMAGES = 150

# BASE_DIR = "dataset"

# SCENARIOS = [
#     "dentist",
#     "doctor_visit",
#     "haircut",
#     "grocery_store",
#     "waiting_in_line",
#     "classroom_test",
#     "new_teacher",
#     "substitute_teacher",
#     "cafeteria_noise",
#     "playground_conflict",
#     "swimming_pool",
#     "moving_house",
#     "fire_alarm",
#     "sharing_toys",
#     "sports_day",
#     "bedtime",
#     "library_visit",
#     "broken_routine",
#     "vaccination",
#     "art_class",
#     "assembly",
#     "losing_game",
#     "group_activity"
# ]

# SCHOOL_SCENARIOS = [
#     "classroom_test", "new_teacher", "substitute_teacher",
#     "assembly", "group_activity", "art_class",
#     "library_visit", "sports_day"
# ]

# CLINIC_SCENARIOS = [
#     "dentist", "doctor_visit", "vaccination"
# ]

# def ensure_folders():
#     for scenario in SCENARIOS:
#         os.makedirs(os.path.join(BASE_DIR, scenario), exist_ok=True)

# def draw_child(draw, emotion="neutral"):
#     # Face
#     draw.ellipse([18, 10, 46, 40], fill=(255, 220, 180))

#     # Eyes
#     draw.ellipse([26, 22, 30, 26], fill=(0, 0, 0))
#     draw.ellipse([34, 22, 38, 26], fill=(0, 0, 0))

#     # Mouth based on emotion
#     if emotion == "happy":
#         draw.arc([26, 28, 38, 36], start=0, end=180, fill=(0, 0, 0))
#     elif emotion == "sad":
#         draw.arc([26, 30, 38, 38], start=180, end=360, fill=(0, 0, 0))
#     elif emotion == "surprised":
#         draw.ellipse([30, 28, 34, 32], fill=(0, 0, 0))
#     else:
#         draw.line([28, 32, 36, 32], fill=(0, 0, 0), width=2)

# def generate_school_scene(draw, emotion):
#     draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=(200, 230, 255))
#     draw_child(draw, emotion)

# def generate_clinic_scene(draw, emotion):
#     draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=(220, 255, 220))
#     draw_child(draw, emotion)

# def generate_home_scene(draw, emotion):
#     draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=(255, 240, 200))
#     draw_child(draw, emotion)

# def generate_image(scenario, index):
#     img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
#     draw = ImageDraw.Draw(img)

#     emotion = random.choice(["happy", "sad", "neutral", "surprised"])

#     if scenario in SCHOOL_SCENARIOS:
#         generate_school_scene(draw, emotion)
#     elif scenario in CLINIC_SCENARIOS:
#         generate_clinic_scene(draw, emotion)
#     else:
#         generate_home_scene(draw, emotion)

#     save_path = os.path.join(BASE_DIR, scenario, f"{scenario}_{index}.png")
#     img.save(save_path)

# def main():
#     ensure_folders()

#     for scenario in SCENARIOS:
#         for i in range(NUM_IMAGES):
#             generate_image(scenario, i)

#     print("Dataset generation complete!")

# if __name__ == "__main__":
#     main()


import os
import random
from PIL import Image, ImageDraw

IMG_SIZE = 64
NUM_IMAGES = 200
BASE_DIR = "dataset"

SCENARIOS = [
    "dentist", "doctor_visit", "haircut",
    "grocery_store", "waiting_in_line",
    "classroom_test", "new_teacher", "substitute_teacher",
    "cafeteria_noise", "playground_conflict",
    "swimming_pool", "moving_house", "fire_alarm",
    "sharing_toys", "sports_day", "bedtime",
    "library_visit", "broken_routine",
    "vaccination", "art_class", "assembly",
    "losing_game", "group_activity"
]

SCHOOL_SCENARIOS = [
    "classroom_test", "new_teacher", "substitute_teacher",
    "assembly", "group_activity", "art_class",
    "library_visit", "sports_day"
]

CLINIC_SCENARIOS = [
    "dentist", "doctor_visit", "vaccination"
]

def ensure_folders():
    for scenario in SCENARIOS:
        os.makedirs(os.path.join(BASE_DIR, scenario), exist_ok=True)

def pastel_color():
    return tuple(random.randint(150, 255) for _ in range(3))

def draw_child(draw, emotion="neutral"):
    # Body
    draw.rectangle([24, 42, 40, 60], fill=pastel_color())

    # Face
    draw.ellipse([20, 12, 44, 40], fill=(255, 220, 180), outline=(200,180,150))

    # Hair
    draw.ellipse([20, 8, 44, 22], fill=(90, 60, 40))

    # Eyes
    draw.ellipse([26, 22, 30, 26], fill=(0, 0, 0))
    draw.ellipse([34, 22, 38, 26], fill=(0, 0, 0))

    # Emotion mouth
    if emotion == "happy":
        draw.arc([26, 28, 38, 36], start=0, end=180, fill=(0, 0, 0), width=2)
    elif emotion == "sad":
        draw.arc([26, 30, 38, 38], start=180, end=360, fill=(0, 0, 0), width=2)
    elif emotion == "surprised":
        draw.ellipse([30, 28, 34, 32], fill=(0, 0, 0))
    else:
        draw.line([28, 32, 36, 32], fill=(0, 0, 0), width=2)

def generate_school_scene(draw, emotion):
    draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=(200, 230, 255))
    draw.rectangle([0, 45, 64, 64], fill=(170, 120, 80))
    draw.rectangle([8, 18, 20, 30], fill=(255, 255, 150))
    draw_child(draw, emotion)

def generate_clinic_scene(draw, emotion):
    draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=(220, 255, 220))
    draw.rectangle([0, 45, 64, 64], fill=(200, 200, 200))
    draw.rectangle([42, 15, 55, 28], fill=(255, 100, 100))
    draw_child(draw, emotion)

def generate_home_scene(draw, emotion):
    draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=(255, 240, 200))
    draw.rectangle([0, 45, 64, 64], fill=(180, 140, 100))
    draw.rectangle([8, 20, 20, 35], fill=(255, 200, 150))
    draw_child(draw, emotion)

def generate_image(scenario, index):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(img)

    emotion = random.choice(["happy", "sad", "neutral", "surprised"])

    if scenario in SCHOOL_SCENARIOS:
        generate_school_scene(draw, emotion)
    elif scenario in CLINIC_SCENARIOS:
        generate_clinic_scene(draw, emotion)
    else:
        generate_home_scene(draw, emotion)

    save_path = os.path.join(BASE_DIR, scenario, f"{scenario}_{index}.png")
    img.save(save_path)

def main():
    ensure_folders()

    for scenario in SCENARIOS:
        for i in range(NUM_IMAGES):
            generate_image(scenario, i)

    print("Storybook cartoon dataset generated!")

if __name__ == "__main__":
    main()