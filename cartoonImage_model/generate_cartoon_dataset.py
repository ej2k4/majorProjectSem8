import os
import random
from PIL import Image, ImageDraw

IMG_SIZE = 64
NUM_IMAGES_PER_COMBO = 50  # 50 images per scenario-emotion combo
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

EMOTIONS = ["happy", "sad", "neutral", "surprised"]

SCHOOL_SCENARIOS = [
    "classroom_test", "new_teacher", "substitute_teacher",
    "assembly", "group_activity", "art_class",
    "library_visit", "sports_day"
]

CLINIC_SCENARIOS = [
    "dentist", "doctor_visit", "vaccination"
]

def ensure_folders():
    """Create folders for each scenario-emotion combination"""
    for scenario in SCENARIOS:
        for emotion in EMOTIONS:
            folder_name = f"{scenario}_{emotion}"
            os.makedirs(os.path.join(BASE_DIR, folder_name), exist_ok=True)

def pastel_color():
    return tuple(random.randint(150, 255) for _ in range(3))

def draw_child(draw, emotion="neutral"):
    """Draw a child character with specific emotion"""
    # Body
    draw.rectangle([24, 42, 40, 60], fill=pastel_color())

    # Face
    draw.ellipse([20, 12, 44, 40], fill=(255, 220, 180), outline=(200, 180, 150))

    # Hair
    draw.ellipse([20, 8, 44, 22], fill=(90, 60, 40))

    # Eyes - vary based on emotion
    if emotion == "sad":
        draw.ellipse([26, 22, 30, 26], fill=(0, 0, 0))
        draw.ellipse([34, 22, 38, 26], fill=(0, 0, 0))
        # Sad eyebrows
        draw.line([26, 20, 30, 22], fill=(0, 0, 0), width=1)
        draw.line([34, 20, 38, 22], fill=(0, 0, 0), width=1)
    elif emotion == "surprised":
        draw.ellipse([26, 20, 30, 28], fill=(0, 0, 0))
        draw.ellipse([34, 20, 38, 28], fill=(0, 0, 0))
    else:
        draw.ellipse([26, 22, 30, 26], fill=(0, 0, 0))
        draw.ellipse([34, 22, 38, 26], fill=(0, 0, 0))

    # Emotion mouth
    if emotion == "happy":
        draw.arc([26, 28, 38, 36], start=0, end=180, fill=(0, 0, 0), width=2)
        # Add extra happy features
        draw.arc([24, 26, 32, 32], start=0, end=180, fill=(255, 200, 200), width=1)
        draw.arc([32, 26, 40, 32], start=0, end=180, fill=(255, 200, 200), width=1)
    elif emotion == "sad":
        draw.arc([26, 30, 38, 38], start=180, end=360, fill=(0, 0, 0), width=2)
        # Tears
        draw.line([28, 36, 28, 40], fill=(100, 150, 255), width=1)
        draw.line([36, 36, 36, 40], fill=(100, 150, 255), width=1)
    elif emotion == "surprised":
        draw.ellipse([30, 28, 34, 34], fill=(0, 0, 0))
        # Surprised eyebrows
        draw.polygon([(26, 18), (30, 16), (34, 18)], fill=(0, 0, 0))
        draw.polygon([(34, 18), (38, 16), (42, 18)], fill=(0, 0, 0))
    else:  # neutral
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

def generate_image(scenario, emotion, index):
    """Generate a single image with specific scenario and emotion"""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(img)

    if scenario in SCHOOL_SCENARIOS:
        generate_school_scene(draw, emotion)
    elif scenario in CLINIC_SCENARIOS:
        generate_clinic_scene(draw, emotion)
    else:
        generate_home_scene(draw, emotion)

    # Save with both scenario and emotion in filename
    folder_name = f"{scenario}_{emotion}"
    save_path = os.path.join(BASE_DIR, folder_name, f"{scenario}_{emotion}_{index}.png")
    img.save(save_path)

def main():
    ensure_folders()

    total_images = len(SCENARIOS) * len(EMOTIONS) * NUM_IMAGES_PER_COMBO
    print(f"Generating {total_images} images...")
    print(f"Scenarios: {len(SCENARIOS)}, Emotions: {len(EMOTIONS)}")

    count = 0
    for scenario in SCENARIOS:
        for emotion in EMOTIONS:
            for i in range(NUM_IMAGES_PER_COMBO):
                generate_image(scenario, emotion, i)
                count += 1
                if (count) % 100 == 0:
                    print(f"Generated {count}/{total_images} images...")

    print(f"Storybook cartoon dataset generation complete! Total: {total_images} images")

if __name__ == "__main__":
    main()