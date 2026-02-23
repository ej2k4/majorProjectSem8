import os
import random
from PIL import Image, ImageDraw

IMG_SIZE = 32
NUM_IMAGES = 100

SCENARIOS = ["dentist", "doctor", "haircut"]

BASE_DIR = "dataset"


def ensure_folders():
    for scenario in SCENARIOS:
        os.makedirs(os.path.join(BASE_DIR, scenario), exist_ok=True)


def random_color():
    return tuple(random.randint(50, 255) for _ in range(3))


def generate_dentist(draw):
    bg_color = (random.randint(80, 150), random.randint(120, 180), 255)
    draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=bg_color)

    x_offset = random.randint(-2, 2)
    y_offset = random.randint(-2, 2)

    draw.ellipse(
        [8 + x_offset, 8 + y_offset, 24 + x_offset, 28 + y_offset],
        fill=(255, 255, 255)
    )

    draw.arc(
        [10 + x_offset, 15 + y_offset, 22 + x_offset, 25 + y_offset],
        start=0,
        end=180,
        fill=(0, 0, 0)
    )


def generate_doctor(draw):
    bg_color = (random.randint(100, 180), 200, random.randint(100, 180))
    draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=bg_color)

    x_offset = random.randint(-3, 3)
    y_offset = random.randint(-3, 3)

    draw.polygon(
        [(16 + x_offset, 10 + y_offset),
         (22 + x_offset, 16 + y_offset),
         (16 + x_offset, 24 + y_offset),
         (10 + x_offset, 16 + y_offset)],
        fill=(255, random.randint(0, 50), random.randint(0, 50))
    )

    draw.ellipse(
        [12 + x_offset, 4 + y_offset, 20 + x_offset, 12 + y_offset],
        fill=(255, 220, 180)
    )



def generate_haircut(draw):
    bg_color = (255, random.randint(200, 255), random.randint(80, 150))
    draw.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=bg_color)

    x_offset = random.randint(-2, 2)
    y_offset = random.randint(-2, 2)

    draw.line(
        [8 + x_offset, 8 + y_offset, 24 + x_offset, 24 + y_offset],
        fill=(0, 0, 0),
        width=2
    )

    draw.line(
        [24 + x_offset, 8 + y_offset, 8 + x_offset, 24 + y_offset],
        fill=(0, 0, 0),
        width=2
    )

    draw.ellipse(
        [6 + x_offset, 6 + y_offset, 12 + x_offset, 12 + y_offset],
        outline=(0, 0, 0),
        width=2
    )

    draw.ellipse(
        [20 + x_offset, 6 + y_offset, 26 + x_offset, 12 + y_offset],
        outline=(0, 0, 0),
        width=2
    )



def generate_image(scenario, index):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(img)

    if scenario == "dentist":
        generate_dentist(draw)
    elif scenario == "doctor":
        generate_doctor(draw)
    elif scenario == "haircut":
        generate_haircut(draw)

    save_path = os.path.join(BASE_DIR, scenario, f"{scenario}_{index}.png")
    img.save(save_path)


def main():
    ensure_folders()

    for scenario in SCENARIOS:
        for i in range(NUM_IMAGES):
            generate_image(scenario, i)

    print("Dataset generation complete!")


if __name__ == "__main__":
    main()
