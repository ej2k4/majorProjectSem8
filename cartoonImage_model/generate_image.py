import torch
import os
from torchvision.utils import save_image
from generator import Generator
from utils import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise_dim = 100
model_path = "generator.pth"
output_dir = "outputs"

os.makedirs(output_dir, exist_ok=True)

# Get class names from dataset
dataloader, num_classes = get_dataloader()
class_names = dataloader.dataset.classes

# Load trained generator
G = Generator(noise_dim=noise_dim, num_classes=num_classes).to(device)
G.load_state_dict(torch.load(model_path, map_location=device))
G.eval()

def generate_image_for_scenario(scenario_name):
    if scenario_name not in class_names:
        print("Invalid scenario.")
        print("Available scenarios:", class_names)
        return

    label_index = class_names.index(scenario_name)
    label_tensor = torch.tensor([label_index]).to(device)

    noise = torch.randn(1, noise_dim).to(device)

    with torch.no_grad():
        fake_img = G(noise, label_tensor)

    save_path = os.path.join(output_dir, f"{scenario_name}_generated.png")
    save_image(fake_img, save_path, normalize=True)

    print("Image saved to:", save_path)

if __name__ == "__main__":
    scenario = input("Enter scenario name: ")
    generate_image_for_scenario(scenario)