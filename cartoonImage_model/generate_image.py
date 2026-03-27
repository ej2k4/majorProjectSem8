import torch
import os
from torchvision.utils import save_image
from generator import Generator
from utils import get_dataloader, EMOTIONS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise_dim = 100
model_path = "best_generator.pth"
output_dir = "outputs"

os.makedirs(output_dir, exist_ok=True)

# Get class information from dataset
dataloader, num_scenarios, num_emotions = get_dataloader()
scenario_to_idx = dataloader.dataset.scenario_to_idx
emotion_to_idx = dataloader.dataset.emotion_to_idx

# Reverse mappings for display
idx_to_scenario = {v: k for k, v in scenario_to_idx.items()}
idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}

# Load trained generator
G = Generator(noise_dim=noise_dim, num_scenarios=num_scenarios, num_emotions=num_emotions).to(device)
G.load_state_dict(torch.load(model_path, map_location=device))
G.eval()

def generate_image_for_scenario_emotion(scenario_name, emotion_name, num_samples=1):
    """
    Generate images for a given scenario and emotion combination
    
    Args:
        scenario_name: Name of the scenario (str)
        emotion_name: Name of the emotion (str)
        num_samples: Number of images to generate (int)
    """
    # Validate scenario
    if scenario_name not in scenario_to_idx:
        print(f"Invalid scenario: '{scenario_name}'")
        print(f"Available scenarios: {list(scenario_to_idx.keys())}")
        return
    
    # Validate emotion
    if emotion_name not in emotion_to_idx:
        print(f"Invalid emotion: '{emotion_name}'")
        print(f"Available emotions: {list(emotion_to_idx.keys())}")
        return
    
    scenario_idx = scenario_to_idx[scenario_name]
    emotion_idx = emotion_to_idx[emotion_name]
    
    print(f"Generating {num_samples} image(s) for scenario='{scenario_name}', emotion='{emotion_name}'...")
    
    for sample_num in range(num_samples):
        # Create label tensors
        scenario_tensor = torch.tensor([scenario_idx]).to(device)
        emotion_tensor = torch.tensor([emotion_idx]).to(device)
        
        # Generate noise
        noise = torch.randn(1, noise_dim).to(device)
        
        # Generate image
        with torch.no_grad():
            fake_img = G(noise, scenario_tensor, emotion_tensor)
        
        # Save image
        filename = f"{scenario_name}_{emotion_name}"
        if num_samples > 1:
            filename += f"_{sample_num+1}"
        filename += ".png"
        
        save_path = os.path.join(output_dir, filename)
        save_image(fake_img, save_path, normalize=True)
        
        print(f"✓ Saved: {save_path}")

def interactive_generation():
    """Interactive mode for generating images"""
    print("\n" + "="*60)
    print("DUAL-INPUT GAN IMAGE GENERATOR")
    print("="*60)
    print(f"\nAvailable scenarios ({len(scenario_to_idx)}):")
    for i, scenario in enumerate(sorted(scenario_to_idx.keys()), 1):
        print(f"  {i:2d}. {scenario}")
    
    print(f"\nAvailable emotions ({len(emotion_to_idx)}):")
    for i, emotion in enumerate(sorted(emotion_to_idx.keys()), 1):
        print(f"  {i:2d}. {emotion}")
    
    print("\n" + "-"*60)
    
    while True:
        scenario = input("\nEnter scenario name (or 'quit' to exit): ").strip().lower()
        
        if scenario == 'quit':
            print("Exiting...")
            break
        
        emotion = input("Enter emotion name: ").strip().lower()
        
        try:
            num_samples = int(input("Number of images to generate (default: 1): ") or "1")
        except ValueError:
            num_samples = 1
        
        generate_image_for_scenario_emotion(scenario, emotion, num_samples)
        print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        # Command line mode: python generate_image.py <scenario> <emotion> [num_samples]
        scenario = sys.argv[1]
        emotion = sys.argv[2]
        num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        generate_image_for_scenario_emotion(scenario, emotion, num_samples)
    else:
        # Interactive mode
        interactive_generation()