import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os

IMG_SIZE = 64
BATCH_SIZE = 8

EMOTIONS = ["happy", "sad", "neutral", "surprised"]

class DualLabelImageFolder(Dataset):
    """
    Custom dataset that extracts both scenario and emotion from folder structure.
    Expected folder structure: dataset/scenario_emotion/images/
    Example: dataset/dentist_happy/, dataset/doctor_visit_sad/, etc.
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.scenario_labels = []
        self.emotion_labels = []
        self.scenario_to_idx = {}
        self.emotion_to_idx = {}
        
        # Create emotion to index mapping
        for idx, emotion in enumerate(EMOTIONS):
            self.emotion_to_idx[emotion] = idx
        
        # Scan directories and build image list
        scenario_idx = 0
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
            
            # Extract scenario and emotion from folder name
            parts = folder_name.rsplit('_', 1)
            if len(parts) != 2:
                continue
            
            scenario = parts[0]
            emotion = parts[1]
            
            # Only process valid emotions
            if emotion not in self.emotion_to_idx:
                continue
            
            # Register scenario if new
            if scenario not in self.scenario_to_idx:
                self.scenario_to_idx[scenario] = scenario_idx
                scenario_idx += 1
            
            # Add all images from this folder
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                self.image_paths.append(img_path)
                self.scenario_labels.append(self.scenario_to_idx[scenario])
                self.emotion_labels.append(self.emotion_to_idx[emotion])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        scenario_label = self.scenario_labels[idx]
        emotion_label = self.emotion_labels[idx]
        
        return image, scenario_label, emotion_label


def get_dataloader(dataset_path="dataset"):
    """
    Load the dual-label dataset (scenario + emotion)
    
    Returns:
        dataloader: DataLoader with batches of (images, scenario_labels, emotion_labels)
        num_scenarios: Number of unique scenarios
        num_emotions: Number of unique emotions
    """
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    dataset = DualLabelImageFolder(
        root_dir=dataset_path,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    num_scenarios = len(dataset.scenario_to_idx)
    num_emotions = len(dataset.emotion_to_idx)

    print(f"Dataset loaded: {len(dataset)} images")
    print(f"Scenarios: {num_scenarios} - {list(dataset.scenario_to_idx.keys())}")
    print(f"Emotions: {num_emotions} - {list(dataset.emotion_to_idx.keys())}")

    return dataloader, num_scenarios, num_emotions