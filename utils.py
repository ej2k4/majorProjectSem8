import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE = 32
BATCH_SIZE = 32

def get_dataloader(dataset_path="dataset"):

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),  # Converts to [0,1]
        transforms.Normalize([0.5, 0.5, 0.5],   # Normalize to [-1,1]
                             [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return dataloader, len(dataset.classes)
