import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CatDataset(Dataset):
    def __init__(self, image_folder, image_size=64):
        self.image_paths = [
            os.path.join(image_folder, f) 
            for f in os.listdir(image_folder) 
            if f.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))
        ]
        
        print(f"Found {len(self.image_paths)} cat images! 🐱")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB normalization
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            # Return a blank image if there's error
            return torch.zeros(3, 64, 64)