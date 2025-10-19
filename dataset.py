
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SimpleImageDataset(Dataset):
    def __init__(self, image_dir="/content/Maya-model-1/data/my_cats/cats_only", image_size=64):
        self.image_dir = image_dir
        self.image_size = image_size
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"🐱 Loading {{len(self.image_files)}} cat images from {{image_dir}}")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)
