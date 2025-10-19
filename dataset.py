import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SimpleImageDataset(Dataset):
    def __init__(self, image_dir="/content/Maya-model-1/data/my_cats/cats_only", image_size=64):
        # HARCODE PATH - tidak bisa salah
        self.image_dir = "/content/Maya-model-1/data/my_cats/cats_only"
        self.image_size = image_size
        
        print(f"🔍 Using HARDCODED path: {self.image_dir}")
        print(f"📁 Directory exists: {os.path.exists(self.image_dir)}")
        
        if os.path.exists(self.image_dir):
            all_files = os.listdir(self.image_dir)
            self.image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print(f"🐱 Cat images found: {len(self.image_files)}")
        else:
            self.image_files = []
            print("❌ Directory not found!")
        
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
