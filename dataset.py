import torch
from model import SimpleUNet
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import glob
import os

def simple_reconstruction_test(model_path, data_folder, num_samples=4):
    """Test rekonstruksi yang lebih sederhana"""
    
    # Load model
    model = SimpleUNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"✓ Model loaded: {model_path}")
    print(f"Using device: {device}")
    
    # Load images manually tanpa Dataset class
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_folder, ext)))
        image_files.extend(glob.glob(os.path.join(data_folder, '**', ext), recursive=True))
    
    print(f"Found {len(image_files)} images")
    
    if len(image_files) < num_samples:
        print(f"Not enough images. Found {len(image_files)}, need {num_samples}")
        return
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load sample images
    real_images = []
    for img_path in image_files[:num_samples]:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            real_images.append(img_tensor)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    if not real_images:
        print("No images could be loaded")
        return
    
    real_batch = torch.stack(real_images).to(device)
    print(f"Real images shape: {real_batch.shape}")
    
    # Test diffusion process
    with torch.no_grad():
        # Add noise
        timesteps = torch.tensor([100] * len(real_batch), device=device, dtype=torch.long)
        noise = torch.randn_like(real_batch)
        
        # Add noise
        alpha = 0.7  # Noise level
        noisy_images = alpha * real_batch + (1 - alpha) * noise
        
        # Denoise
        pred_noise = model(noisy_images, timesteps)
        reconstructed = (noisy_images - (1 - alpha) * pred_noise) / alpha
    
    # Denormalize untuk display
    def denormalize(tensor):
        return torch.clamp((tensor * 0.5) + 0.5, 0, 1)
    
    # Plot results
    fig, axes = plt.subplots(3, len(real_images), figsize=(15, 9))
    
    if len(real_images) == 1:
        axes = axes.reshape(3, 1)  # Handle single image case
    
    for i in range(len(real_images)):
        # Original
        axes[0, i].imshow(denormalize(real_batch[i]).cpu().permute(1, 2, 0))
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Noisy
        axes[1, i].imshow(denormalize(noisy_images[i]).cpu().permute(1, 2, 0))
        axes[1, i].set_title('Noisy')
        axes[1, i].axis('off')
        
        # Reconstructed
        axes[2, i].imshow(denormalize(reconstructed[i]).cpu().permute(1, 2, 0))
        axes[2, i].set_title('Reconstructed')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_reconstruction_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate metrics
    mse = torch.mean((real_batch - reconstructed) ** 2).item()
    print(f"Reconstruction MSE: {mse:.6f}")
    print("✓ Reconstruction test completed!")

# Jalankan test
model_path = "model_epoch_100.pth"
data_folder = "/content/Maya-model-1/data/my_cats/cats_only"
simple_reconstruction_test(model_path, data_folder)
