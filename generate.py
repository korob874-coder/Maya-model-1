import torch
from model import SimpleUNet
import matplotlib.pyplot as plt
import numpy as np

class SimpleGenerator:
    def __init__(self, model_path, timesteps=1000):
        self.model = SimpleUNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.timesteps = timesteps
        self.beta = torch.linspace(1e-4, 0.02, timesteps)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
    
    def generate(self, image_size=64, num_images=1):
        with torch.no_grad():
            # Start from random noise
            x = torch.randn(num_images, 3, image_size, image_size)
            
            # Reverse process
            for t in range(self.timesteps-1, -1, -1):
                t_batch = torch.full((num_images,), t, dtype=torch.long)
                
                # Predict noise
                pred_noise = self.model(x, t_batch)
                
                # Remove some noise
                alpha_t = self.alpha[t]
                beta_t = self.beta[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                
                x = (1 / torch.sqrt(alpha_t)) * (
                    x - ((1 - alpha_t) / torch.sqrt(1 - self.alpha_cumprod[t])) * pred_noise
                ) + torch.sqrt(beta_t) * noise
            
            return x

def denormalize_and_clip(tensor):
    """Convert from model output range to displayable range"""
    # Model output biasanya dalam range [-1, 1] atau variasi lainnya
    # Normalize ke [0, 1] untuk display
    if tensor.min() < -1.0 or tensor.max() > 1.0:
        # Jika range melebihi [-1, 1], lakukan min-max normalization
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    else:
        # Standard denormalization untuk range [-1, 1]
        tensor = (tensor + 1) / 2
    
    # Clip ke range [0, 1] untuk menghindari warning
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor

def visualize_images(images, filename='generated_images.png'):
    """Visualize images dengan normalisasi yang benar"""
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Process image untuk display
            img = images[i].permute(1, 2, 0)  # CHW -> HWC
            img = denormalize_and_clip(img)
            
            # Convert to numpy untuk matplotlib
            img_np = img.cpu().numpy()
            
            # Pastikan tidak ada nilai di luar [0, 1]
            if img_np.min() < 0 or img_np.max() > 1:
                img_np = np.clip(img_np, 0, 1)
            
            ax.imshow(img_np)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Generated images saved to {filename}")
    plt.show()

# Usage
if __name__ == "__main__":
    # Coba beberapa model checkpoint untuk perbandingan
    model_paths = [
        'model_epoch_50.pth',
        'model_epoch_100.pth', 
        'cat_model.pth'  # model lama
    ]
    
    for model_path in model_paths:
        try:
            print(f"\n=== Generating with {model_path} ===")
            generator = SimpleGenerator(model_path)
            generated_images = generator.generate(num_images=4)
            
            # Save dengan nama yang berbeda untuk setiap model
            filename = f'generated_{model_path.replace(".pth", "")}.png'
            visualize_images(generated_images, filename)
            
        except Exception as e:
            print(f"Error with {model_path}: {e}")
            continue
    
    # Generate dengan model terbaru untuk testing lebih lanjut
    print("\n=== Testing Latest Model ===")
    latest_generator = SimpleGenerator('model_epoch_100.pth')
    
    # Generate multiple sets untuk melihat konsistensi
    for i in range(3):
        generated_images = latest_generator.generate(num_images=4)
        filename = f'generated_test_set_{i+1}.png'
        visualize_images(generated_images, filename)
        
        # Print stats untuk debugging
        img_tensor = generated_images[0]
        print(f"Set {i+1} - Range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        print(f"Set {i+1} - Mean: {img_tensor.mean():.3f}, Std: {img_tensor.std():.3f}")
