import torch
from model import SimpleUNet
import matplotlib.pyplot as plt

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
            
            # Denormalize
            x = (x * 0.5) + 0.5  # [-1, 1] to [0, 1]
            return x

# Usage
if __name__ == "__main__":
    generator = SimpleGenerator('model_epoch_50.pth')
    generated_images = generator.generate(num_images=4)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i].permute(1, 2, 0))
        ax.axis('off')
    plt.savefig('generated_images.png')
    plt.show()