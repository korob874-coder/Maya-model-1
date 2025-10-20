import torch
from model import SimpleUNet
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import numpy as np

def test_reconstruction_standalone(model_path, data_folder, num_samples=4):
    """Test rekonstruksi gambar - standalone version"""
    
    # Load model
    model = SimpleUNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"✓ Model loaded: {model_path}")
    print(f"Using device: {device}")
    
    # Load images manually
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
    successful_files = []
    for img_path in image_files[:num_samples*2]:  # Load extra for safety
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            real_images.append(img_tensor)
            successful_files.append(os.path.basename(img_path))
            if len(real_images) >= num_samples:
                break
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    if len(real_images) < num_samples:
        print(f"Only {len(real_images)} images loaded successfully")
        num_samples = len(real_images)
    
    real_batch = torch.stack(real_images).to(device)
    print(f"Real images shape: {real_batch.shape}")
    print(f"Loaded images: {successful_files[:num_samples]}")
    
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
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    
    if num_samples == 1:
        axes = axes.reshape(3, 1)  # Handle single image case
    
    for i in range(num_samples):
        # Original
        orig_img = denormalize(real_batch[i]).cpu().permute(1, 2, 0)
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Noisy
        noisy_img = denormalize(noisy_images[i]).cpu().permute(1, 2, 0)
        axes[1, i].imshow(noisy_img)
        axes[1, i].set_title(f'Noisy {i+1}')
        axes[1, i].axis('off')
        
        # Reconstructed
        recon_img = denormalize(reconstructed[i]).cpu().permute(1, 2, 0)
        axes[2, i].imshow(recon_img)
        axes[2, i].set_title(f'Reconstructed {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstruction_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate metrics
    mse = torch.mean((real_batch - reconstructed) ** 2).item()
    psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()
    
    print(f"📊 Evaluation Metrics:")
    print(f"   Reconstruction MSE: {mse:.6f}")
    print(f"   PSNR: {psnr:.2f} dB")
    print(f"   Noise Level: {alpha}")
    print("✅ Reconstruction test completed!")
    
    return mse, psnr

def compare_all_models(data_folder):
    """Bandingkan semua model checkpoint yang ada"""
    model_files = [f for f in os.listdir('.') if f.startswith('model_epoch_') and f.endswith('.pth')]
    model_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    print("🧪 Comparing all models...")
    print(f"Found {len(model_files)} model files: {model_files}")
    
    results = {}
    for model_file in model_files:
        try:
            print(f"\n🔍 Testing {model_file}...")
            mse, psnr = test_reconstruction_standalone(model_file, data_folder, num_samples=2)
            results[model_file] = {'mse': mse, 'psnr': psnr}
        except Exception as e:
            print(f"❌ Failed for {model_file}: {e}")
            results[model_file] = {'mse': float('inf'), 'psnr': 0}
    
    # Print comparison results
    print("\n📈 MODEL COMPARISON RESULTS:")
    print("=" * 50)
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['mse']):
        print(f"{model_name:20} | MSE: {metrics['mse']:.6f} | PSNR: {metrics['psnr']:6.2f} dB")
    
    return results

# Jalankan test
if __name__ == "__main__":
    data_folder = "/content/Maya-model-1/data/my_cats/cats_only"
    
    print("🎯 Choose test mode:")
    print("1. Test single model")
    print("2. Compare all models")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        # Compare semua model
        results = compare_all_models(data_folder)
    else:
        # Test model tertentu
        model_path = "model_epoch_100.pth"  # Ganti dengan model yang ingin di-test
        test_reconstruction_standalone(model_path, data_folder)
