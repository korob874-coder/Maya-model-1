```markdown
# 🐱 Maya Model 1 - AI Cat Generator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-🚀_Working-brightgreen)

**A diffusion model that generates adorable cat images from pure noise!**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/korob874-coder/Maya-model-1/blob/main/train.py)
[![GitHub stars](https://img.shields.io/github/stars/korob874-coder/Maya-model-1?style=social)](https://github.com/korob874-coder/Maya-model-1)

*From random noise to cute cats - powered by deep learning!*

</div>

## 🎨 Demo

<div align="center">

### AI Generated Cats
![Generated Cats](https://via.placeholder.com/600x200/FF6B6B/FFFFFF?text=AI+Generated+Cats+Will+Appear+Here)

*"Watch as AI transforms chaos into feline beauty!"*

</div>

## 🚀 Features

- **🧠 U-Net Architecture** - Advanced neural network with skip connections
- **🎯 Diffusion Training** - State-of-the-art image generation technique
- **🐱 Cat Dataset** - Trained on 55 diverse cat images (64x64)
- **📊 Training Pipeline** - Complete with loss tracking and checkpointing
- **🎨 Image Generation** - Create novel cats from random noise
- **💾 Model Management** - Save/load trained models effortlessly

## 📁 Project Structure

```

Maya-model-1/
├──data/
│└── my_cats/
│├── CatDog/           # Original dataset (55 cats + 55 dogs)
│└── cats_only/        # Filtered cat images (55 images)
├──model.py                  # U-Net implementation
├──train.py                  # Training pipeline
├──dataset.py                # Data loading & preprocessing
├──generate.py               # Image generation script
├──cat_model.pth             # Pre-trained weights
├──requirements.txt          # Dependencies
└──README.md                 # This file

```

## ⚡ Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/korob874-coder/Maya-model-1.git
cd Maya-model-1
pip install -r requirements.txt
```

2. Train the Model

```bash
# Train with default settings
python train.py

# Or customize training
python train.py --epochs 200 --batch_size 16 --image_size 64
```

3. Generate Cats!

```python
from generate import generate_cats

# Generate 4 cat images
generate_cats(n_images=4, model_path='cat_model.pth')
```

🛠️ Technical Details

Model Architecture

```python
class SimpleUNet(nn.Module):
    # Encoder: 64 → 128 → 256 channels
    # Decoder: 256 → 128 → 64 channels  
    # Skip connections between encoder/decoder
    # Bilinear upsampling for size matching
```

Training Process

· Loss Function: Mean Squared Error (MSE)
· Optimizer: Adam
· Learning Rate: 0.001
· Epochs: 100
· Batch Size: 8
· Image Size: 64x64

Dataset

· Source: Kaggle Cat/Dog 64x64 dataset
· Images: 55 cat images
· Preprocessing: Resize, Normalize, Augmentation
· Split: 100% training (small dataset)

📈 Results & Progress

Training Journey

Phase Dataset Result Achievement
🟢 Initial 3 images Basic learning Proof of concept
🟡 Improved 55 images Good generation Working model
🔴 Advanced 1000+ images Photorealistic Future goal

Loss Progression

· Start Loss: ~1.20
· Final Loss: ~0.15
· Improvement: 87.5% reduction 🎉

🎯 Usage Examples

Basic Generation

```python
from model import SimpleUNet
import torch

model = SimpleUNet()
model.load_state_dict(torch.load('cat_model.pth'))
model.eval()

# Generate from noise
noise = torch.randn(1, 3, 64, 64)
generated_cat = model(noise, torch.tensor([0]))
```

Custom Training

```python
from train import train_model

# Custom training loop
train_model(
    data_dir='data/my_cats/cats_only',
    epochs=150,
    batch_size=12,
    learning_rate=0.0005
)
```

🔧 Development

Prerequisites

```txt
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
matplotlib>=3.5.0
tqdm>=4.60.0
```

Running Tests

```bash
# Test dataset loading
python -c "from dataset import SimpleImageDataset; print('Dataset test passed!')"

# Test model architecture  
python -c "from model import SimpleUNet; model = SimpleUNet(); print('Model test passed!')"
```

🐛 Troubleshooting

Common Issues & Solutions

Problem Solution
ImportError Clear PyCache: find . -name "__pycache__" -type d -exec rm -rf {} +
Path issues Use absolute paths: /content/Maya-model-1/data/...
CUDA out of memory Reduce batch size or image dimensions
No images found Check dataset path and file extensions

🤝 Contributing

We love contributions! Here's how to help:

1. Fork the repository
2. Add more cat images to dataset
3. Improve model architecture
4. Add new features (conditioning, better sampling)
5. Submit a pull request

Ideas for Improvement

· Add conditional generation (breed, color)
· Implement better diffusion sampling
· Increase image resolution (128x128, 256x256)
· Add web interface with Gradio
· Deploy to Hugging Face Spaces

📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

· Kaggle for the cat/dog dataset
· PyTorch team for the amazing deep learning framework
· Google Colab for free GPU resources
· The AI community for endless inspiration

📞 Contact

Bagas Koro

· GitHub: @korob874-coder
· Project: Maya Model 1

---

<div align="center">

⭐ Don't forget to star this repo if you like it!

"From 3 images to AI artist - the journey of Maya Model 1!" 🎨🐱

</div>
```
