import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image

# Khmer text classes
khmer_texts = ['ក', 'ខ', 'គ', 'ឃ', 'ង', 'ច', 'ឆ', 'ជ', 'ឈ', 'ញ',
               'ដ', 'ឋ', 'ឌ', 'ឍ', 'ណ', 'ត', 'ថ', 'ទ', 'ធ', 'ន',
               'ប', 'ផ', 'ព', 'ភ', 'ម', 'យ', 'រ', 'ល', 'វ', 'ស',
               'ហ', 'ឡ', 'អ']

# Define Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 64),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        x = torch.cat((noise, labels), dim=1)
        return self.model(x).view(-1, 1, 64, 64)

# Load trained generator
def load_generator(model_path, latent_dim=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim, len(khmer_texts)).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    return generator

# Generate and display a specific image
def generate_specific_image(generator, letter='ក', latent_dim=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = torch.randn(1, latent_dim, device=device)
    label_idx = khmer_texts.index(letter)
    label_one_hot = torch.nn.functional.one_hot(torch.tensor([label_idx], device=device), num_classes=len(khmer_texts)).float()
    
    with torch.no_grad():
        fake_image = generator(noise, label_one_hot).cpu()
    
    plt.imshow(fake_image.squeeze(), cmap='gray')
    plt.title(letter)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    generator = load_generator("generator.pth")
    generate_specific_image(generator, letter='យ')
