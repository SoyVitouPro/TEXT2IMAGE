import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Khmer text classes
khmer_texts = ['ក', 'ខ', 'គ', 'ឃ', 'ង', 'ច', 'ឆ', 'ជ', 'ឈ', 'ញ',
               'ដ', 'ឋ', 'ឌ', 'ឍ', 'ណ', 'ត', 'ថ', 'ទ', 'ធ', 'ន',
               'ប', 'ផ', 'ព', 'ភ', 'ម', 'យ', 'រ', 'ល', 'វ', 'ស',
               'ហ', 'ឡ', 'អ']

# Custom Dataset Class
class KhmerDataset(Dataset):
    def __init__(self, dataset_folder, annotation_file, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.data = []
        
        with open(annotation_file, "r", encoding="utf-8") as file:
            for line in file:
                image_file, label = line.strip().split(" ")
                if label in khmer_texts:
                    label_idx = khmer_texts.index(label)
                    self.data.append((image_file, label_idx))
                else:
                    print(f"Warning: Unknown label '{label}' in annotation file.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_file, label = self.data[idx]
        image_path = os.path.join(self.dataset_folder, image_file)

        try:
            image = Image.open(image_path).convert("L")  # Convert to grayscale
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None  # Handle missing files

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# Generator Model
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

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, label_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(64 * 64 + label_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        x = torch.cat((images.view(images.size(0), -1), labels), dim=1)  # Flatten images
        return self.model(x)

# Training Function
def train_gan(dataset_folder, annotation_file, epochs=100, batch_size=64, latent_dim=100, lr=0.0002, save_path="generator.pth"):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    dataset = KhmerDataset(dataset_folder, annotation_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim, len(khmer_texts)).to(device)
    discriminator = Discriminator(len(khmer_texts)).to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(epochs):
        for real_images, labels in dataloader:
            if real_images is None:
                continue  # Skip missing images
            
            batch_size = real_images.size(0)
            real_images, labels = real_images.to(device), labels.to(device)
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=len(khmer_texts)).float().to(device)

            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise, labels_one_hot)

            real_loss = criterion(discriminator(real_images, labels_one_hot), torch.ones(batch_size, 1, device=device))
            fake_loss = criterion(discriminator(fake_images.detach(), labels_one_hot), torch.zeros(batch_size, 1, device=device))
            d_loss = real_loss + fake_loss

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            gen_loss = criterion(discriminator(fake_images, labels_one_hot), torch.ones(batch_size, 1, device=device))
            optimizer_G.zero_grad()
            gen_loss.backward()
            optimizer_G.step()

        print(f"Epoch {epoch+1}/{epochs} - D Loss: {d_loss.item():.4f}, G Loss: {gen_loss.item():.4f}")
        if epoch == 50:
            torch.save(generator.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            
    torch.save(generator.state_dict(), save_path)
    print(f"Model saved to {save_path}")


train_gan(r"D:\Soy Vitou\GITHUB\TEXT2IMAGE\dataset_generated\dataset", 
          r"D:\Soy Vitou\GITHUB\TEXT2IMAGE\dataset_generated\annotation.txt", epochs=200)
