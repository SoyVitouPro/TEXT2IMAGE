import torch
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

# Load KhmerDataset class from the existing code
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

        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset
dataset_folder = r"D:\Soy Vitou\GITHUB\TEXT2IMAGE\dataset_generated\dataset"
annotation_file = r"D:\Soy Vitou\GITHUB\TEXT2IMAGE\dataset_generated\annotation.txt"
dataset = KhmerDataset(dataset_folder, annotation_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Display some images with their labels
for i, (image, label) in enumerate(dataloader):
    if image is None:
        continue
    
    label_text = khmer_texts[label.item()]
    print(f"Label: {label_text}")
    
    image_np = image.squeeze().numpy()  # Convert tensor to numpy
    plt.imshow(image_np, cmap='gray')
    plt.title(f"Khmer Character: {label_text}")
    plt.show()
    
    if i == 4:  # Display only 5 images
        break
