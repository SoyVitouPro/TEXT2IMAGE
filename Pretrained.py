import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pickle

# 🔹 Set the correct Khmer font (Update the path for your system)
# khmer_font_path = "/usr/share/fonts/truetype/khmeros/KhmerOS.ttf"  # Linux (Ubuntu)
khmer_font_path = "C:/Users/jingl/Downloads/All-Khmer-Fonts-9-26-15/KhmerOSsiemreap.ttf"  # Windows
# khmer_font_path = "/Library/Fonts/NotoSansKhmer-Regular.ttf"  # Mac

# Load and set Khmer-compatible font
if os.path.exists(khmer_font_path):
    khmer_font = fm.FontProperties(fname=khmer_font_path, size=14)
    plt.rcParams["font.family"] = khmer_font.get_name()
else:
    print("⚠️ Khmer font not found! Using default font.")

# Function to preprocess the image
def preprocess_image(image_path):
    """Resize image to 32x32 and convert to grayscale."""
    try:
        img = Image.open(image_path)
        img = img.resize((32, 32))  # Resize to 32x32
        img = img.convert('L')  # Convert to grayscale
        return np.array(img)
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

# Function to load and preprocess the dataset
def load_dataset(image_folder, label_file):
    """Load images and labels from the dataset."""
    images = []
    labels = []
    filenames = []  # Store filenames separately

    # Read the label file with UTF-8 encoding
    try:
        with open(label_file, 'r', encoding="utf-8", errors="replace") as file:
            lines = file.readlines()
    except UnicodeDecodeError as e:
        print(f"❌ Encoding error in file {label_file}: {e}")
        return images, labels, filenames

    # Load and preprocess each image
    for line in lines:
        try:
            parts = line.strip().split()
            if len(parts) < 2:
                print(f"⚠️ Skipping invalid line: {line.strip()}")
                continue

            image_name, label = parts[0], " ".join(parts[1:])  # Allow multi-word labels
            image_path = os.path.join(image_folder, image_name)

            img = preprocess_image(image_path)
            if img is not None:
                images.append(img)  # Store NumPy array
                labels.append(label)
                filenames.append(image_name)  # Store filename separately
        except ValueError as e:
            print(f"⚠️ Skipping line due to format issue: {line.strip()} ({e})")

    return images, labels, filenames

# Function to display images inline with Khmer labels
def display_images(images, labels, filenames, dataset_name):
    """Display the first 5 images with correct Khmer labels."""
    print(f"\n📌 Displaying first 5 images for {dataset_name}:")

    num_images = min(5, len(images))
    fig, axes = plt.subplots(1, num_images, figsize=(10, 5))

    if num_images == 1:
        axes = [axes]  # Ensure list compatibility for single image

    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')  # Display the image
        ax.set_title(f"Label: {labels[i]}\n{filenames[i]}", fontproperties=khmer_font, fontsize=12)
        ax.axis('off')

    plt.show()

# Paths to the train and test datasets
train_image_folder = 'C:/thesis/Code/after_generation_font_char'
train_label_file = 'C:/thesis/Code/train_original_char.txt'

test_image_folder = 'C:/thesis/Code/after_generation_font_char_test'
test_label_file = 'C:/thesis/Code/train_original_char_test.txt'

# Load and preprocess the train dataset
train_images, train_labels, train_filenames = load_dataset(train_image_folder, train_label_file)

# Load and preprocess the test dataset
test_images, test_labels, test_filenames = load_dataset(test_image_folder, test_label_file)

# ✅ Save the processed data using pickle (.pkl format)
with open("train_data.pkl", "wb") as f:
    pickle.dump((train_images, train_labels), f)

with open("test_data.pkl", "wb") as f:
    pickle.dump((test_images, test_labels), f)

print("✅ Preprocessed data saved successfully as 'train_data.pkl' and 'test_data.pkl'.")

# Display images with correct Khmer labels
display_images(train_images, train_labels, train_filenames, 'Train Dataset')
display_images(test_images, test_labels, test_filenames, 'Test Dataset')
