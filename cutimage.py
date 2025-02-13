import os
from PIL import Image

# ✅ Define paths
dataset_folder = "./dataset"
output_folder = "./cropped_dataset"

# ✅ Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# ✅ Process images
for filename in os.listdir(dataset_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Only image files
        img_path = os.path.join(dataset_folder, filename)
        img = Image.open(img_path)

        # 🎯 Check if image size is correct (100x64)
        if img.size != (100, 64):
            print(f"⚠️ Skipping {filename}, incorrect size {img.size}")
            continue

        # 🎯 Crop: Remove 23 pixels from left & right
        cropped_img = img.crop((18, 0, 100 - 18, 64))  # (left, top, right, bottom)

        # ✅ Save cropped image
        cropped_img.save(os.path.join(output_folder, filename))
        print(f"✅ Cropped & saved: {filename}")

print("🎉 Cropping completed!")
