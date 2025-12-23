import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import time
import matplotlib.pyplot as plt

cv2.setNumThreads(0)

# Preprocess image for PyTorch model
def preprocess_image(img, size=(224, 224)):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess(img).unsqueeze(0)

# Extract features using PyTorch model
def extract_features_with_pytorch(image, model, device):
    image = image.to(device)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy()

# Stitch images using OpenCV
def stitch_images(images):
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        return stitched_image
    else:
        print("Error during stitching:", status)
        return None

def main():
    # Load input images
    img1 = cv2.imread("/home/shivam/Downloads/fc68eedproject_files/project files/xray_image_stitching/xray3.jpg")
    img2 = cv2.imread("/home/shivam/Downloads/fc68eedproject_files/project files/xray_image_stitching/xray4.jpg")

    if img1 is None or img2 is None:
        print("Error: Could not read one or both images.")
        return

    # Define different image sizes (resolutions) to test
    image_sizes = [(224, 224), (256, 256), (512, 512), (1024, 1024)]
    times = []

    # Load pre-trained PyTorch model
    print("Loading PyTorch model for feature extraction...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()  # Remove classification layer to use as feature extractor
    model = model.to(device)
    model.eval()

    # Iterate over each resolution and measure computation time
    for size in image_sizes:
        print(f"\nTesting resolution: {size[0]}x{size[1]}")

        # Measure time for image preprocessing and feature extraction
        start_time = time.time()

        # Preprocess images
        preprocessed_img1 = preprocess_image(img1, size=size)
        preprocessed_img2 = preprocess_image(img2, size=size)

        # Extract features
        features1 = extract_features_with_pytorch(preprocessed_img1, model, device)
        features2 = extract_features_with_pytorch(preprocessed_img2, model, device)

        # Measure the total time
        total_time = time.time() - start_time
        times.append(total_time)

        # Print the time for the current resolution
        print(f"Time taken for {size[0]}x{size[1]} resolution: {total_time:.4f} seconds")

    # After processing all resolutions, plot the results
    resolutions = [f"{size[0]}x{size[1]}" for size in image_sizes]
    plt.figure(figsize=(8, 6))
    plt.plot(resolutions, times, marker='o', color='b')
    plt.xlabel('Image Resolution', fontsize=12)
    plt.ylabel('Computation Time (seconds)', fontsize=12)
    plt.title('Computation Time vs Image Resolution', fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent label overlap
    plt.show()

if __name__ == "__main__":
    main()
