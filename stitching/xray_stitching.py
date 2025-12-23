import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

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

# Display feature matches using SIFT
def show_feature_matches(img1, img2):
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches
    match_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], None)
    cv2.imshow("Feature Matches", match_img)
    cv2.imwrite("feature_matches.jpg", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    #img1 = cv2.imread("xray1.jpg")
    #img2 = cv2.imread("xray2.jpg")
    img1 = cv2.imread("/home/shivam/Downloads/fc68eedproject_files/project files/xray_image_stitching/xray3.jpg")
    img2 = cv2.imread("/home/shivam/Downloads/fc68eedproject_files/project files/xray_image_stitching/xray4.jpg")


    if img1 is None or img2 is None:
        print("Error: Could not read one or both images.")
        return

    # Display feature matches
    show_feature_matches(img1, img2)

    # Preprocess images
    preprocessed_img1 = preprocess_image(img1)
    preprocessed_img2 = preprocess_image(img2)

    # Load pre-trained PyTorch model
    print("Loading PyTorch model for feature extraction...")
    #device = torch.device("hpu" if torch.cuda.is_available() else "cpu")  # Use Gaudi (hpu) if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()  # Remove classification layer to use as feature extractor
    model = model.to(device)
    model.eval()

    # Extract features
    features1 = extract_features_with_pytorch(preprocessed_img1, model, device)
    features2 = extract_features_with_pytorch(preprocessed_img2, model, device)

    # Debug: Print feature dimensions
    print("Features1 shape:", features1.shape)
    print("Features2 shape:", features2.shape)

    # Stitch the original images using OpenCV
    images = [img1, img2]
    stitched_image = stitch_images(images)

    # Display and save the stitched image
    if stitched_image is not None:
        cv2.imshow("Stitched X-ray Image", stitched_image)
        cv2.imwrite("stitched_xray_output.jpg", stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image stitching failed.")

if __name__ == "__main__":
    main()
