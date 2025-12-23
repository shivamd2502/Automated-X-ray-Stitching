# import cv2
# import numpy as np
# from sewar.full_ref import vifp, uqi

# def calculate_vif(img1, img2):
#     return vifp(img1, img2)

# def calculate_uiqi(img1, img2):
#     return uqi(img1, img2)

# def preprocess_image(img):
#     if len(img.shape) == 3:  # Convert to grayscale if needed
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return img

# def main():
#     # Load original and stitched X-ray images
#     original_img = cv2.imread('/home/shivam/Downloads/fc68eedproject_files/project files/xray_image_stitching/stock_image_2.jpg')  # Original Image
#     stitched_img = cv2.imread('stitched_xray_output.jpg')  # Stitched Image
    
#     if original_img is None or stitched_img is None:
#         print("Error: Could not read one or more images.")
#         return

#     # Preprocessing to grayscale
#     original_img_gray = preprocess_image(original_img)
#     stitched_img_gray = preprocess_image(stitched_img)

#     # Calculate VIF and UIQI
#     vif_value = calculate_vif(original_img_gray, stitched_img_gray)
#     uiqi_value = calculate_uiqi(original_img_gray, stitched_img_gray)

#     print("Visual Information Fidelity (VIF):", vif_value)
#     print("Universal Image Quality Index (UIQI):", uiqi_value)

# if __name__ == "__main__":
#     main()





import cv2
import numpy as np
from sewar.full_ref import vifp, uqi

def calculate_vif(img1, img2):
    return vifp(img1, img2)

def calculate_uiqi(img1, img2):
    return uqi(img1, img2)

def preprocess_image(img):
    if len(img.shape) == 3:  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def resize_to_match(img, target_img):
    return cv2.resize(img, (target_img.shape[1], target_img.shape[0]))

def main():
    # Load original and stitched X-ray images
    original_img = cv2.imread('/home/shivam/Downloads/fc68eedproject_files/project files/xray_image_stitching/stock_image_2.jpg')  
    stitched_img = cv2.imread('stitched_xray_output.jpg')  

    if original_img is None or stitched_img is None:
        print("Error: Could not read one or more images.")
        return

    # Preprocessing to grayscale
    original_img_gray = preprocess_image(original_img)
    stitched_img_gray = preprocess_image(stitched_img)

    # Resize stitched image to match the original image
    stitched_img_gray_resized = resize_to_match(stitched_img_gray, original_img_gray)

    # Calculate VIF and UIQI
    vif_value = calculate_vif(original_img_gray, stitched_img_gray_resized)
    uiqi_value = calculate_uiqi(original_img_gray, stitched_img_gray_resized)

    print("Visual Information Fidelity (VIF):", vif_value)
    print("Universal Image Quality Index (UIQI):", uiqi_value)

if __name__ == "__main__":
    main()




