import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2):
   
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
   
    ssim_value, _ = ssim(img1, img2, full=True)
    return ssim_value

def resize_to_match(img, target_img):
    
    return cv2.resize(img, (target_img.shape[1], target_img.shape[0]))

def calculate_pixel_density(image):
    
    height, width = image.shape[:2]
    total_pixels = height * width
   
    reference_area = 1024 * 768
    pixel_density = total_pixels / reference_area
    return pixel_density

def stitch_images(images):
   
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        return stitched_image
    else:
        print("Error during stitching:", status)
        return None

def main():
   
    xray_img1 = cv2.imread('/home/shivam/Downloads/fc68eedproject_files/project files/xray_image_stitching/xray3.jpg')
    xray_img2 = cv2.imread('/home/shivam/Downloads/fc68eedproject_files/project files/xray_image_stitching/xray4.jpg')
    #output_stitched_image = cv2.imread('stitched_xray_output.jpg')
    output_stitched_image = cv2.imread('/home/shivam/Downloads/fc68eedproject_files/project files/xray_image_stitching/stock_image_2.jpg')

    if xray_img1 is None or xray_img2 is None or output_stitched_image is None:
        print("Error: Could not read one or more images.")
        return

   
    reference_stitched_image = stitch_images([xray_img1, xray_img2])
    if reference_stitched_image is None:
        print("Failed to stitch xray1 and xray2.")
        return

    
    
    output_stitched_image = resize_to_match(output_stitched_image, reference_stitched_image)

    
    ssim_value = calculate_ssim(reference_stitched_image, output_stitched_image)
  
    pixel_density_reference = calculate_pixel_density(reference_stitched_image)
    pixel_density_output = calculate_pixel_density(output_stitched_image)
    
    ssim_threshold = 0.95
    print("SSIM between reference and output stitched image:", ssim_value)
    print("Pixel Density of Reference Stitched Image:", pixel_density_reference)
    print("Pixel Density of Output Stitched Image:", pixel_density_output)

    if ssim_value >= ssim_threshold:
        print("The output stitched image is similar to the reference stitched image.")
    else:
        print("The output stitched image is not similar to the reference stitched image.")

    combined_image = cv2.hconcat([reference_stitched_image, output_stitched_image])
    cv2.imshow("Reference Stitched Image (Left) vs Output Stitched Image (Right)", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
