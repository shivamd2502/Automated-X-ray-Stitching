#Project Summary:

This project demonstrates how to stitch two X-ray images together using feature matching and deep learning feature extraction. The images are first processed using a deep learning model to extract feature vectors, and these features are then matched using the SIFT algorithm from OpenCV for accurate alignment. The project utilizes PyTorch for feature extraction, where the images are preprocessed and passed through a modified ResNet18 model (with the classification layer removed to output feature vectors) to extract features. The OpenCV Stitcher is used to combine the images based on the matched features. To evaluate image similarity, the project includes the use of SSIM (Structural Similarity Index) to measure the similarity between the images and pixel density measuring to assess the spatial density of features. The results are displayed visually with feature matches and the stitched X-ray image, with the stitched image saved for output. This project serves as a proof of concept for image stitching using deep learning in medical imaging, providing insights into image similarity and alignment techniques.

---------------------------------------------------------------------------------------------------------------------

#Dependencies:

OpenCV: For image processing, including reading images, feature detection, and image stitching.

NumPy: For numerical operations on arrays and matrices.

PyTorch: For deep learning feature extraction using a pre-trained model.

Torchvision: To access pre-trained models and image transformations.

Matplotlib: For visualizing images and feature matches.

---------------------------------------------------------------------------------------------------------------------

#You can install the required packages using pip:

pip install opencv-python numpy torch torchvision matplotlib

---------------------------------------------------------------------------------------------------------------------

#Important Notes:

1> Make sure to open the 'xray1' and 'xray2' images before running the code. This is to load the images in your device

2>You can run the 'SSIM' and 'pixel_density' code only after running the 'xray_stitching' code. after running 'xray_stitching', a stitched_output image is generated which is required for running 'SSIM' and 'pixel_density' code.

3>for using the dataset you will have to first breake the images in 2 verticle segments