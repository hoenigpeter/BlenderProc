import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the path to your dataset directory
dataset_path = 'imgs2'

# List all image files in the dataset directory
image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]

# Initialize variables to accumulate frequency spectra
accumulated_spectrum = None

# Loop through each image in the dataset
for image_file in image_files:
    # Load and preprocess the image
    image = cv2.imread(os.path.join(dataset_path, image_file), cv2.IMREAD_GRAYSCALE)
    
    # Perform a 2D Fourier Transform
    f_transform = np.fft.fft2(image)
    
    # Shift the frequency spectrum
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)
    
    # Accumulate the magnitude spectra
    if accumulated_spectrum is None:
        accumulated_spectrum = magnitude_spectrum
    else:
        accumulated_spectrum += magnitude_spectrum

# Calculate the mean magnitude spectrum (overall frequency distribution)
mean_spectrum = accumulated_spectrum / len(image_files)

# Display the combined frequency spectrum
plt.figure(figsize=(8, 8))
plt.imshow(np.log(mean_spectrum), cmap='gray')
plt.title('Overall Frequency Distribution')
plt.colorbar()
plt.axis('off')
plt.show()
