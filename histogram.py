import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_and_plot_normalized_hue_histogram(image, num_bins=180):
    """
    Calculate and plot a normalized histogram of the Hue channel in the HSV color space for an image.

    Args:
        image (numpy.ndarray): The input image (BGR or RGB).
        num_bins (int): The number of bins for the histogram (default is 180, covering the full range of hues).
    """
    # Convert the image to HSV format
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract the Hue channel
    hue_channel = hsv_image[:, :, 0]

    # Calculate the histogram of the Hue channel
    histogram, _ = np.histogram(hue_channel, bins=num_bins, range=(0, 180))

    # Normalize the histogram
    histogram = histogram / np.sum(histogram)

    # Plot the normalized histogram
    plt.figure(figsize=(8, 6))
    plt.title('Normalized Hue Channel Histogram')
    plt.xlabel('Hue Value')
    plt.ylabel('Normalized Frequency')
    plt.xlim(0, 180)
    plt.plot(histogram, label='Hue Channel')
    plt.legend()
    plt.show()

# Load an example image (you should replace this with your own image)
image_path = 'imgs/1_Carpet013_2K_Color.jpg'
image = cv2.imread(image_path)

# Calculate and plot the color histogram
calculate_and_plot_normalized_hue_histogram(image)
