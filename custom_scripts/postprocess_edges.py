from PIL import Image
import numpy as np

# Load the image
input_path = "/home/hoenig/BlenderProc/custom_scripts/blender.png"
output_path = "/home/hoenig/BlenderProc/custom_scripts/binarized.png"

image = Image.open(input_path).convert("RGBA")
image_data = np.array(image)

# Extract RGB channels
r, g, b, a = image_data[:,:,0], image_data[:,:,1], image_data[:,:,2], image_data[:,:,3]
print(np.max(r))
print(np.max(g))
print(np.max(b))
print(np.min(r))
print(np.min(g))
print(np.min(b))
print(np.max(a))

# Compute binary mask: if R, G, B < 0.5 (128 in 8-bit scale), set to white (255), else set to black (0)
binary_mask = ((r == 0) & (g == 0) & (b == 0) & (a == 1)).astype(np.uint8) * 255

# Create an RGB image with black background
binarized_image = np.stack([binary_mask, binary_mask, binary_mask], axis=-1)

# Save the binarized image
Image.fromarray(binarized_image).save(output_path)

print(f"Binarized image saved to {output_path}")