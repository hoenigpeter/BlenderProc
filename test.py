import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Load NOCS image (PNG, RGB in [0, 255])
nocs_img = imageio.imread("test.png").astype(np.float32) / 255.0  # Normalize to [0, 1]

# Reshape into point cloud (H*W, 3)
h, w, _ = nocs_img.shape
points = nocs_img.reshape(-1, 3)

# Optionally remove background (where all coords are 0)
mask = ~(np.all(points == 0, axis=1))
points = points[mask]

# Compute 3D bounding box
min_pt = points.min(axis=0)
max_pt = points.max(axis=0)

# Diagonal length
bbox_diagonal = np.linalg.norm(max_pt - min_pt)
print(f"3D bounding box diagonal length: {bbox_diagonal:.4f}")

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D NOCS Point Cloud")
plt.show()
