import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load image
image = Image.open('data/siebel.jpg')
image_array = np.array(image)

# Separate RGB channels
R, G, B = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

# # Display image and channels
# plt.imshow(image)
# plt.title("Original Image")
# plt.show()

# You can plot each channel if needed

# Average the three channels to convert to grayscale
X = np.mean(image_array, axis=2)

# # Display the grayscale image
# plt.imshow(X, cmap='gray')
# plt.title("Grayscale Image")
# plt.show()

# Select 8 random consecutive rows
start_row = np.random.randint(0, X.shape[0] - 8)
XR = X[start_row:start_row+8, :]

# plt.imshow(XR, cmap='gray')
# plt.title(f"8x197 Block starting from row {start_row}")
# plt.show()

# Subtract the mean from XR
XR_centered = XR - np.mean(XR, axis=1, keepdims=True)

# Calculate covariance matrix
cov_matrix = np.cov(XR_centered)

# Perform eigendecomposition (PCA)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Plot eigenvectors (rows of W^T)
plt.imshow(eigenvectors, aspect='equal')
plt.colorbar()
plt.title("Eigenvectors from PCA")
plt.show()
