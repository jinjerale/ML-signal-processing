import numpy as np
import matplotlib.pyplot as plt

# Power Iteration routine to estimate the eigenvector corresponding to the largest eigenvalue
def power_iteration(X, num_simulations: int):
    # Random vector initialization
    b_k = np.random.rand(X.shape[1])

    for _ in range(num_simulations):
        # Calculate the matrix-by-vector product
        b_k1 = np.dot(X, b_k)

        # Normalize the vector
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm

    return b_k

# Load flute.npy
X = np.load('data/flute.npy')
print(f"Matrix X shape: {X.shape}")

# Plot the matrix X using a color map
plt.imshow(X, cmap='hot', aspect='auto')
plt.colorbar(label='Intensity')
plt.title('Matrix Representation of Two Flute Notes')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

# 1. Compute the sample covariance matrix (X is 128x142)
X_mean = X - np.mean(X, axis=1, keepdims=True)
cov_matrix = np.cov(X_mean)

print(f"Sample covariance matrix shape: {cov_matrix.shape}")

# 2. Estimate two eigenvectors using Power Iteration
num_iterations = 1000
eigenvector1 = power_iteration(cov_matrix, num_iterations)

# Deflate the covariance matrix to find the second eigenvector
cov_matrix_deflated = cov_matrix - np.outer(eigenvector1, eigenvector1) * np.dot(eigenvector1, np.dot(cov_matrix, eigenvector1))
eigenvector2 = power_iteration(cov_matrix_deflated, num_iterations)

# Plot the two eigenvectors
# plt.plot(eigenvector1, label='Eigenvector 1')
# plt.plot(eigenvector2, label='Eigenvector 2')
# plt.legend()
# plt.title('Estimated Eigenvectors (Representative Spectra)')
# plt.xlabel('Frequency Element')
# plt.ylabel('Eigenvector Value')
# plt.show()
print(f"e1 {eigenvector1}")
print(f"e2 {eigenvector2}")

# 3. Compute the temporal activation vectors for the two notes
# Temporal activation is obtained by projecting the original data X onto the eigenvectors.
activation_vector1 = np.dot(eigenvector1, X)
activation_vector2 = np.dot(eigenvector2, X)

# Plot the temporal activation vectors
# plt.plot(activation_vector1, label='Activation Vector 1')
# plt.plot(activation_vector2, label='Activation Vector 2')
# plt.legend()
# plt.title('Temporal Activation Vectors')
# plt.xlabel('Time')
# plt.ylabel('Activation Value')
# plt.show()
print(f"a1 {activation_vector1}")
print(f"a2 {activation_vector2}")

# 4. Alternative Approach: Perform power iteration on the transpose of X (X.T)
# Compute the covariance matrix for X^T (which will be 142x142)
cov_matrix_transpose = np.cov(X_mean.T)

# Perform power iteration twice to estimate two eigenvectors
eigenvector1_T = power_iteration(cov_matrix_transpose, num_iterations)

# Deflate to get the second eigenvector
cov_matrix_transpose_deflated = cov_matrix_transpose - np.outer(eigenvector1_T, eigenvector1_T) * np.dot(eigenvector1_T, np.dot(cov_matrix_transpose, eigenvector1_T))
eigenvector2_T = power_iteration(cov_matrix_transpose_deflated, num_iterations)

# Plot the two eigenvectors from the transposed matrix
plt.plot(eigenvector1_T, label='Eigenvector 1 (Transpose)')
plt.plot(eigenvector2_T, label='Eigenvector 2 (Transpose)')
plt.legend()
plt.title('Eigenvectors from Transposed Matrix')
plt.xlabel('Dimension')
plt.ylabel('Eigenvector Value')
plt.show()

# 5. Recover representative spectra from transposed matrix eigenvectors
# Representative spectra are projections of X.T onto the eigenvectors obtained from X^T.
representative_spectrum1 = np.dot(X, eigenvector1_T)
representative_spectrum2 = np.dot(X, eigenvector2_T)

#
