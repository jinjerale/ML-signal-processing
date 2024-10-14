import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import IPython.display as ipd

# Load the audio files into a 20 x [time samples] matrix
x = []
for ii in range(1,21):
    data, rate = sf.read('data/x_ica_{}.wav'.format(ii))
    x.append(data)
x = np.array(x)

# Compute covariance matrix and its eigenvalues/eigenvectors
cov_matrix = np.cov(x)
V, U = np.linalg.eig(cov_matrix)
print(V.shape)

print(V) # only the first three value is greater than 1e-5

# Plot the eigenvalues
plt.figure()
plt.plot(V)
plt.show()

# Whiten the data
W = U / np.sqrt(V[None,:])
print(W.shape)
z = W[:, :3].T@x


# ICA Implementation
Wica = np.eye(3)
rho = 0.000001
maxIter = 500
dWSum = np.zeros(maxIter)
l = z.shape[1] # l = N

for ii in range(maxIter):
    y = Wica@z
    dW = rho * (l * np.eye(3) - np.dot(np.tanh(y), np.power(y, 3).T)) @ Wica
    # y = k * N,
    Wica += dW
    dWSum[ii] = np.sum(dW ** 2)

# Plot the convergence curve
plt.figure()
plt.plot(dWSum)
plt.show()
