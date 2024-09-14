import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

# Load the audio files
x1, rate1 = sf.read('data/x1.wav')
x2, rate2 = sf.read('data/x2.wav')
s, rate_s = sf.read('data/s.wav')

# Standardize the signals (zero mean and unit variance)
def standardize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

x1_standardized = standardize(x1)
x2_standardized = standardize(x2)
s_standardized = standardize(s)

# Calculate kurtosis
kurtosis_x1 = kurtosis(x1_standardized, fisher=False)
kurtosis_x2 = kurtosis(x2_standardized, fisher=False)
kurtosis_s = kurtosis(s_standardized, fisher=False)

print(f'Kurtosis of x1.wav: {kurtosis_x1}')
print(f'Kurtosis of x2.wav: {kurtosis_x2}')
print(f'Kurtosis of s.wav (ground truth): {kurtosis_s}')

# Plot histograms
def plot_histogram(signal, title):
    plt.hist(signal, bins=100, density=True, alpha=0.6, color='b')
    plt.title(title)
    plt.xlabel('Amplitude')
    plt.ylabel('Density')
    plt.show()

# Plot histograms for comparison
plot_histogram(x1_standardized, 'Histogram of x1.wav')
plot_histogram(x2_standardized, 'Histogram of x2.wav')
plot_histogram(s_standardized, 'Histogram of s.wav (ground truth)')

# Determine which one is less Gaussian-like
if kurtosis_x1 > kurtosis_x2:
    print("x1.wav is less Gaussian-like (fewer interfering sources).")
else:
    print("x2.wav is less Gaussian-like (fewer interfering sources).")
