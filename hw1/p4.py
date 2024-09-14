import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Step 1: Define function to create the DFT matrix
def dft_matrix(N):
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    return W

# Step 2: Define the inverse DFT matrix
def inverse_dft_matrix(N):
    n = np.arange(N)
    k = n.reshape((N, 1))
    W_inv = np.exp(2j * np.pi * k * n / N)
    return W_inv / N  # Normalization by N

# Step 3: STFT - Convert the signal into frames and apply window
def stft(signal, N, hop_size):
    num_frames = 1 + (len(signal) - N) // hop_size
    X = np.zeros((N, num_frames), dtype=complex)

    # Apply Hann window and frame the signal
    hann_window = np.hanning(N)
    for i in range(num_frames):
        frame = signal[i * hop_size:i * hop_size + N] * hann_window
        X[:, i] = np.dot(dft_matrix(N), frame)
    return X

# Step 4: ISTFT - Inverse STFT to reconstruct the time-domain signal
def istft(X, N, hop_size):
    num_frames = X.shape[1]
    signal_reconstructed = np.zeros(N + hop_size * (num_frames - 1))

    hann_window = np.hanning(N)
    for i in range(num_frames):
        frame_reconstructed = np.dot(inverse_dft_matrix(N), X[:, i]).real
        signal_reconstructed[i * hop_size:i * hop_size + N] += frame_reconstructed * hann_window

    return signal_reconstructed

# Step 5: Function to plot the spectrogram
def plot_spectrogram(X, N, hop_size, fs):
    magnitude = np.abs(X)
    time_axis = np.arange(X.shape[1]) * hop_size / fs
    freq_axis = np.arange(N) * fs / N
    plt.pcolormesh(time_axis, freq_axis, magnitude[:, :])
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Magnitude')
    plt.show()

# Load the contaminated signal (x.wav)
signal, fs = sf.read('data/x.wav')
print(f"Signal length: {len(signal)}, Sample rate: {fs}")

# STFT parameters
N = 512  # Frame size
hop_size = N // 2  # 50% overlap

# Step 6: Perform STFT
X = stft(signal, N, hop_size)

# Step 7: Plot the spectrogram
plot_spectrogram(X, N, hop_size, fs)

# Step 8: Manually identify and remove beep frequencies (zero-out rows)
# Assuming we have identified the rows corresponding to beep frequencies manually:
beep_rows = np.arange(1950, 2051)
beep_idx = beep_rows * N // fs
for freq in beep_idx:
    X[beep_idx, :] = 0
    X[N - beep_idx - 1, :] = 0  # Conjugate symmetry

# Step 9: Apply inverse STFT to reconstruct the signal
signal_reconstructed = istft(X, N, hop_size)

# Check if signal_reconstructed has any imaginary components
if np.iscomplexobj(signal_reconstructed):
    print("Imaginary components detected, taking the real part.")
    signal_reconstructed = np.real(signal_reconstructed)
else:
    print("No imaginary components detected.")

# Step 10: Save the cleaned signal
sf.write('cleaned_signal.wav', signal_reconstructed, fs)

# Step 11: Plot the spectrogram of the cleaned signal
X_cleaned = stft(signal_reconstructed, N, hop_size)
plot_spectrogram(X_cleaned, N, hop_size, fs)
