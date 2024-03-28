import numpy as np
import matplotlib.pyplot as plt


def plot_mag_fft(sig, Fs, added_title=""):
    fft_result = np.fft.fft(sig)
    fft_result_shifted = np.fft.fftshift(fft_result)

    # frequency axis
    N = len(sig)  # Number of samples
    frequencies = np.fft.fftshift(np.fft.fftfreq(N, d=1 / Fs))

    # Plot the magnitude spectrum
    plt.plot(frequencies, np.abs(fft_result_shifted))
    plt.title('Magnitude Spectrum' + f" {added_title}")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()
