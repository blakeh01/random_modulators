import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile

import tools.eye_diagram

# => DATA GENERATION - PULSE TRAINS AND PULSE SHAPING

Fs = 48000  # Sampling rate
Tsym = 1 / 4800  # Symbol period
L = int(Fs * Tsym)  # Up-sampling rate, L samples per symbol
f_c = 5000

# Generate random bits
with open('../input.txt', 'r') as file:
    text = file.read()

ascii_bits = np.unpackbits(np.array([ord(c) for c in text], dtype=np.uint8))
sync_symbols = np.ones(100)  # 100 sync symbols
sync_bits = np.random.randint(0, 2, 500)   # 500 random alternating 1's and 0's (todo: change, chance @ getting preamble)

# Combine sync symbols, sync bits, and data preamble
sync_sequence = np.concatenate((sync_symbols, sync_bits, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]))
print(sync_sequence)

# Combine sync sequence and ASCII bits
all_bits = np.concatenate((sync_sequence, ascii_bits))

num_symbols = len(all_bits) // 2

pulse_train_I = np.array([])
pulse_train_Q = np.array([])

for i in range(0, len(all_bits), 2):
    # Map the pair of bits to I and Q channels
    bit_I = all_bits[i] * 2 - 1
    bit_Q = all_bits[i + 1] * 2 - 1

    # Create pulses for I and Q channels
    pulse_I = np.zeros(L)
    pulse_Q = np.zeros(L)

    pulse_I[0] = bit_I
    pulse_Q[0] = bit_Q

    pulse_train_I = np.concatenate((pulse_train_I, pulse_I))
    pulse_train_Q = np.concatenate((pulse_train_Q, pulse_Q))

# Combining I and Q channels to get QPSK signal
sig = pulse_train_I + 1j * pulse_train_Q

plt.figure(0)
plt.subplot(1, 2, 1)
plt.stem(np.real(sig))
plt.title("Generated I")
plt.subplot(1, 2, 2)
plt.stem(np.imag(sig))
plt.title("Generated Q")
plt.grid(True)
plt.show()

# Create our raised-cosine filter
num_taps = 101
beta = 0.6
t = np.arange(num_taps) - (num_taps - 1) // 2
h_rcc = np.sinc(t / L) * np.cos(np.pi * beta * t / L) / (1 - (2 * beta * t / L) ** 2)
plt.figure(1)
plt.plot(t, h_rcc, '.')
plt.title("RRC Filter Response")
plt.grid(True)
plt.show()

# Match filter both I and Q

# RRC Matched Filter for I channel
samples_I = np.convolve(sig.real, h_rcc, 'full')

# RRC Matched Filter for Q channel
samples_Q = np.convolve(sig.imag, h_rcc, 'full')

plt.figure(2)

plt.subplot(1, 2, 1)
plt.plot(samples_I, '.-')
for i in range(num_symbols):
    plt.plot([i * L + num_taps // 2, i * L + num_taps // 2], [0, samples_I[i * L + num_taps // 2]])
plt.grid(True)
plt.title("RRC Filtered I")

plt.subplot(1, 2, 2)
plt.plot(samples_Q, '.-')
for i in range(num_symbols):
    plt.plot([i * L + num_taps // 2, i * L + num_taps // 2], [0, samples_Q[i * L + num_taps // 2]])
plt.grid(True)
plt.title("RRC Filtered Q")
plt.show()

# create I + Q

# add carrier wave
t = np.linspace(0, len(samples_I) / Fs, len(samples_I))
carrier_I = np.cos(2 * np.pi * f_c * t)  # Carrier wave for I component
carrier_Q = np.sin(2 * np.pi * f_c * t)  # Carrier wave for Q component

qpsk_signal = samples_I * carrier_I + samples_Q * carrier_Q

# cap volume to 1
adj_samples = np.divide(qpsk_signal, np.max(qpsk_signal))
attenuation_factor = 1 / np.max(qpsk_signal)  # plus other factors ie. volume?

print(np.shape(adj_samples))
print("Attenuated signal by: ", attenuation_factor)
wavfile.write('QPSK_modulated_wave.wav', Fs, adj_samples)
print("Wrote .wav file!")

# FFT of the samples array
fft_result = np.fft.fft(qpsk_signal)
fft_result_shifted = np.fft.fftshift(fft_result)

# frequency axis
N = len(qpsk_signal)  # Number of samples
frequencies = np.fft.fftshift(np.fft.fftfreq(N, d=1 / Fs))

# Plot the magnitude spectrum
plt.plot(frequencies, np.abs(fft_result_shifted))
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()