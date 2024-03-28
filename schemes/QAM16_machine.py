"""
    BPSK example of pulse-trains being shaped using RRC
"""
import random

import numpy as np
import matplotlib.pyplot as plt

import tools.eye_diagram
import sk_dsp_comm.synchronization as sync

from scipy import signal
from scipy.io import wavfile

# => DATA GENERATION - PULSE TRAINS AND PULSE SHAPING

num_symbols = 2400  # Number of symbols
Fs = 48000  # Sampling rate
Tsym = 1 / 4800  # Symbol period
L = int(Fs * Tsym)  # Up-sampling rate, L samples per symbol
f_c = 5000

# Generate random symbol indices (0 to 15 for 16-QAM)
symbol_indices = np.random.randint(16, size=num_symbols)

# Amplitude levels for I and Q channels
amplitudes_I = [-3, -1, 1, 3]
amplitudes_Q = [-3, -1, 1, 3]

# Initialize arrays for I and Q channels
pulse_train_I = np.array([])
pulse_train_Q = np.array([])

# Map each symbol index to amplitude levels for I and Q channels
for symbol_index in symbol_indices:
    symbol_I = amplitudes_I[symbol_index % 4]
    symbol_Q = amplitudes_Q[symbol_index // 4]

    # Create pulses for I and Q channels with oversampling
    pulse_I = np.zeros(L)
    pulse_Q = np.zeros(L)

    pulse_I[0] = symbol_I
    pulse_Q[0] = symbol_Q

    pulse_train_I = np.concatenate((pulse_train_I, pulse_I))
    pulse_train_Q = np.concatenate((pulse_train_Q, pulse_Q))

# Combining I and Q channels to get 16-QAM signal
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
beta = 0.30
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
plt.title("RRC Filtecolor_code[1] I")

plt.subplot(1, 2, 2)
plt.plot(samples_Q, '.-')
for i in range(num_symbols):
    plt.plot([i * L + num_taps // 2, i * L + num_taps // 2], [0, samples_Q[i * L + num_taps // 2]])
plt.grid(True)
plt.title("RRC Filtecolor_code[1] Q")
plt.show()

# create I + Q

# add carrier wave
t = np.linspace(0, len(samples_I) / Fs, len(samples_I))
carrier_I = np.cos(2 * np.pi * f_c * t)  # Carrier wave for I component
carrier_Q = np.sin(2 * np.pi * f_c * t)  # Carrier wave for Q component

qam_signal = samples_I * carrier_I + samples_Q * carrier_Q

# cap volume to 1
adj_samples = np.divide(qam_signal, np.max(qam_signal))
attenuation_factor = 1 / np.max(qam_signal)  # plus other factors ie. volume?

print(np.shape(adj_samples))
print("Attenuated signal by: ", attenuation_factor)
wavfile.write('QAM16_modulated_wave.wav', Fs, adj_samples)
print("Wrote .wav file!")

# FFT of the samples array
fft_result = np.fft.fft(qam_signal)
fft_result_shifted = np.fft.fftshift(fft_result)

# frequency axis
N = len(qam_signal)  # Number of samples
frequencies = np.fft.fftshift(np.fft.fftfreq(N, d=1 / Fs))

# Plot the magnitude spectrum
plt.plot(frequencies, np.abs(fft_result_shifted))
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

# => BEGIN CHANNEL SIMULATION
Fs, samples = wavfile.read("QAM16_modulated_wave.wav")
print(np.shape(samples))

# samples = np.divide(samples, 32767)
# print(f"largest sample amp: {np.max(samples)}")

add_noise = True
gain_control = True
remove_carrier = True

assert Fs == 48000

if remove_carrier:
    # Define local oscillator components (in-phase and quadrature)
    t = np.linspace(0, len(samples) / Fs, len(samples))
    local_oscillator_I = np.cos(2 * np.pi * f_c * t)
    local_oscillator_Q = np.sin(2 * np.pi * f_c * t)

    # Perform quadrature demodulation
    demod_I = samples * local_oscillator_I * 2
    demod_Q = samples * local_oscillator_Q * 2

    # Low-pass filtering
    cutoff_frequency = 4500  # Hz
    filter_length = 101  # Filter length, make it odd

    nyquist_frequency = 0.5 * Fs  # Nyquist frequency
    cutoff_normalized = cutoff_frequency / nyquist_frequency
    h_lp = signal.firwin(filter_length, cutoff_normalized, window='hamming')

    h_lp /= np.sum(h_lp)  # unity gain

    demod_I_filtered = np.convolve(demod_I, h_lp, 'full')  # apply filter
    demod_Q_filtered = np.convolve(demod_Q, h_lp, 'full')  # apply filter

    # Combine into complex numbers
    samples = demod_I_filtered + 1j * demod_Q_filtered

    # t = np.linspace(0, len(samples) / Fs, len(samples))
    # samples = np.multiply(samples, np.cos(2 * np.pi * f_c * t))
    #
    # cutoff_frequency = 4000  # Hz
    # filter_length = 101  # Filter length, make it odd
    #
    # nyquist_frequency = 0.5 * Fs  # Nyquist frequency
    # cutoff_normalized = cutoff_frequency / nyquist_frequency
    # h_lp = signal.firwin(filter_length, cutoff_normalized, window='hamming')
    #
    # h_lp /= np.sum(h_lp)  # unity gain
    #
    # samples = np.convolve(samples, h_lp, 'full')  # apply filter
    # # samples = np.convolve(samples, h_rcc)  # apply filter
    #
    # FFT of the samples array
    fft_result = np.fft.fft(samples)
    fft_result_shifted = np.fft.fftshift(fft_result)

    # frequency axis
    N = len(samples)  # Number of samples
    frequencies = np.fft.fftshift(np.fft.fftfreq(N, d=1 / Fs))

    # Plot the magnitude spectrum
    plt.plot(frequencies, np.abs(fft_result_shifted))
    plt.title('Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

if gain_control:
    samples = np.multiply(samples, 1 / attenuation_factor)

if add_noise:
    # Create and apply fractional delay filter and plot constellation maps
    delay = 0.4  # fractional delay, in samples
    N = 21  # number of taps
    n = np.arange(N)  # 0,1,2,3...
    h = np.sinc(n - (N - 1) / 2 - delay)  # calc filter taps
    h *= np.hamming(N)  # window the filter to make sure it decays to 0 on both sides
    h /= np.sum(h)  # normalize to get unity gain, we don't want to change the amplitude/power
    samples = np.convolve(samples, h)  # apply filter

    # Apply a freq offset
    fo = 50  # simulate freq offset
    Ts = 1 / Fs  # calc sample period
    t = np.arange(0, Ts * len(samples), Ts)  # create time vector

    t = t[0:len(samples)]
    samples = samples * np.exp(2 * np.pi * 1j * fo * t)  # freq shift

# Plot constellation
plt.plot(np.real(samples), np.imag(samples), '.')
plt.axis([-4, 4, -4, 4])
plt.title('Raw RX Constellation Map')
plt.xlabel("In-phase Component")
plt.ylabel("Quadrature Component")
plt.grid()
plt.show()

tools.eye_diagram.plot_eye(np.real(samples), np.imag(samples), L)

# ==> BEGIN DEMODULATION

# TIME SYNC - Mueller and muller clock recovery
interp_factor = 1024
samples_interpolated = signal.resample_poly(samples, interp_factor, 1)

mu = 0  # initial estimate of phase of sample
out = np.zeros(len(samples) + 10, dtype=np.complex64)
out_rail = np.zeros(len(samples) + 10, dtype=np.complex64)
i_in = 0  # input samples index
i_out = 2  # output index (let first two outputs be 0)
while i_out < len(samples) and i_in < len(samples):
    out[i_out] = samples_interpolated[
        i_in * interp_factor + int(mu * interp_factor)]  # grab what we think is the "best" sample
    out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j * int(np.imag(out[i_out]) > 0)
    x = (out_rail[i_out] - out_rail[i_out - 2]) * np.conj(out[i_out - 1])
    y = (out[i_out] - out[i_out - 2]) * np.conj(out_rail[i_out - 1])
    mm_val = np.real(y - x)
    mm_val = min(mm_val, 4.0)  # For the sake of there being less code to explain
    mm_val = max(mm_val, -4.0)
    mu += L + 0.1 * mm_val
    i_in += int(np.floor(mu))  # round down to nearest int since we are using it as an index
    mu = mu - np.floor(mu)  # remove the integer part of mu
    i_out += 1  # increment output index
out = out[2:i_out]  # remove the first two, and anything after i_out (that was never filled out)

_, (ax1, ax2) = plt.subplots(2, figsize=(8, 3.5))  # 7 is nearly full width
ax1.plot(np.real(sig), '.-')
ax1.plot(np.imag(sig))
ax2.plot(np.real(out[6:-7]), '.-')
ax2.plot(np.imag(out[6:-7]), '.-')
plt.show()

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))  # 7 is nearly full width
ax1.plot(np.real(samples), np.imag(samples), '.')
plt.axis([-4, 4, -4, 4])
ax1.set_title('Before Time Sync')
plt.xlabel("In-phase Component")
plt.ylabel("Quadrature Component")
ax1.grid()

ax2.plot(np.real(out), np.imag(out), '.')
plt.axis([-4, 4, -4, 4])
ax2.set_title('After Time Sync')
ax2.grid()
plt.xlabel("In-phase Component")
plt.ylabel("Quadrature Component")
plt.show()

tools.eye_diagram.plot_eye(np.real(out), np.imag(out), L)

# => SYMBOL TIMING RECOVERY USING PLL

samples = out  # copy for plotting

# PLL from lib
out, a_hat, e_phi, theta_hat = sync.DD_carrier_sync(samples, 16, 0.05, 0.707, mod_type='MQAM', type=2)

fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 5))  # 7 is nearly full width
fig.tight_layout(pad=2.0)  # add space between subplots
ax1.plot(np.real(samples), '.-')
ax1.plot(np.imag(samples), '.-')
ax1.set_title('Before Costas Loop')

ax2.plot(np.real(out), '.-')
ax2.plot(np.imag(out), '.-')
ax2.set_title('After Costas Loop')
plt.show()

fig, ax = plt.subplots(figsize=(7, 3))  # 7 is nearly full width
# For some reason you have to divide the steady state freq by 50,
#   to get the fraction of fs that the fo is...
#   and changing loop_bw doesn't matter
ax.plot(theta_hat, '.-')
ax.set_xlabel('Sample')
ax.set_ylabel('Freq Offset')
plt.show()

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))  # 7 is nearly full width
ax1.plot(np.real(samples), np.imag(samples), '.')
plt.axis([-4, 4, -4, 4])
ax1.set_title('Before Carrier Sync')
plt.xlabel("In-phase Component")
plt.ylabel("Quadrature Component")
ax1.grid()

ax2.plot(np.real(out), np.imag(out), '.')
plt.axis([-4, 4, -4, 4])
ax2.set_title('After Carrier Sync')
ax2.grid()
plt.xlabel("In-phase Component")
plt.ylabel("Quadrature Component")
plt.show()
