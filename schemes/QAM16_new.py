"""
    BPSK example of pulse-trains being shaped using RRC
"""
import random

import numpy as np
import matplotlib.pyplot as plt

import tools.eye_diagram
import sk_dsp_comm.synchronization as sync
import sk_dsp_comm.sigsys as sigsys

from scipy import signal
from scipy.io import wavfile

from tools.helpers import plot_mag_fft

# => DATA GENERATION - PULSE TRAINS AND PULSE SHAPING

Fs = 48000  # Sampling rate
num_symbols = 2400  # Number of symbols
symbol_rate = 2400  # symbols per second (2400*4 = 9600 bps)
Ns = int(Fs / symbol_rate)  # samples per bit
f_c = 10000

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
    pulse_I = np.zeros(Ns)
    pulse_Q = np.zeros(Ns)

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

# Create our raised-cosine filter and filter
alpha = 0.7
b = sigsys.sqrt_rc_imp(Ns, alpha)
xbb = signal.lfilter(b, 1, sig)
b = b/sum(b)  # make unity gain for RX

plt.figure(1)
plt.plot(b, '.')
plt.title("RRC Filter Response")
plt.grid(True)
plt.show()

# add carrier
n = np.arange(0, len(xbb))
xc1 = xbb * np.exp(1j * 2 * np.pi * (f_c / Fs) * n)

# normalize
attenuation_factor = 1 / np.max(np.abs(xc1))
xc1 = xc1 / np.max(np.abs(xc1))
wavfile.write('QAM16_modulated_wave.wav', Fs, np.array(xc1, dtype=np.float32))
plot_mag_fft(xc1, Fs)

print("Wrote .wav file! Normalized: " + str(attenuation_factor))

# => BEGIN CHANNEL SIMULATION
Fs, rcc = wavfile.read("QAM16_modulated_wave.wav")
assert Fs == 48000

add_noise = True
gain_ctrl = True

# remove carrier
n = np.arange(0, len(rcc))
rbb = rcc * np.exp(-1j * 2 * np.pi * (f_c / Fs) * n)  # carrier removal
rbb = signal.lfilter(b, 1, rbb)  # matched filter

if gain_ctrl:
    rbb = rbb * (1/attenuation_factor)

if add_noise:
    # # Create and apply fractional delay filter and plot constellation maps
    delay = 0.4  # fractional delay, in samples
    N = 21  # number of taps
    n = np.arange(N)  # 0,1,2,3...
    h = np.sinc(n - (N - 1) / 2 - delay)  # calc filter taps
    h *= np.hamming(N)  # window the filter to make sure it decays to 0 on both sides
    h /= np.sum(h)  # normalize to get unity gain, we don't want to change the amplitude/power
    rbb = np.convolve(rbb, h)  # apply filter

    # Apply a freq offset
    fo = 1  # simulate freq offset
    Ts = 1 / Fs  # calc sample period
    t = np.arange(0, Ts * len(rbb), Ts)  # create time vector

    t = t[0:len(rbb)]
    rbb = rbb * np.exp(2 * np.pi * 1j * fo * t)  # freq shift

# Plot constellation
plt.plot(np.real(rbb), np.imag(rbb), '.')
plt.axis([-4, 4, -4, 4])
plt.title('Raw RX Constellation Map')
plt.xlabel("In-phase Component")
plt.ylabel("Quadrature Component")
plt.grid()
plt.show()

tools.eye_diagram.plot_eye(np.real(rbb), np.imag(rbb), Ns)

# ==> BEGIN DEMODULATION

# TED
out, _ = sync.NDA_symb_sync(rbb, Ns, 4,  0.05)
print(len(out))

_, (ax1, ax2) = plt.subplots(2, figsize=(8, 3.5))  # 7 is nearly full width
ax1.plot(np.real(sig), '.-')
ax1.plot(np.imag(sig))
ax2.plot(np.real(out[6:-7]), '.-')
ax2.plot(np.imag(out[6:-7]), '.-')
plt.show()

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))  # 7 is nearly full width
ax1.plot(np.real(rbb), np.imag(rbb), '.')
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

tools.eye_diagram.plot_eye(np.real(out), np.imag(out), Ns)

# => SYMBOL TIMING RECOVERY USING PLL
rbb = out

# PLL
out, a_hat, e_phi, theta_hat = sync.DD_carrier_sync(rbb, 16, 0.05, 0.707, mod_type='MQAM', type=2)

fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 5))  # 7 is nearly full width
fig.tight_layout(pad=2.0)  # add space between subplots
ax1.plot(np.real(rbb), '.-')
ax1.plot(np.imag(rbb), '.-')
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
ax1.plot(np.real(rbb), np.imag(rbb), '.')
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
