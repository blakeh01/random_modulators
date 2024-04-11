import numpy as np
import sk_dsp_comm.digitalcom as dc
import sk_dsp_comm.sigsys as ss
import sk_dsp_comm.synchronization as sync
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from tools.helpers import plot_mag_fft

# ===> TEST MODULATOR <===

Fs = 48000  # samp freq
symbol_rate = 1200  # symbols per second (2400*4 = 9600 bps)
Ns = int(Fs / symbol_rate)  # samples per bit
M = 16  # constellation size
N_sym = 20000  # num of symbols
f_c = 2500  # carrier freq
alpha = 0.7  # roll-off factor, increases bandwidth

xbb, h, data = dc.qam_gray_encode_bb(N_sym, Ns, M, pulse='src', alpha=alpha)
n = np.arange(0, len(xbb))

# add carrier
xc1 = xbb * np.exp(1j * 2 * np.pi * (f_c / Fs) * n)

plot_mag_fft(xc1, Fs)

# Normalize the signal to prevent clipping
xc1 = xc1 / np.max(np.abs(xc1))

# Save xc1 to a .wav file
wavfile.write('xc1_complex.wav', Fs, np.array(xc1, dtype=np.float32))

# ===> TEST CHANNEL <===

# Read the saved .wav file
Fs, rcc = wavfile.read('xc1_complex_dsp_5kc_1200sps.wav')
n = np.arange(0, len(rcc))

# # AWGN
# EbN0_dB = 20
# EsN0_dB = 10 * np.log10(np.log2(M)) + EbN0_dB
# rcc = dc.cpx_awgn(rcc, EsN0_dB, Ns)
#
# # Freq error
# F_error = 10  # hz
# rcc *= np.exp(1j * 2 * np.pi * (F_error / Fs) * n)  # Df = 0.012*Rb or 1.2% of Rb

# ===> TEST DEMODULATOR <===
rbb = rcc * np.exp(-1j * 2 * np.pi * (f_c / Fs) * n)  # carrier removal
rbb = signal.lfilter(h, 1, rbb)  # matched filter

dc.eye_plot(rbb, 2 * Ns, 2 * M * Ns)  # eye diagram after filter
plt.show()

# down-sample to current sampling instant (need to time)
rbb = ss.downsample(rbb, Ns)  # plot IQ at sample instant

# PLL
zz, a_hat, e_phi, theta_hat = sync.DD_carrier_sync(rbb, M, 0.05, 0.707, mod_type='MQAM', type=2)

# dc.eye_plot(zz, 2 * Ns, 2 * M * Ns)  # eye diagram after filter
plt.show()

plt.plot(e_phi[:1000])
plt.ylabel(r'Phase Error (rad)')
plt.xlabel(r'Bits')
plt.title(r'Carrier Phase Tracking')
plt.grid()
plt.show()

Nstart = 30
Nbits = M * N_sym
plt.subplot(1, 2, 1)
plt.plot(rbb[Nstart:Nbits].real, rbb[Nstart:Nbits].imag, 'g.')
plt.axis('equal')
plt.xlabel(r'In-phase (real part)')
plt.ylabel(r'Quadrature (imag part)')
plt.title(r'Before Phase Track')
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(zz[Nstart:Nstart + Nbits].real, zz[Nstart:Nstart + Nbits].imag, 'g.')
plt.axis('equal')
plt.xlabel(r'In-phase (real part)')
plt.ylabel(r'Quadrature (imag part)')
plt.title(r'After Phase Track')
plt.grid()
plt.tight_layout()
plt.show()

# 'estimated' BEP, no full support for QAM
data_hat = dc.qam_gray_decode(rbb, M)
Nbits, Nerrors = dc.bpsk_bep(data, data_hat)
print('BEP: Nbits = %d, Nerror = %d, Pe_est = %1.3e' % \
      (Nbits, Nerrors, Nerrors / Nbits))
