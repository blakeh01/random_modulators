"""
    General implementation of a QAM modulation/demodulator *16 QAM for now
"""

import numpy as np
import matplotlib.pyplot as plt
import sk_dsp_comm.sigsys as sigsys
from scipy import signal
from scipy.io import wavfile

from tools.helpers import plot_mag_fft


class QAMModem:

    def __init__(self, symbol_rate, sps=16, fs=48000):
        self.fs = fs
        self.symbol_rate = symbol_rate
        self.sps = sps
        pass

    def modulate_bits(self, bits, mod=16, carrier=10000, alpha=0.3, write_to_wav=True, play_sig=False,
                      plots=True):

        bin2gray = [0, 1, 3, 2]

        word_len = int(np.log2(mod) / 2)
        w = 2 ** np.arange(word_len - 1, -1, -1)
        x_m = np.sqrt(mod) - 1

        # truncate bits to even powers of the modulation order
        num_symbols = int(np.floor(len(bits) / np.log2(mod)))
        x_IQ = np.zeros(num_symbols)

        # apply rrc filter
        b = sigsys.sqrt_rc_imp(self.sps, alpha)
        xbb = signal.lfilter(b, 1, sigsys.upsample(x_IQ, self.sps))  # baseband signal
        if plots:
            plt.stem(b)
            plt.title("RRC Impulse Response")
            plt.show()
            plot_mag_fft(xbb, self.fs, 'Filtered Symbols')

        # apply carrier using DTFT shift property
        n = np.arange(len(xbb))
        xc = xbb * np.exp(1j * 2 * np.pi * (carrier / self.fs) * n)  # shift in freq domain to carrier
        if plots:
            plot_mag_fft(xc, self.fs, 'Carrier Modulated Symbols')

        # write out to .wav file
        if write_to_wav:
            xc = xc / np.max(np.abs(xc))
            wavfile.write('xc1_complex.wav', self.fs, np.array(xc, dtype=np.float32))

        return bits, xc  # return raw data bits for BER

    def generate_sync(self, sync_len):
        return [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    def generate_preamble(self):
        return [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    def demodulate_signal(self, sig, carrier=10000, plots=True):
        pass

    def detect_preamble(self):
        pass

    def decode_symbols(self):
        pass


n_symbols = 128
modem = QAMModem(2400)

bits, _ = modem.modulate_bits(np.random.randint(0, 2, size=4 * n_symbols))


