"""
    General implementation of a QAM modulation/demodulator *16 QAM for now
"""
import math

import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import sk_dsp_comm.sigsys as sigsys
import sk_dsp_comm.synchronization as sync
from crcmod import crcmod

from scipy import signal
from scipy.io import wavfile
from tools.helpers import plot_mag_fft, plot_constellation_2


def create_packet(data, num_reps):
    header = len(data).to_bytes(1, byteorder='big')

    # repeat data as basic FEC
    encoded_data = ""
    for bit in data:
        encoded_data += bit * num_reps

    # create CRC to check for errors at RX (if this passes, dont do any FEC!)
    crc16 = crcmod.predefined.mkPredefinedCrcFun('crc-16')
    crc = crc16(header + encoded_data).to_bytes(2, byteorder='big')

    # Assemble packet
    packet = header + encoded_data + crc

    return packet


class QAMModem:

    def __init__(self, symbol_rate, M=16, alpha=0.9, f_carrier=1500, fs=48000):
        """
        Generates an M-QAM modem (only 16-QAM supported atm) that is capable of modulating and demodulating at audio
        frequencies.

        :param symbol_rate: symbols/sec
        :param M: modulation order (M=2,4,16,64,128,256 support by libs, however, 16 is only working in this impl.)
        :param alpha: roll-off factor of matched filter, higher means lower ISI however increased bandwidth
        :param f_carrier: carrier frequency
        :param fs: sampling rate
        """
        self.fs = fs
        self.symbol_rate = symbol_rate
        self.M = M
        self.sps = int(self.fs / self.symbol_rate)
        self.alpha = alpha
        self.f_carrier = f_carrier

        self.matched_filter = sigsys.sqrt_rc_imp(self.sps, alpha)
        self.matched_filter_RX = self.matched_filter / sum(self.matched_filter)  # normalize gain for RX

        self.sync_bits = self.generate_sync()
        self.preamble_bits = self.generate_preamble()

    def packet_and_modulate_bits(self, data_bits, write_to_wav=True, play_sig=False, plots=True):
        """
        Given a list of bits to encode, generates a string of bits (sync, preamble, packet data) then performs a pulse
        shaping filter. This is then modulated with 'f_carrier' to send over audio channel.

        :param data_bits: raw bits to packet (pure ascii bits or similar)
        :param write_to_wav: if true, will create a .wav file
        :param play_sig: if true, play the signal on the primary channel
        :param plots: if true, plot matched filter response and FFTs
        :return:
        """

        # # # Initialize arrays for I and Q channels
        pulse_train_I = np.array([])
        pulse_train_Q = np.array([])

        bits = np.concatenate([self.sync_bits, self.preamble_bits,
                               data_bits, np.zeros(int(math.log2(self.M)) * 100).astype(int)])
        constellation = self.constellation_map()
        group_size = int(math.log2(self.M))
        grouped_bits = [''.join(str(bit) for bit in bits[i:i + group_size]) for i in range(0, len(bits), group_size)]

        print(grouped_bits)

        # Map each symbol index to amplitude levels for I and Q channels
        for sym in grouped_bits:
            symbol_I = constellation[sym][0]
            symbol_Q = constellation[sym][1]

            pulse_train_I = np.concatenate((pulse_train_I, [symbol_I]))
            pulse_train_Q = np.concatenate((pulse_train_Q, [symbol_Q]))

        # Combining I and Q channels to get 16-QAM signal
        x_IQ = pulse_train_I + 1j * pulse_train_Q
        x_IQ = sigsys.upsample(x_IQ, self.sps)

        # apply rrc filter
        xbb = signal.lfilter(self.matched_filter, 1, x_IQ)  # baseband signal
        if plots:
            plt.stem(self.matched_filter)
            plt.title("RRC Impulse Response")
            plt.show()
            plot_mag_fft(xbb, self.fs, 'Filtered Symbols')

        # apply carrier using DTFT shift property
        n = np.arange(len(xbb))
        xc = xbb * np.exp(1j * 2 * np.pi * (self.f_carrier / self.fs) * n)  # shift in freq domain to carrier
        if plots:
            plot_mag_fft(xc, self.fs, 'Carrier Modulated Symbols')

        # write out to .wav file
        if write_to_wav:
            print("Attenuation: ", 1 / np.max(np.abs(xc)))
            xc = xc / np.max(np.abs(xc))
            wavfile.write('MQAM_modulated_wave.wav', self.fs, np.array(xc, dtype=np.float32))

        # play .wav file here
        if play_sig:
            p = pyaudio.PyAudio()

            stream = p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=self.fs,
                            output=True)

            stream.write(np.array(xc, dtype=np.float32).tobytes())
            stream.stop_stream()
            stream.close()
            p.terminate()

        return xc, bits  # return raw data bits for BER

    def generate_sync(self, sync_len=100):
        """
        Generates sync bits to align PLL and TED before preamble & data.
        :param sync_len: length of sync bits, might have to adjust depending on alpha and symbol rate
        :return: bit string of random words
        """
        return np.random.randint(0, 2, size=int(math.log2(self.M)) * sync_len)

    def generate_preamble(self):
        """
        todo: create some kind of algorithm to generate based on modulation order
        :return: random preamble ( 8 symbols for M=16 )
        """
        return np.array(
            [1, 1, 1, 1,
             0, 0, 0, 0,
             1, 1, 1, 1,
             0, 0, 0, 0,
             1, 1, 0, 0,
             0, 0, 1, 1,
             1, 1, 1, 1,
             0, 0, 0, 0])

    def demodulate_signal(self, sig, plots=True):
        """
        Takes an M-QAM modulated signal and recovers the symbol through a matched filter w/ a TED algorithm and
        PLL with M-QAM error function.

        :param sig: raw signal from RX
        :param plots: show constellation maps and FFTs
        :return: all symbols received from RX including sync words, preamble, and data.
        """
        if plots: plot_mag_fft(sig, self.fs, "Received Signal")

        # equalize
        sig = sig * 4  # todo calc channel attenuation

        # remove carrier (shift back to 0)
        n = np.arange(0, len(sig))
        rbb = sig * np.exp(-1j * 2 * np.pi * (self.f_carrier / self.fs) * n)

        # apply matched filter (acts as LP to remove carrier)
        rbb = signal.lfilter(self.matched_filter_RX, 1, rbb)

        if plots: plot_mag_fft(rbb, self.fs, "DTFT Shifted and Filtered Signal")

        # TED
        out, e = sync.NDA_symb_sync(rbb, self.sps, 2, 0.05)
        rbb = sigsys.downsample(rbb, self.sps)  # downsample at L to get sampling instant before time sync.
        if plots: plot_constellation_2(rbb.real, rbb.imag, out.real, out.imag, ["Before Time Sync", "After Time Sync"])

        # phase locked loop
        out_pll, a_hat, e_phi, theta_hat = sync.DD_carrier_sync(out, self.M, 0.1, 0.707, mod_type='MQAM', type=1)
        if plots: plot_constellation_2(out.real, out.imag, out_pll.real, out_pll.imag, ["Before PLL", "After PLL"])

        if plots:
            plt.plot(e_phi)
            plt.title("Carrier Timing Error")
            plt.ylabel("Error")
            plt.xlabel("Sample [n]")
            plt.show()

        # map constellation points to ideal ones
        ideal_x = np.array([point[0] for point in self.constellation_map().values()])
        ideal_y = np.array([point[1] for point in self.constellation_map().values()])

        ideal_x = ideal_x / np.max(ideal_x)
        ideal_y = ideal_y / np.max(ideal_y)  # normalize to match RX conditions

        # map constellation points to ideal ones
        symbols = []
        idx = []  # Array to store the index of the closest ideal constellation point for each RX point

        for point in out_pll:
            x = point.real
            y = point.imag

            distances = np.sqrt((x - ideal_x) ** 2 + (y - ideal_y) ** 2)
            closest_index = np.argmin(distances)
            idx.append(closest_index)  # Append the index of the closest point
            closest_binary = list(self.constellation_map().keys())[closest_index]
            symbols.append(closest_binary)

        closest_points = np.array(idx)

        # plot ideal vs. actual constellation
        if plots:
            colors = ['orange', 'green', 'cyan', 'magenta', 'yellow', 'brown', 'purple', 'pink',
                      'lime', 'teal', 'indigo', 'maroon', 'olive', 'navy', 'turquoise', 'salmon']

            plt.plot(ideal_x, ideal_y, 'r.', label="Ideal Points")
            for i, (symbol, (x, y)) in enumerate(self.constellation_map().items()):
                plt.text(x / 3, (y / 3) + 0.1, f'{symbol}', fontsize=9, color='red', ha='center', va='center')
                plt.scatter(out_pll.real[closest_points == i], out_pll.imag[closest_points == i], color=colors[i],
                            label=symbol)

            plt.title("RX Constellation Map")
            plt.xlabel("In-phase Component")
            plt.ylabel("Quadrature Component")
            plt.legend()
            plt.grid(True)
            plt.show()

        # rotate symbols to fix phase ambiguity
        desired_symbols = ['0000', '1000', '1010', '0010']  # possible configurations
        symbols_filtered = [symbol for symbol in symbols if symbol in desired_symbols]
        symbols_unique, symbols_counts = np.unique(symbols_filtered, return_counts=True)

        if plots:
            plt.figure(figsize=(8, 6))
            plt.bar(symbols_unique, symbols_counts, color=colors[:len(symbols_unique)])
            plt.title("Symbol Occurrence")
            plt.xlabel("Symbols")
            plt.ylabel("Occurrence Count")
            plt.grid(True)
            plt.show()

        most_common_symbol = symbols_unique[np.argmax(symbols_counts)]
        print("Possible phase detection symbol:", most_common_symbol)

        if most_common_symbol == '1000':
            print("Phase ambiguity detected by pi/2")
            out_pll = out_pll * np.exp(1j * (np.pi / 2))
        elif most_common_symbol == '1010':
            print("Phase ambiguity detected by pi")
            out_pll = out_pll * np.exp(1j * np.pi)
        elif most_common_symbol == '0010':
            print("Phase ambiguity detected by -pi/2")
            out_pll = out_pll * np.exp(-1j * (np.pi / 2))
        else:
            print("No phase ambiguity detected.")
            return symbols

        # RECALC SYMBOLS WITH ROTATION
        # map constellation points to ideal ones
        ideal_x = np.array([point[0] for point in self.constellation_map().values()])
        ideal_y = np.array([point[1] for point in self.constellation_map().values()])

        ideal_x = ideal_x / np.max(ideal_x)
        ideal_y = ideal_y / np.max(ideal_y)  # normalize to match RX conditions

        # map constellation points to ideal ones
        symbols = []
        idx = []  # Array to store the index of the closest ideal constellation point for each RX point

        for point in out_pll:
            x = point.real
            y = point.imag

            distances = np.sqrt((x - ideal_x) ** 2 + (y - ideal_y) ** 2)
            closest_index = np.argmin(distances)
            idx.append(closest_index)  # Append the index of the closest point
            closest_binary = list(self.constellation_map().keys())[closest_index]
            symbols.append(closest_binary)

        closest_points = np.array(idx)

        # plot ideal vs. actual constellation
        if plots:
            colors = ['orange', 'green', 'cyan', 'magenta', 'yellow', 'brown', 'purple', 'pink',
                      'lime', 'teal', 'indigo', 'maroon', 'olive', 'navy', 'turquoise', 'salmon']

            plt.plot(ideal_x, ideal_y, 'r.', label="Ideal Points")
            for i, (symbol, (x, y)) in enumerate(self.constellation_map().items()):
                plt.text(x / 3, (y / 3) + 0.1, f'{symbol}', fontsize=9, color='red', ha='center', va='center')
                plt.scatter(out_pll.real[closest_points == i], out_pll.imag[closest_points == i], color=colors[i],
                            label=symbol)

            plt.title("RX Constellation Map")
            plt.xlabel("In-phase Component")
            plt.ylabel("Quadrature Component")
            plt.legend()
            plt.grid(True)
            plt.show()

        # rotate symbols to fix phase ambiguity
        desired_symbols = ['0000', '1000', '1010', '0010']  # possible configurations
        symbols_filtered = [symbol for symbol in symbols if symbol in desired_symbols]

        if plots:
            plt.figure(figsize=(8, 6))
            symbols_unique, symbols_counts = np.unique(symbols_filtered, return_counts=True)
            plt.bar(symbols_unique, symbols_counts, color=colors[:len(symbols_unique)])
            plt.title("Symbol Occurrence")
            plt.xlabel("Symbols")
            plt.ylabel("Occurrence Count")
            plt.grid(True)
            plt.show()

        return symbols

    def get_data_from_stream(self, symbols):
        """
        Detects preamble in symbol stream and returns the data within the message.

        :param symbols: symbols from demodulation step. Should include all sync words, preamble, and data
        :return: All symbols sent with the preamble if detected, otherwise 'None'.
        """
        bits = [int(bit) for group in symbols for bit in group]
        correlation = np.correlate(bits, self.preamble_bits, mode='valid')
        max_correlation_index = np.argmax(np.abs(correlation))
        extracted_preamble_and_symbols = bits[max_correlation_index: max_correlation_index + len(self.preamble_bits)]

        if np.array_equal(extracted_preamble_and_symbols, self.preamble_bits):
            print("Preamble detected, extracting data...")
            return bits[max_correlation_index + len(self.preamble_bits):]  # return only the data

        print("Preamble not detected.")
        return None

    def get_text_from_data(self):
        pass

    def constellation_map(self):
        """
        :return: Gray coded constellation map depending on modulation order
        """
        if self.M == 16:
            return \
                {"0000": [-3, 3],
                 "0001": [-3, 1],
                 "0011": [-3, -1],
                 "0010": [-3, -3],
                 "0100": [-1, 3],
                 "0101": [-1, 1],
                 "0111": [-1, -1],
                 "0110": [-1, -3],
                 "1100": [1, 3],
                 "1101": [1, 1],
                 "1111": [1, -1],
                 "1110": [1, -3],
                 "1000": [3, 3],
                 "1001": [3, 1],
                 "1011": [3, -1],
                 "1010": [3, -3]}

### TEST CODE ###

# np.set_printoptions(threshold=sys.maxsize)
#
# # ==> MODULATE <===
# # n_symbols = 2400
# modem = QAMModem(800)
# test_bits = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1,
#                       0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0,
#                       1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0,
#                       0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
#
# # test, TX_bits = modem.packet_and_modulate_bits(test_bits, plots=False)
#
# # ==> DEMODULATE <===
# Fs, rcc = wavfile.read("../gui/ReceivedAudio.wav")
# add_noise = False
# assert Fs == 48000
#
# plt.plot(rcc)
# plt.show()
#
# if add_noise:
#     # Create and apply fractional delay filter and plot constellation maps
#     delay = 60  # fractional delay, in samples
#     N = 21  # number of taps
#     n = np.arange(N)  # 0,1,2,3...
#     h = np.sinc(n - (N - 1) / 2 - delay)  # calc filter taps
#     h *= np.hamming(N)  # window the filter to make sure it decays to 0 on both sides
#     h /= np.sum(h)  # normalize to get unity gain, we don't want to change the amplitude/power
#     rcc = np.convolve(rcc, h)  # apply filter
#
#     # # Apply a freq offset
#     fo = 1  # simulate freq offset
#     Ts = 1 / Fs  # calc sample period
#     t = np.arange(0, Ts * len(rcc), Ts)  # create time vector
#
#     rcc = rcc * np.exp(2 * np.pi * 1j * fo * t[0:len(rcc)])  # freq shift
#
# RX_symbols = modem.demodulate_signal(rcc)
# print(RX_symbols)
# data = modem.get_data_from_stream(RX_symbols)
#
# print(data)
