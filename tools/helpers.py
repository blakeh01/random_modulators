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


def plot_constellation(real, imag, added_title=""):
    plt.plot(real, imag, '.')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.title(added_title)
    plt.xlabel("In-phase Component")
    plt.ylabel("Quadrature Component")
    plt.grid(True)
    plt.show()


def plot_constellation_2(real1, imag1, real2, imag2, titles):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))  # 7 is nearly full width
    ax1.plot(real1, imag1, '.')
    ax1.set_title(titles[0])
    # ax1.set_xlim([-4, 4])
    # ax1.set_ylim([-4, 4])
    plt.xlabel("In-phase Component")
    plt.ylabel("Quadrature Component")
    ax1.grid()

    ax2.plot(real2, imag2, '.')
    ax2.set_title(titles[1])
    ax2.grid()
    plt.xlabel("In-phase Component")
    plt.ylabel("Quadrature Component")
    # ax2.set_xlim([-4, 4])
    # ax2.set_ylim([-4, 4])
    plt.show()

def preamble_gen(preamble):
    shift_preamble = [[], [], [], []]
    shift_preamble[0] = preamble
    preamble = np.reshape(preamble, (int(len(preamble) / 4), 4))
    for i in preamble:
        if np.array_equal(i, [0, 0, 0, 0]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [0, 0, 1, 0]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [1, 0, 1, 0]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [1, 0, 0, 0]))
        elif np.array_equal(i, [0, 0, 0, 1]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [0, 1, 1, 0]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [1, 0, 1, 1]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [1, 1, 0, 0]))
        elif np.array_equal(i, [0, 0, 1, 0]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [1, 0, 1, 0]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [1, 0, 0, 0]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [0, 0, 0, 0]))
        elif np.array_equal(i, [0, 0, 1, 1]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [1, 1, 1, 0]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [1, 0, 0, 1]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [0, 1, 0, 0]))
        elif np.array_equal(i, [0, 1, 0, 0]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [0, 0, 1, 1]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [1, 1, 1, 0]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [1, 0, 0, 1]))
        elif np.array_equal(i, [0, 1, 0, 1]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [0, 1, 1, 1]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [1, 1, 1, 1]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [1, 1, 0, 1]))
        elif np.array_equal(i, [0, 1, 1, 0]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [1, 0, 1, 1]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [1, 1, 0, 0]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [0, 0, 0, 1]))
        elif np.array_equal(i, [0, 1, 1, 1]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [1, 1, 1, 1]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [1, 1, 0, 1]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [0, 1, 0, 1]))
        elif np.array_equal(i, [1, 0, 0, 0]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [0, 0, 0, 0]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [0, 0, 1, 0]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [1, 0, 1, 0]))
        elif np.array_equal(i, [1, 0, 0, 1]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [0, 1, 0, 1]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [0, 0, 1, 1]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [1, 1, 1, 0]))
        elif np.array_equal(i, [1, 0, 1, 0]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [0, 1, 0, 0]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [0, 0, 1, 1]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [1, 1, 1, 0]))
        elif np.array_equal(i, [1, 0, 1, 1]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [1, 1, 0, 0]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [0, 0, 0, 1]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [0, 1, 1, 0]))
        elif np.array_equal(i, [1, 1, 0, 0]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [0, 0, 0, 1]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [0, 1, 1, 0]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [1, 0, 1, 1]))
        elif np.array_equal(i, [1, 1, 0, 1]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [0, 1, 0, 1]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [0, 1, 1, 1]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [1, 1, 1, 1]))
        elif np.array_equal(i, [1, 1, 1, 0]):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [1, 0, 0, 1]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [0, 1, 0, 0]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [0, 0, 1, 1]))
        elif (np.array_equal(i, [1, 1, 1, 1])):
            shift_preamble[1] = np.concatenate((shift_preamble[1], [1, 1, 0, 1]))
            shift_preamble[2] = np.concatenate((shift_preamble[2], [0, 1, 0, 1]))
            shift_preamble[3] = np.concatenate((shift_preamble[3], [0, 1, 1, 1]))
        for j in range(4):
            shift_preamble[j] = list([int(x) for x in shift_preamble[j]])
    return shift_preamble
