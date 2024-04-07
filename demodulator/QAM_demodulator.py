import numpy as np
import matplotlib.pyplot as plt
import sk_dsp_comm.digitalcom as dc
import sk_dsp_comm.sigsys as ss
import sk_dsp_comm.synchronization as sync

from scipy.io import wavfile
from tools.constants import *

M = 16

Fs_prime, samples = wavfile.read("QAM16_modulated_wave.wav")
assert Fs_prime == Fs

