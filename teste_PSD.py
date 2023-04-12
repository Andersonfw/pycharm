#  Copyright (c) 2023. Ânderson Felipe Weschenfelder

"""
Created on abril 01 23:03:02 2023

@author: Ânderson Felipe Weschenfelder
"""

import numpy as np

from scipy import signal

import matplotlib.pyplot as plt

rng = np.random.default_rng()

fs = 10e3

N = 1e5

amp = 2*np.sqrt(2)

freq = 1234.0

noise_power = 0.001 * fs / 2

time = np.arange(N) / fs

x = amp*np.sin(2*np.pi*freq*time)

x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)

f, Pxx_den = signal.welch(x, fs, nperseg=1024)

plt.semilogy(f, Pxx_den)

plt.ylim([0.5e-3, 1])

plt.xlabel('frequency [Hz]')

plt.ylabel('PSD [V**2/Hz]')

# plt.show()

plt.figure()
f, psd = signal.periodogram(x, fs=fs)
plt.semilogy(f, psd)
plt.xlabel('Freq (Hz)')
plt.ylabel('PSD (dB/Hz)')
plt.legend(['PSD with LIB'])
plt.ylim([0.5e-3, 1])
plt.show()