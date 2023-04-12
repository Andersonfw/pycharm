#  Copyright (c) 2023. Ânderson Felipe Weschenfelder

"""
Created on abril 10 22:58:48 2023

@author: Ânderson Felipe Weschenfelder
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp

Fs = 9.9e3;  # taxa de amostragem
Ts = 1.0 / Fs;  # periodo de amostragem
t = np.arange(0, 1, Ts)  # vetor de tempo

# x = chirp(t, f0=20, f1=20000, t1=1, method='linear')
f1 = 2.5e3
f2 = 1e3
x = np.cos(2 * np.pi * f1 * t) +  np.cos(2 * np.pi * f2 * t)

fc = 2000;  # frequencia de corte
zeta = 0.707;
C = 1 / (np.tan(np.pi * fc / Fs));

b0 = 1 / (1 + 2 * zeta * C + C * C);
b1 = 2 * b0;
b2 = b0;

a0 = 1;
a1 = 2 * b0 * (1 - C * C);
a2 = b0 * (1 - 2 * zeta * C + C * C);

# Variaveis de estado
xh1 = 0;
xh2 = 0;

yh1 = 0;
yh2 = 0;

y = np.zeros(len(x));

for n in range(0, len(x)):
    y[n] = b0 * x[n] + b1 * xh1 + b2 * xh2 - a1 * yh1 - a2 * yh2;

    yh2 = yh1;
    yh1 = y[n];

    xh2 = xh1;
    xh1 = x[n];

n = len(y)  # tamanho do sinal
k = np.arange(n)  # vetor em k
T = n / Fs
frq = k / T  # os dois lados do vetor de frequencia
frq = frq[range(int(n / 2))]  # apenas um lado

Y = np.fft.fft(y) / n  # calculo da fft e normalização por n
Y = Y[range(int(n / 2))]

X = np.fft.fft(x) / n  # calculo da fft e normalização por n
X = X[range(int(n / 2))]

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.title('Sweep')
plt.xlabel('Tempo')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(frq, abs(X), 'r')
plt.title('Fourier - Sweep')
plt.xlabel('Freq (Hz)')
plt.ylabel('|X(freq)|')

plt.subplot(3, 1, 3)
plt.plot(frq, abs(Y), 'r')
plt.grid(True, which="both")
plt.title('Fourier - Filtro')
plt.xlabel('Freq (Hz)')
plt.ylabel('|X(freq)|')
plt.show()