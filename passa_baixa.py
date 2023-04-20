"""
Created on abril 18 19:45:28 2023

@author: Ânderson Felipe Weschenfelder
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp


fs = 100e3 # frequência de amostragem
fc = 10e3   # frequência de corte
ts = 1/fs
tc = 1/fc
t = np.arange(0, 5000 * tc, ts) # vetor de tempo
V = 1   # amplitude do sinal
fb = 2e3    # largura de banda para filtros passa faixa e rejeita paixa


x1 = V * np.sin(2 * np.pi * fc * t)
x2 = V * np.sin(2 * np.pi * 1e3 * t)
x3 = V * np.sin(2 * np.pi * 12e3 * t)
x4 = V * np.sin(2 * np.pi * 10e3 * t)
x5 = V * np.sin(2 * np.pi * 25e3 * t)
x6 = V * np.sin(2 * np.pi * 18e3 * t)
x = x1 + x2 + x3 + x4 + x5 + x6
x = chirp(t, f0=20, f1=100000, t1=1, method='linear')
"""
 De acordo com o artigo Sistema de identificação autom ática de caracteríısticas de filtros analógicos 
 e substituição por filtros digitais implementados em SoC foi definido os valores de Q e K
"""

# passa alta e passa baixa
# Q = 1 / np.sqrt(2)
K = np.tan(np.pi * fc / fs)

# passa faixa e rejeita faixa
Q = fc/fb

# definição dos argumentos do filtro
b0 = K / ( K**2 * Q + K + Q)
b1 = 0
b2 = - b0
a1 = (2 * Q * (K**2 - 1)) / ( K**2 * Q + K + Q)
a2 = (K**2 * Q - K + Q) / ( K**2 * Q + K + Q)


# Variaveis de estado
xh1 = 0
xh2 = 0

yh1 = 0
yh2 = 0


y = np.zeros(len(x))

for n in range(0, len(x)):
    y[n] = b0 * x[n] + b1 * xh1 + b2 * xh2 - a1 * yh1 - a2 * yh2

    yh2 = yh1
    yh1 = y[n]

    xh2 = xh1
    xh1 = x[n]

n = len(y)  # tamanho do sinal
k = np.arange(n)  # vetor em k
T = n / fs
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
plt.plot(frq, 2 * abs(X), 'r')
plt.title('Fourier - Sweep')
plt.xlabel('Freq (Hz)')
plt.ylabel('|X(freq)|')

plt.subplot(3, 1, 3)
plt.plot(frq,  2 * abs(Y), 'r')
plt.grid(True, which="both")
plt.title('Fourier - Filtro')
plt.xlabel('Freq (Hz)')
plt.ylabel('|X(freq)|')


index_max_value = np.argmax(Y)
max_value = 2 * abs(Y[index_max_value])

band = max_value * np.sqrt(2)/2

index_band = np.argwhere(abs(Y) >= band/2)
fmin = frq[index_band[0]]
fmax = frq[index_band[len(index_band) - 1]]

deltaf = fmax - fmin

print("banda de frequência é de ",deltaf,"Hz, e  frequência central é de",frq[index_max_value],"Hz")
plt.axvline(x=frq[index_band[0]], ymin=-100, ymax=100, color='red', linestyle='--')
plt.axvline(x=frq[index_band[len(index_band) - 1]], ymin=-100, ymax=100, color='red', linestyle='--')
plt.tight_layout()
plt.show()