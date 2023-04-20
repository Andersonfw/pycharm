import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import chirp
import control

L = 100e-6  # Indutor
R = 1   # Resistor
C = 220e-6  # Capacitor

# Função de transferência em S
num = [1]  # Numerador
den = [L*C,R*C,  1]  # Denominador

# Taxa de amostragem em Hz
fs = 44100

# Método de discretização (por exemplo, método de Tustin)
method = 'bilinear'

# Ordem do filtro
order = len(den) - 1

# Converter função de transferência para função de transferência discreta em Z
# retornando os coefficients a_n e b_n. Podendo ser usado outros metodos de conversão
num_d, den_d, dt = signal.cont2discrete((num, den), 1/fs, method=method)

# Imprimir os coeficientes discretos discreta em Z
print("Numerador discreto em Z: ", num_d)
print("Denominador discreto em Z: ", den_d)
print("Passo de tempo: ", dt)

print("")
print("")

# Utilizando outra função para aconversão
b, a = signal.bilinear(num,den, fs)

# Imprimir os coeficientes discretos discreta em Z
print("Numerador discreto em Z: ", num_d)
print("Denominador discreto em Z: ", den_d)
print("Passo de tempo: ", dt)

# Imprimir coeficientes do filtro digital
print("b0: ", b[0])
print("b1: ", b[1])
print("b2: ", b[2])
print("a0: ", a[0])
print("a1: ", a[1])
print("a2: ", a[2])

b0 = b[0]
b1 = b[1]
b2 = 0
a0 = a[0]
a1 = a[1]
a2 = a[2]

t = np.arange(0,1,1/fs) # vetor de tempo
x = chirp(t, f0=20, f1=20000, t1=1, method='linear')
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