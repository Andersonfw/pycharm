#  Copyright (c) 2023. Ânderson Felipe Weschenfelder

"""
Created on março 15 20:18:40 2023

@author: Ânderson Felipe Weschenfelder
"""

"""
DFT
A transformada discreta de Fourier e dada por:

X(k) = DFT[(x(n)] = somatorio( x[n]*e^((-j2 pi n k)/N); Onde o somátorio vai de n=0, até N-1, sendo N o número de pontos do sinal x[n];

Desta forma e^((-j2 pi n k)/N) é definido pela formúla de EULER como sendo:

cos(2*pi*n*k/N) - jsen(2*pi*n*k/N)

o k no calculo da DFT representa as frequências, para cada valor representa k vezes a frequência de resolução. 
A frequência de resolução é definida pelo valor no qual é dividido a frequência de amostragem. 
Por exemplo,    
    Tendo fs=1kHz; 
    A largura de banda será de 1kHz; 
    A taxa de amostragem (ts) será de de 1ms;
    o sinal x(t) será amostrado a cada 1ms por um tempo total de T;
    Assim a resolução será a divisão da fs pelo número de amostras (N) obtidas no T, separadas por ts.
    Neste exemplo fs/T/ts; sendo T = 0.5 segundos; Assim  1k/0.5/1m = 1/T = 1/0.5 = 2 Hz 


Para calcular a DFT no indice onde k = 0, a parte complexa e^((-j2 pi n k)/N) é sempre constante 1, assim, x[0] é a somatorio das N amostras
de x(n), e sendo uma senoide resulta em zero, partes positivas se somam as negativas.
Nos demais indices de k, a parte complexa movimenta-se dentro de um circulo em passos definidos por n*k, já que o restante é constante,
para cada valor representa um ponto diferente dentro do circulo trigonomêtrico.
Quando k for igual ao valor de f do sinal x[n] ressultara é um valor x[k] maior que zero, significando que a frequência que este valor de k representa
está contido dentro do sinal de x[n]:


Agora suponha um sinal X[n] definido como:

X[n] = cos(2*pi*f*t), onde t = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], e f = 3 Hz.
Perceba que ts= 0.1s, logo fs=10Hz, N = 10 e a resolução em frequência é de 1/T = 1/1 = 1Hz;

Na DFT quando k = 3 teremos:

cos(2*pi*n*k/N) - jsen(2*pi*n*k/N) = cos(2*pi*n*3/10) - jsen(2*pi*n*3/10)

Agora variando o valor de n de 0 até 9,  temos:

n = 0 ;  cos(2*pi*3*n/10) - jsen(2*pi*3*n/10) =  cos(2*pi*3*0) - jsen(2*pi*3*0) = 0 +0j = X[n] = cos(2*pi*3*0)  = 0;
n = 1 ;  cos(2*pi*3*n/10) - jsen(2*pi*3*n/10) =  cos(2*pi*3*0.1) - jsen(2*pi*3*0.1) = -0.30 - j0.95 = X[n] = cos(2*pi*3*0.1) = -0.30
n = 2 ;  cos(2*pi*3*n/10) - jsen(2*pi*3*n/10) =  cos(2*pi*3*0.2) - jsen(2*pi*3*0.2) = -0.80 + j0.58 = X[n] = cos(2*pi*3*0.2) = -0.80
.
.
.
n = 9 ;  cos(2*pi*3*n/10) - jsen(2*pi*3*n/10) =  cos(2*pi*3*0.9) - jsen(2*pi*3*0.9) = -0.30 +j0.95 = X[n] = cos(2*pi*3*0.9) = -0.30

Percebe-se que a parte real do complexo é exatamente igual ao sinal x[n], significando que esta frequência está contida no sinal, e o somatorio
das partes imaginarias resulta em zero;

Por se tratar de um função par o cosseno da DFT, ela possui uma segunda frequência que vai gerar extamente o mesmo valor, 
está segunda frequência está contida em N-k, ou seja 10-3 = 7; 
Em k = 7 será obtido um segundo pico de magnitude da DFT.

Quando k = 7;

n = 1 ;  cos(2*pi*7*n/10) - jsen(2*pi*7*n/10) =  cos(2*pi*7*0.1) - jsen(2*pi*7*0.1) = -0.30 - j0.95 = X[n] = cos(2*pi*3*0.1) = -0.30
n = 2 ;  cos(2*pi*7*n/10) - jsen(2*pi*7*n/10) =  cos(2*pi*7*0.2) - jsen(2*pi*7*0.2) = -0.80 + j0.58 = X[n] = cos(2*pi*3*0.2) = -0.80
.
.
.
n = 9 ;  cos(2*pi*7*n/10) - jsen(2*pi*7*n/10) =  cos(2*pi*7*0.9) - jsen(2*pi*7*0.9) = -0.30 +j0.95 = X[n] = cos(2*pi*3*0.9) = -0.30

Isso ocorre devido a função cosseno ser uma função par, ou seja cos(-x)=cos(x), e também pode ser visto que mudando k 
de 3 para 7, tem-se um acrescimo de 4*pi, o que signifa 2 voltas no circulo trigónométrico, que também é o ponto de 
partida. Assim para k=3 ou k=7 obtém-se valores de magnitude elevados;

Para calcular a DFT corretamente são necessários dois passos importantes:

*   Normalizar o valor de X[k] pelo valor de N, pois o somátorio, como diz o nome, soma N vezes seus valores. 
E para ficar coerente o resultando é normalizado por N
*   Para mostrar o gráfico com a magnitude e frequência correta, é utilizado apenas metade dos pontos de x[k] e estes são ainda
dobrados de valor.

"""

import matplotlib.pyplot as plt
import numpy as np

Fs = 10  # taxa de amostragem
Ts = 1.00 / Fs  # periodo de amostragem
Freq_resolution = 1  # frequência de resolução da DFT
t = np.arange(0, 1 / Freq_resolution, Ts)  # vetor de tempo

# x =np.cos(2* np.pi*freq*t + 0.5*np.sin(2*np.pi*10e6*t))

f = 3  # frequencia do sinal 1
x1_n = np.cos(2 * np.pi * f * t + 0)
f = 1  # frequencia do sinal 2
x2_n = np.sin(2 * np.pi * f * t + np.pi)

# Set the length of the white noise signal
length = len(x1_n)
# Set the mean and standard deviation of the white noise distribution
mean = 0
std_dev = 0.5

# Generate the white noise signal
white_noise = 0  # np.random.normal(mean, std_dev, length)
x_n = x1_n  # x1_n + x2_n + white_noise
size_x = len(x_n)  # tamanho do vetor do sinal

print("iniciando")
a = 0
X_k = np.array([])  # Cria um array vazio
# X_k = np.append(X_k, 0)
# X_k[0] = 0
for k in range(size_x):
    dftsum = 0
    sumx = 0
    sum_complex = 0
    for n in range(size_x):
        xPlus = x_n[n]
        Complex = (np.cos(2 * np.pi * n * k / size_x)) - (1j * np.sin(2 * np.pi * n * k / size_x))
        k_now = xPlus * Complex
        sumx = sumx + xPlus
        sum_complex = sum_complex + Complex
        dftsum = dftsum + k_now
        # dftsum = dftsum + x_n[n + 1] * (
        #         (np.cos(2 * np.pi * n * k / size_x)) - (1j * np.sin(2 * np.pi * n * k / size_x)))
    x_k = np.append(X_k, dftsum)
    X_k = np.append(X_k, dftsum / size_x)
    # X_k = np.append(X_k, dftsum / size_x)
    # X_k.append(k+1)
    # X_k[k+1] =  dftsum
    # print("dftsum",dftsum)
    # print("xk",X_k[k])

k = np.arange(size_x)  # vetor em k
T = size_x / Fs  # resolução de frequência = 1Hz
frq = k / T  # os dois lados do vetor de frequencia
# frq = frq[range(int(size_x / 2))]  # apenas um lado

X = np.fft.fft(x_n) / size_x  # calculo da fft e normalização por n
# X = X[range(int(size_x / 2))]
# X_k = X_k[range(int(size_x / 2))]

mag = np.zeros(len(X_k))
fase = np.zeros(len(X_k))
for i in range(len(X_k)):
    mag[i] = np.sqrt(X_k[i].real ** 2 + X_k[i].imag ** 2)
    if X_k[i].real == 0:
        fase[i] = 0
    else:
        fase[i] = np.degrees(np.arctan(X_k[i].imag / X_k[i].real))

freq = (Fs / size_x) * np.arange(size_x)
# freq = freq[range(int(size_x / 2))]

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, x_n)
ax[0].set_xlabel('Tempo')
ax[0].set_ylabel('Amplitude')

ax[1].plot(frq, 2 * abs(X), 'r')
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|X(freq)|')

ax[2].plot(freq, 2 * mag, 'r')
ax[2].set_xlabel('Freq (Hz)')
ax[2].set_ylabel('|X(freq)|')
print("terminou")
plt.show()
