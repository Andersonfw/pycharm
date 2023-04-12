#  Copyright (c) 2023. Ânderson Felipe Weschenfelder

"""
Created on março 15 20:18:40 2023

@author: Ânderson Felipe Weschenfelder
"""

"""
DFT
A transformada discreta de Fourier e dada por:

X(k) = DFT[(x(n)] = somatório( x[n]*e^((-j2 pi n k)/N); Onde o somatório vai de n=0, até N-1, sendo N o número de pontos do sinal x[n];

Desta forma e^((-j2 pi n k)/N) é definido pela formula de EULER como sendo:

cos(2*pi*n*k/N) - jsen(2*pi*n*k/N)

o k no calculo da DFT representa as frequências, para cada valor representa k vezes a frequência de resolução. 
A frequência de resolução é definida pelo valor no qual é dividido a frequência de amostragem. 
Por exemplo,    
    f = 100Hz
    Tendo fs=1kHz; 
    A largura de banda será de a metade de fs, 500Hz;
    A taxa de amostragem (ts) será de de 1ms;
    o sinal x(t) será amostrado a cada 1ms por um tempo total de T;
    Assim a resolução será a divisão da fs pelo número de amostras (N) obtidas em T, separadas por ts.
    Sendo T = 2 segundos, N = T/ts = 2/1m = 2000; Assim  1k/2000 =  2 Hz 


Para calcular a DFT no índice onde k = 0, a parte complexa e^((-j2 pi n k)/N) é sempre constante 1, assim, x[0] é a somatório das N amostras
de x(n), e sendo uma senoide resulta em zero, partes positivas se somam as negativas.
Nos demais índices de k, a parte complexa movimenta-se dentro de um circulo em passos definidos por n*k, já que o restante é constante,
para cada valor representa um ponto diferente dentro do circulo trigonométrico.
Quando k for igual ao valor de f do sinal x[n] resultara é um valor x[k] maior que zero, significando que a frequência que este valor de k representa
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

Percebe-se que a parte real do complexo é exatamente igual ao sinal x[n], significando que esta frequência está contida no sinal,
a parte imaginaria o somatório é zero.

Por se tratar de um função par o cosseno da DFT, ela possui uma segunda frequência que vai gerar exatamente o mesmo valor, 
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
de 3 para 7, tem-se um acréscimo de 4*pi, o que significa 2 voltas no circulo trigonométrico, que também é o ponto de 
partida. Assim para k=3 ou k=7 obtém-se valores de magnitude elevados;

Para calcular a DFT corretamente são necessários dois passos importantes:

*   Normalizar o valor de X[k] pelo valor de N, pois o somatório, como diz o nome, soma N vezes seus valores. 
E para ficar coerente o resultando é normalizado por N
*   Para mostrar o gráfico com a magnitude e frequência correta, é utilizado apenas metade dos pontos de x[k] e estes são ainda
dobrados de valor.

"""

import matplotlib.pyplot as plt
import numpy as np
import datetime


class Modulo:
    def __init__(self, x_k, mode):
        self.x_k = x_k
        self.mode = mode
        self.mod, self.ang = calcular_modulo(x_k, mode)


def calcular_modulo(x_k, mode):
    if mode == 'SSB':
        x_k = x_k[range(int(len(x_k) / 2))]
    mag = np.zeros(len(x_k))
    fase = np.zeros(len(x_k))
    for i in range(len(x_k)):
        mag[i] = np.sqrt(x_k[i].real ** 2 + x_k[i].imag ** 2)
        if x_k[i].real != 0:
            fase[i] = np.degrees(np.arctan(x_k[i].imag / x_k[i].real))
        else:
            fase[i] = 0
    return mag, fase


def dft(x_n):
    x_k = np.array([])
    size = len(x_n)
    for m in range(size):
        dftsum = 0
        for n in range(size):
            dftsum = dftsum + x_n[n] * (
                    (np.cos(2 * np.pi * n * m / size)) - (1j * np.sin(2 * np.pi * n * m / size)))
        x_k = np.append(x_k, dftsum / size)

    return x_k


if __name__ == "__main__":
    Fs = 1000  # taxa de amostragem
    Ts = 1.00 / Fs  # periodo de amostragem
    Freq_resolution = 0.5  # frequência de resolução da DFT
    t = np.arange(0, 1 / Freq_resolution, Ts)  # vetor de tempo
    starTime = datetime.datetime.now()
    print("iniciando simulação da DFT em: ", starTime.strftime("%H:%M:%S"))

    f = 300  # frequencia do sinal 1
    x1_n = np.cos(2 * np.pi * f * t + 0)
    f = 10  # frequencia do sinal 2
    x2_n = np.sin(2 * np.pi * f * t + np.pi)

    # Generate the white noise signal
    length = len(x1_n)  # Set the length of the white noise signal
    # Set the mean and standard deviation of the white noise distribution
    mean = 0
    std_dev = 0#0.5
    white_noise = np.random.normal(mean, std_dev, length)

    x = x1_n + x2_n + white_noise
    size_x = len(x)  # tamanho do vetor do sinal

    k = np.arange(size_x)  # vetor em k
    T = size_x / Fs  # resolução de frequência = 1Hz
    frq = k / T  # os dois lados do vetor de frequencia
    frq = frq[range(int(size_x / 2))]  # apenas um lado

    X = np.fft.fft(x) / size_x  # calculo da fft e normalização por n
    X = X[range(int(size_x / 2))]

    X_k = dft(x)
    mod = Modulo(X_k, 'SSB')
    freq = (Fs / size_x) * np.arange(size_x)
    freq = freq[range(int(size_x / 2))]

    fig, ax = plt.subplots(3, 1)
    # adiciona um título para a figura
    fig.suptitle('DFT')

    # define o espaçamento entre subplots
    fig.subplots_adjust(wspace=0.4, hspace=0.2)

    ax[0].plot(t, x, label='Sinal Discretizado')
    ax[0].set_xlabel('Tempo')
    ax[0].set_ylabel('Amplitude')
    ax[0].legend()
    # ax[0].set_title('Sinal Discretizado')

    ax[1].plot(frq, 2 * abs(X), 'r', label='FFT com Numpy')
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|X(freq)|')
    ax[1].legend()
    # ax[1].set_title('FFT com Numpy')

    ax[2].plot(freq, 2 * mod.mod, 'g', label='DFT calculada')
    ax[2].set_xlabel('Freq (Hz)')
    ax[2].set_ylabel('|X(freq)|')
    ax[2].legend()
    # ax[2].set_title('DFT calculada')

    stopTime = datetime.datetime.now()
    diftime = stopTime - starTime
    print("Encerando a simulação da DFT em: ", stopTime.strftime("%H:%M:%S"))
    print("Duração da simulação: ", diftime.total_seconds())

    plt.tight_layout()
    plt.show()
