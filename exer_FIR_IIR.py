#  Copyright (c) 2023. Ânderson Felipe Weschenfelder

"""
Created on março 21 21:37:46 2023

@author: Ânderson Felipe Weschenfelder
"""
import matplotlib.pyplot as plt
import numpy as np


def IIR_filter(x, b_k, a_k,size):
    n_x = len(x)
    M = len(a_k)
    N = len(b_k)
    x_array = np.zeros(size)
    x_array[:n_x] = x
    array_a = np.zeros(size)
    array_b = np.zeros(size)
    y = np.zeros(size)
    for i in range(size):
        res_now_a = 0
        res_now_b = 0
        for k in np.arange(1, M, 1):
            res_now_a = res_now_a + a_k[k] * y[i - k]
        array_a[i] = res_now_a
        for k in range(N):
            res_now_b = res_now_b + b_k[k] * x_array[i - k]
        array_b[i] = res_now_b
        y[i] = array_a[i] + array_b[i]
    return y


def convolucao(x, b):
    n = len(x)  # tamanho do vetor de x(k)
    k = len(b)  # tamanho do vetor de b(k)
    x_n = np.zeros(n + k - 1)  # cria vetor novo de x(k) com pontos nulos. Necessario para calcular
    x_n[:n] = x  # adiciona os valores de x(k) ao vetor
    sum_array = np.zeros(n + k - 1)  # cria vetor de sáida
    for i in range(n + k - 1):
        res = float(0)
        for m in range(k):
            res = res + b[m] * x_n[i - m]
        sum_array[i] = res

    return sum_array


# questão 1
x_n = [5, 1, 8, 0, 2, 4, 6, 7]
b_n = [-1, 1]
conv = convolucao(x_n, b_n)
print("QUESTÂO 1: ", conv)


# questão 2
freq = 1
ts = (1 / freq) / 10
t = np.arange(0, 1 / freq, ts)  # vetor de tempo
x = np.cos(2 * np.pi * freq * t)
b1 = [0.4, 0.4]
b2 = [0.2, 0.2, 0.2, 0.2]
b3 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
b3 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

conv1 = convolucao(x, b1)
conv2 = convolucao(x, b2)
conv3 = convolucao(x, b3)

print("\n\n\nQUESTÂO 2\r\n")
print("conv1: \r\n", conv1)
print("\nconv2: \r\n", conv2)
print("\nconv3: \r\n", conv3)

plt.figure()
plt.plot(np.arange(0, len(conv1), 1), conv1)
plt.plot(np.arange(0, len(conv2), 1), conv2)
plt.plot(np.arange(0, len(conv3), 1), conv3)
#plt.show()

# questão 4
x_n = [6, 1, 3, 5, 1, 4]
b_n = [-1, 3, -1]
conv = convolucao(x_n, b_n)
print("\r\n\n\nQUESTÂO 4: ", conv)

# questão 5
ir1 = IIR_filter([0, 1, 0], [0.6, 0.2], [0, 0.4], 20)
ir3 = IIR_filter([0, 1, 0], [0.6, 0.2], [0, 1], 20)
ir2 = IIR_filter([0, 1, 0], [0.6, 0.2], [0, 1.1], 20)
plt.figure()
plt.plot(np.arange(0, len(ir1), 1), ir1)
plt.plot(np.arange(0, len(ir2), 1), ir2)
plt.plot(np.arange(0, len(ir3), 1), ir3)
plt.show()