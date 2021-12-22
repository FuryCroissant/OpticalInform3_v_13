import numpy as np
import matplotlib.pyplot as plt
from scipy.special import *
import time

# Мода Бесселя
def f(r, m, n):
    return jv(abs(n),m*r)
def fto2d(y):
    n = len(y)
    arr_2d = np.zeros((2 * n, 2 * n), dtype=np.complex)
    j, k = np.meshgrid(np.arange(0, 2 * n), np.arange(0, 2 * n))
    j = j - n
    k = k - n
    #индекс каждой ячейки
    a = np.round(np.sqrt(j ** 2 + k ** 2)).astype(np.int)
    mask = a < n
    arr_2d[mask] = y[a[mask]]
    fi = np.arctan2(k, j)
    return arr_2d * np.exp(complex(0, 1) * m * fi)

def hankel(x, y, m):
    x2 = x
    Y = np.zeros(N, dtype=np.complex128)
    for i, j in zip(x2, range(len(x))):
        Y[j] = np.sum(y * jv(m, 2 * np.pi * x * i) * x * (R / N))
    return Y * (2 * np.pi / (complex(0, 1) ** m))
# Отрисовка одномерных графиков
def plot(x, y, label):
    _, arr = plt.subplots(1, 2, figsize=(15, 5))
    arr[0].plot(x, np.absolute(y), color='b')
    arr[0].grid()
    arr[0].set_title('Амплитуда'+label)
    arr[1].plot(x, np.angle(y), color='b')
    arr[1].grid()
    arr[1].set_title('Фаза'+label)
    plt.show()

# Отрисовка двумерных графиков
def plot_2d(y2d, label):
    fig, arr = plt.subplots(1, 2, figsize=(15, 5))
    amp = arr[0].imshow(np.absolute(y2d), cmap='hot', interpolation='nearest')
    arr[0].set_title('Амплитуда'+ label)
    fig.colorbar(amp, ax=arr[0])
    phase = arr[1].imshow(np.angle(y2d), cmap='hot', interpolation='nearest')
    arr[1].set_title('Фаза'+label)
    fig.colorbar(phase, ax=arr[1])
    plt.show()

# Одномерное БПФ
def FFT(f, a, b, n, m):
    h = (b - a) / (N - 1)
    # дополнение нулями, разбиение на две части и их обмен
    zeros = np.zeros(int((m - n) / 2))
    f = np.concatenate((zeros, f, zeros), axis=None)
    # Свап частей вектора:
    middle = int(len(f) / 2)
    f = np.concatenate((f[middle:], f[:middle]))
    # БПФ
    F = np.fft.fft(f, m) * h
    # Свап частей вектора:
    middle = int(len(F) / 2)
    F = np.concatenate((F[middle:], F[:middle]))
    # Выделение центральных N отсчетов:
    F = F[int((m - n) / 2): int((m - n) / 2 + n)]
    return F


# Двумерное БПФ
def FFT2d(function, a, b, n, m):
    fft = np.zeros((n, n), dtype=complex)
    temp = np.zeros((n, n), dtype=complex)
    # проход по строкам
    for i in range(n):
        temp[i] = FFT(function[i], a, b, n, m)
    temp = temp.T
    # проход по столбцам
    for i in range(n):
        fft[i] = FFT(temp[i], a, b, n, m)
    return fft.T


N = 300
M = 4096
m = 2
n = 3
R = 5
x = np.linspace(0, R, N)
y = f(x, n, m)
plot(x, y, " входной функции")
y2d = fto2d(y)
plot_2d(y2d, " восстановленного изображения")
start = time.time()
y_hankel = hankel(x, y, m)
y_hankel_2D = fto2d(y_hankel)
end = time.time()
print("Преобразование Ханкеля: %s сек" % (end - start))
plot(x, y_hankel, " после преобразования Ханкеля")
plot_2d(y_hankel_2D, " после преобразования Ханкеля")
N = 2 * N
start1 = time.time()
y_f = FFT2d(y2d, 0, R, N, M)
end1 = time.time()
print("БПФ: %sсек" % (end1 - start1))
plot_2d(y_f, " БПФ")