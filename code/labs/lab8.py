import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from code.thinkdsp import decorate, SquareSignal, Wave, zero_pad

filePath = "../lab_captions/lab8/"
fileExtension = ".pdf"

"""
Exercise 8.1 
The notebook for this chapter is chap08.ipynb. Read through
it and run the code.
It contains an interactive widget that lets you experiment with the parameters
of the Gaussian window to see what effect they have on the cutoff
frequency.
What goes wrong when you increase the width of the Gaussian, std, without
increasing the number of elements in the window, M?
"""


def plot_filter(M=20, std=2, name="normal"):
    signal = SquareSignal(freq=440)
    wave = signal.make_wave(duration=1, framerate=44100)
    spectrum = wave.make_spectrum()

    gaussian = scipy.signal.gaussian(M=M, std=std)
    gaussian /= sum(gaussian)

    ys = np.convolve(wave.ys, gaussian, mode='same')
    smooth = Wave(ys, framerate=wave.framerate)
    spectrum2 = smooth.make_spectrum()

    amps = spectrum.amps
    amps2 = spectrum2.amps
    ratio = amps2 / amps
    ratio[amps < 560] = 0

    padded = zero_pad(gaussian, len(wave))
    dft_gaussian = np.fft.rfft(padded)

    plt.plot(np.abs(dft_gaussian), color='gray', label='Gaussian filter')
    plt.plot(ratio, label='amplitude ratio')

    decorate(xlabel='Frequency (Hz)', ylabel='Amplitude ratio')
    plt.savefig(filePath + "1." + name + ".plot" + fileExtension)
    plt.close()


plot_filter(M=20, std=2)
plot_filter(M=20, std=5, name="partly.changed")
plot_filter(M=20, std=20, name="fully.changed")

"""Как можно видеть, без увеличения M окно постепенно сжимается, высокочастотные гармоники спадают меделеннее, 
из-за чего появляются боковые лепестки """

"""
Exercise 8.2 
In this chapter I claimed that the Fourier transform of a Gaussian
curve is also a Gaussian curve. For discrete Fourier transforms, this
relationship is approximately true.
Try it out for a few examples. What happens to the Fourier transform as
you vary std?
"""

# "Напишем функцию, которая будет строить графики для гаусовской кривой и преобразования Фурье от гауссовской кривой"


def gaussian(M, std, name):
    gaussian = scipy.signal.gaussian(M=M, std=std)
    plt.subplot(1, 2, 1)
    plt.plot(gaussian)
    decorate(xlabel='Time')

    fft_gaussian = np.fft.fft(gaussian)
    fft_rolled = np.roll(fft_gaussian, M // 2)
    plt.subplot(1, 2, 2)
    plt.plot(np.abs(fft_rolled))
    decorate(xlabel='Frequency')
    plt.savefig(filePath + name + fileExtension)
    plt.close()


gaussian(32, 2, name="2.normal.plot")

# "При уменьшении std гауссова кривая сжимается а преобразование Фурье расширяется"
gaussian(32, 0.1, name="2.compressed.plot")

# "При увеличении std гауссова кривая расширяется а преобразование Фурье сжимается"
gaussian(32, 10, name="2.bloated.plot")

# "Таким образом, можно говорить об обратной зависимости между гауссовой кривой и ее преобразованием Фурье"

"""
Exercise 8.3 
If you did the exercises in Chapter 3, you saw the effect of the
Hamming window, and some of the other windows provided by NumPy,
on spectral leakage. We can get some insight into the effect of these windows
by looking at their DFTs.
In addition to the Gaussian window we used in this window, create a Hamming
window with the same size. Zero pad the windows and plot their
DFTs. Which window acts as a better low-pass filter? You might find it
useful to plot the DFTs on a log-y scale.
Experiment with a few different windows and a few different sizes.
"""

M_3 = 30
std_3 = 2.5
square_signal_3 = SquareSignal(freq=440)
wave_3 = square_signal_3.make_wave(duration=1, framerate=40000)
blackman_3 = np.blackman(M_3)
bartlett_3 = np.bartlett(M_3)
hamming_3 = np.hamming(M_3)
hanning_3 = np.hanning(M_3)
gaussian_3 = scipy.signal.gaussian(M=M_3, std=std_3)

windows_3 = [blackman_3, bartlett_3, gaussian_3, hanning_3, hamming_3]
names_3 = ['blackman', 'bartlett', 'gaussian', 'hanning', 'hamming']

for window_3 in windows_3:
    window_3 /= sum(window_3)

for window_3, name_3 in zip(windows_3, names_3):
    plt.plot(window_3, label=name_3)
decorate(xlabel='Index')
plt.savefig(filePath + "3.window.comparison.plot" + fileExtension)
plt.close()


def zero_pad(array, n):
    res = np.zeros(n)
    res[:len(array)] = array
    return res


def window_dfts(windows, names, file_name, yscale):
    for window, name in zip(windows, names):
        padded = zero_pad(window, len(wave_3))
        dft_window = np.fft.rfft(padded)
        plt.plot(abs(dft_window), label=name)
    if yscale != '':
        decorate(xlabel='Frequency (Hz)', yscale=yscale)
    else:
        decorate(xlabel='Frequency (Hz)')
    plt.savefig(filePath + file_name + fileExtension)
    plt.close()


window_dfts(windows_3, names_3, file_name="3.windows.frequency.comparison", yscale='')

window_dfts(windows_3, names_3, file_name="3.windows.frequency.log.comparison", yscale='log')

"""В результате анализа всех графиков можно прийти к выводу, что окно Хемминга является лучшим вариантом для фильтрации
НЧ лучей, так как дает меньше всего "выпуклостей" и, как видно из логарифмического графика,
имеет самые стойкие боковые лепестки"""
