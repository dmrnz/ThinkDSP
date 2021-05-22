import matplotlib.pyplot as plt
import numpy as np

from code.thinkdsp import SawtoothSignal, decorate

filePath = "/code/lab_captions/lab7/"
fileExtension = ".pdf"

"""
Exercise 7.1 

The notebook for this chapter, chap07.ipynb, contains 
additional examples and explanations.
Read through it and run the code.
"""

wave_1 = SawtoothSignal(freq=500).make_wave(duration=0.1, framerate=10000)

spectrum_1 = wave_1.make_spectrum(full=True)

spectrum_1.plot()
decorate(xlabel='Frequency (Hz)', ylabel='DFT')
plt.savefig(filePath + "1.full.fft.demo" + fileExtension)
plt.close()

"""
Exercise 7.2

In this chapter, I showed how we can express the DFT and inverse DFT
as matrix multiplications.  These operations take time proportional to
N^2, where N is the length of the wave array.  That is fast enough
for many applications, but there is a faster
algorithm, the Fast Fourier Transform (FFT), which takes time
proportional to N * log(N).

The key to the FFT is the Danielson-Lanczos lemma:

DFT(y)[n] = DFT(e)[n] + exp(-2 * pi * i * n / N) DFT(o)[n]

Where  DFT(y)[n] is the nth element of the DFT of y;
e is the even elements of y, and o is the odd elements of y.

This lemma suggests a recursive algorithm for the DFT:

1. Given a wave array, y, split it into its
even elements, e, and its odd elements, o.

2. Compute the DFT of e and o by making recursive calls.

3. Compute DFT(y) for each value of n using the Danielson-Lanczos lemma.

For the base case of this recursion, you could wait until the length
of y is 1.  In that case, DFT(y) = y.  Or if the length of y
is sufficiently small, you could compute its DFT by matrix multiplication,
possibly using a precomputed matrix.
"""

ys_2 = [-0.2, 0.9, 0.5, -0.3]
hs_2 = np.fft.fft(ys_2)

# [ 0.9+0.j  -0.7-1.2j -0.3+0.j  -0.7+1.2j]
print(hs_2)


def effective_fft(ys):
    # noinspection PyPep8Naming
    N = len(ys)
    if N == 1:
        return ys

    fft_even = np.tile(effective_fft(ys[::2]), 2)
    fft_odd = np.tile(effective_fft(ys[1::2]), 2)

    ns = np.arange(N)
    exp = np.exp(-1j * 2 * np.pi * ns / N)

    return fft_even + exp * fft_odd


hs2_2 = effective_fft(ys_2)
print(np.sum(np.abs(hs_2 - hs2_2)))
# 4.065457153363882e-16
