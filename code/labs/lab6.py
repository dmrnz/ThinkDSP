import timeit

from code.thinkdsp import UncorrelatedGaussianNoise, decorate, read_wave, Spectrogram
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

filePath = "../lab_captions/lab6/"
fileExtension = ".pdf"

"""
Exercise 6.1 In this chapter I claim that analyze1 takes time proportional
to n^3 and analyze2 takes time proportional to n^2. To see if that’s true, run
them on a range of input sizes and time them. In Jupyter, you can use the
“magic command” %timeit.
If you plot run time versus input size on a log-log scale, you should get a
straight line with slope 3 for analyze1 and slope 2 for analyze2.
You also might want to test dct_iv and scipy.fftpack.dct.
"""

signal_6_1 = UncorrelatedGaussianNoise()
noise_6_1 = signal_6_1.make_wave(duration=1.0, framerate=16384)
print(noise_6_1.ys.shape)

loglog = dict(xscale='log', yscale='log')
fig_number = 1

def plot_bests(ns, bests, fig_name):
    plt.plot(ns, bests)
    decorate(**loglog)
    plt.savefig(filePath + "1.best.plot." + fig_name + fileExtension)
    plt.close()
    x = np.log(ns)
    y = np.log(bests)
    t = linregress(x, y)
    slope = t[0]

    return slope


PI2 = np.pi * 2


def analyze1(ys, fs, ts):
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    amps = np.linalg.solve(M, ys)
    return amps


def run_speed_test(ns, func):
    results = []
    for N in ns:
        print(N)
        ts = (0.5 + np.arange(N)) / N
        freqs = (0.5 + np.arange(N)) / 2
        ys = noise_6_1.ys[:N]
        result = timeit.timeit(stmt='func(ys, freqs, ts)', number=1)
        results.append(result)

    bests = [result.best for result in results]
    return bests


ns = 2 ** np.arange(6, 13)

bests_6_1 = run_speed_test(ns, analyze1)
print(plot_bests(ns, bests_6_1, "analyze1"))


def analyze2(ys, fs, ts):
    args = np.outer(ts, fs)
    M = np.cos(PI2 * args)
    amps = np.dot(M, ys) / 2
    return amps


bests_6_1_2 = run_speed_test(ns, analyze2)
plot_bests(ns, bests_6_1_2, "analyze2")


def scipy_dct(ys, freqs, ts):
    return scipy.fftpack.dct(ys, type=3)


bests_6_1_3 = run_speed_test(ns, scipy_dct)
plot_bests(ns, bests_6_1_3, "fftpack.dct")

plt.plot(ns, bests_6_1, label='analyze1')
plt.plot(ns, bests_6_1_2, label='analyze2')
plt.plot(ns, bests_6_1_3, label='fftpack.dct')

decorate(xlabel='Wave length (N)', ylabel='Time (s)', **loglog)
plt.savefig(filePath + "1.bests.comparing" + fileExtension)
plt.close()

"""
Exercise 6.2 One of the major applications of the DCT is compression for
both sound and images. In its simplest form, DCT-based compression
works like this:
1. Break a long signal into segments.
2. Compute the DCT of each segment.
3. Identify frequency components with amplitudes so low they are inaudible,
and remove them. Store only the frequencies and amplitudes
that remain.
4. To play back the signal, load the frequencies and amplitudes for each
segment and apply the inverse DCT.
Implement a version of this algorithm and apply it to a recording of music
or speech. How many components can you eliminate before the difference
is perceptible?
74 Chapter 6. Discrete cosine transform
In order to make this method practical, you need some way to store a sparse
array; that is, an array where most of the elements are zero. NumPy provides
several implementations of sparse arrays, which you can read about
at http://docs.scipy.org/doc/scipy/reference/sparse.html.
"""

wave_6_2 = read_wave('100475__iluppai__saxophone-weep.wav')
wave_6_2.make_audio()


segment_6_2 = wave_6_2.segment(start=1.2, duration=0.5)
segment_6_2.normalize()
segment_6_2.make_audio()


seg_dct_6_2 = segment_6_2.make_dct()
seg_dct_6_2.plot(high=4000)
decorate(xlabel='Frequency (Hz)', ylabel='DCT')
plt.savefig(filePath + "2.bests.comparing" + fileExtension)
plt.close()


def compress(dct, thresh=1):
    count = 0
    for i, amp in enumerate(dct.amps):
        if np.abs(amp) < thresh:
            dct.hs[i] = 0
            count += 1

    n = len(dct.amps)
    print(count, n, 100 * count / n, sep='\t')


seg_dct_6_2 = segment_6_2.make_dct()
compress(seg_dct_6_2, thresh=10)
seg_dct_6_2.plot(high=4000)


segment_6_2_2 = seg_dct_6_2.make_wave()
segment_6_2_2.make_audio()



def make_dct_spectrogram(wave, seg_length):
    """Computes the DCT spectrogram of the wave.

    seg_length: number of samples in each segment

    returns: Spectrogram
    """
    window = np.hamming(seg_length)
    i, j = 0, seg_length
    step = seg_length // 2

    # map from time to Spectrum
    spec_map = {}

    while j < len(wave.ys):
        segment = wave.slice(i, j)
        segment.window(window)

        # the nominal time for this segment is the midpoint
        t = (segment.start + segment.end) / 2
        spec_map[t] = segment.make_dct()

        i += step
        j += step

    return Spectrogram(spec_map, seg_length)


spectrum_6_2 = make_dct_spectrogram(wave_6_2, seg_length=1024)
for t, dct in sorted(spectrum_6_2.spec_map.items()):
    compress(dct, thresh=0.2)


wave_6_2_2 = spectrum_6_2.make_wave()
wave_6_2_2.make_audio()


wave_6_2.make_audio()

"""
Exercise 6.3 In the repository for this book you will find a Jupyter notebook
called phase.ipynb that explores the effect of phase on sound perception.
Read through this notebook and run the examples. Choose another segment
of sound and run the same experiments. Can you find any general
relationships between the phase structure of a sound and how we perceive
it?
"""
