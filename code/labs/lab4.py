import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from code.thinkdsp import read_wave, decorate, Spectrum, Wave, Noise

filePath = "/code/lab_captions/lab4/"
fileExtension = ".pdf"

"""
Exercise 4.1

``A Soft Murmur'' is a web site that plays a mixture of natural
noise sources, including rain, waves, wind, etc.  At
https://asoftmurmur.com/about/ you can find their list
of recordings, most of which are at https://freesound.org.

Download a few of these files and compute the spectrum of each
signal. Does the power spectrum look like white noise, pink noise,
or Brownian noise?  How does the spectrum vary over time?
"""
wave_1 = read_wave('../139337__felix-blume__wind.wav')
wave_1.make_audio()

segment_1 = wave_1.segment(start=140, duration=1.0)
segment_1.make_audio()

spectrum_1 = segment_1.make_spectrum()
spectrum_1.plot_power()
decorate(xlabel='Frequency (Hz)')
plt.savefig(filePath + "1.wind.spectrum" + fileExtension)
plt.close()

spectrum_1.plot_power()
loglog = dict(xscale='log', yscale='log')
decorate(xlabel='Frequency (Hz)', **loglog)
plt.savefig(filePath + "1.wind.spectrum.log" + fileExtension)
plt.close()

segment_1.make_spectrogram(512).plot(high=5000)
decorate(xlabel='Time(s)', ylabel='Frequency (Hz)')
plt.savefig(filePath + "1.wind.spectrogram" + fileExtension)
plt.close()

print(spectrum_1.estimate_slope().slope)
# -1.4066186066815058

wave_1_2 = read_wave('../83986__inchadney__fireplace.wav')
wave_1_2.make_audio()

segment_1_2 = wave_1_2.segment(start=8, duration=1.0)
segment_1_2.make_audio()

spectrum_1_2 = segment_1_2.make_spectrum()
spectrum_1_2.plot_power()
decorate(xlabel='Frequency (Hz)')
plt.savefig(filePath + "1.2.fire.spectrum" + fileExtension)
plt.close()

spectrum_1_2.plot_power()
decorate(xlabel='Frequency (Hz)', **loglog)
plt.savefig(filePath + "1.2.fire.spectrum.log" + fileExtension)
plt.close()

segment_1_2.make_spectrogram(512).plot(high=5000)
decorate(xlabel='Time(s)', ylabel='Frequency (Hz)')
plt.savefig(filePath + "1.2.fire.spectrogram" + fileExtension)
plt.close()

print(spectrum_1_2.estimate_slope().slope)
# -1.0766220332743197

"""
Exercise 4.2

In a noise signal, the mixture of frequencies changes over time.
In the long run, we expect the power at all frequencies to be equal,
but in any sample, the power at each frequency is random.

To estimate the long-term average power at each frequency, we can
break a long signal into segments, compute the power spectrum
for each segment, and then compute the average across
the segments.  You can read more about this algorithm at
https://en.wikipedia.org/wiki/Bartlett's_method.

Implement Bartlett's method and use it to estimate the power
spectrum for a noise wave.  Hint: look at the implementation
of `make_spectrogram`.
"""


def bartlett_spectrum(wave, seg_length=512, win_flag=True):
    spectrum_list = wave.make_spectrogram(seg_length, win_flag).spec_map.values()
    psd_array = [spectrum.power for spectrum in spectrum_list]

    return Spectrum(np.sqrt(sum(psd_array) / len(psd_array)),
                    next(iter(spectrum_list)).fs,
                    wave.framerate)


psd_2_1 = bartlett_spectrum(wave_1.segment(start=120, duration=1.0))
psd_2_2 = bartlett_spectrum(wave_1.segment(start=150, duration=1.0))
psd_2_3 = bartlett_spectrum(wave_1.segment(start=160, duration=1.0))

psd_2_1.plot_power()
psd_2_2.plot_power()
psd_2_3.plot_power()

decorate(xlabel='Frequency (Hz)',
         ylabel='Power',
         **loglog)
plt.savefig(filePath + "2.bartlett.estimation" + fileExtension)
plt.close()

"""
Exercise 4.3

At https://coindesk.com/price/bitcoin you can download
the daily price of a BitCoin as a CSV file. Read this file
and compute the spectrum of BitCoin prices as a function of time.
Does it resemble white, pink, or Brownian noise?
"""

df_3 = pd.read_csv('../BTC_USD_2013-10-01_2021-05-08-CoinDesk.csv',
                   parse_dates=[0])

ys_3 = df_3['Closing Price (USD)']
ts_3 = df_3.index

wave_3 = Wave(ys_3, ts_3, framerate=1)
wave_3.plot()
decorate(xlabel='Time (days)')
plt.savefig(filePath + "3.bitcoin.graph" + fileExtension)
plt.close()

spectrum_3 = wave_3.make_spectrum()
spectrum_3.plot_power()
decorate(xlabel='Frequency (1/days)', **loglog)
plt.savefig(filePath + "3.bitcoin.spectrum" + fileExtension)
plt.close()

print(spectrum_3.estimate_slope()[0])
# -1.7851119050620752

"""
Exercise 4.4

Write a class called `UncorrelatedPoissonNoise` that inherits
from ` _Noise` and provides `evaluate`. It should use `np.random.poisson`
to generate random values from a Poisson distribution. The parameter of
this function, `lam`, is the average number of particles
during each interval. You can use the attribute `amp` to specify `lam`.
For example, if the framerate is 10 kHz and `amp` is 0.001,
we expect about 10 “clicks” per second.

Generate about a second of UP noise and listen to it.
For low values of `amp`, like 0.001, it should sound
like a Geiger counter. For higher values it should sound
like white noise. Compute and plot the power spectrum
to see whether it looks like white noise.
"""


class UncorrelatedPoissonNoise(Noise):
    def evaluate(self, ts):
        ys = np.random.poisson(self.amp, len(ts))
        return ys


amp_4 = 0.001
framerate_4 = 10000
duration_4 = 1

signal_4 = UncorrelatedPoissonNoise(amp=amp_4)
wave_4 = signal_4.make_wave(duration=duration_4, framerate=framerate_4)
wave_4.make_audio()

expected_4 = amp_4 * framerate_4 * duration_4
actual_4 = sum(wave_4.ys)
print(expected_4, actual_4)
# 10.0 10

wave_4.plot()
plt.savefig(filePath + "4.radiation.wave" + fileExtension)
plt.close()

amp_4 = 2

signal_4_2 = UncorrelatedPoissonNoise(amp=amp_4)
wave_4_2 = signal_4_2.make_wave(duration=duration_4, framerate=framerate_4)
wave_4_2.make_audio()

expected_4_2 = amp_4 * framerate_4 * duration_4
actual_4_2 = sum(wave_4_2.ys)
print(expected_4_2, actual_4_2)
# 20000 20051

wave_4_2.plot()
plt.savefig(filePath + "4.radiation.high.wave" + fileExtension)
plt.close()

spectrum_4 = wave_4.make_spectrum()
spectrum_4_2 = wave_4_2.make_spectrum()
spectrum_4.plot_power(alpha=0.5)
spectrum_4_2.plot_power(alpha=0.5)
decorate(xlabel='Frequency (Hz)',
         ylabel='Power',
         **loglog)
plt.savefig(filePath + "4.radiation.spectrum" + fileExtension)
plt.close()

"""
Exercise 4.5

The algorithm in this chapter for generating pink noise is
conceptually simple but computationally expensive. There are
more efficient alternatives, like the Voss-McCartney algorithm.
Research this method, implement it, compute the spectrum of
the result, and confirm that it has the desired relationship
between power and frequency.
"""


def voss_mcccartney_pink_noise(n_rows, n_cols=16):
    array = np.empty((n_rows, n_cols))
    array.fill(np.nan)
    array[0, :] = np.random.random(n_cols)
    array[:, 0] = np.random.random(n_rows)

    cols = np.random.geometric(0.5, n_rows)
    cols[cols >= n_cols] = 0
    rows = np.random.randint(n_rows, size=n_rows)
    array[rows, cols] = np.random.random(n_rows)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)

    # noinspection PyArgumentList
    return df.sum(axis=1).values


wave = Wave(voss_mcccartney_pink_noise(2 ** 10))
wave.unbias()
wave.normalize()

wave.plot()
plt.savefig(filePath + "5.voss.wave" + fileExtension)
plt.close()

wave.make_audio()

spectrum = wave.make_spectrum()
spectrum.hs[0] = 0
spectrum.plot_power()
decorate(xlabel='Frequency (Hz)',
         **loglog)
plt.savefig(filePath + "5.voss.spectrum" + fileExtension)
plt.close()

print(spectrum.estimate_slope().slope)
# -0.9961332129929098

seg_length = 2 ** 16
iterations_amount = 100
wave_5 = Wave(voss_mcccartney_pink_noise(seg_length * iterations_amount))

spectrum = bartlett_spectrum(wave_5, seg_length=seg_length, win_flag=False)
spectrum.hs[0] = 0

spectrum.plot_power()
decorate(xlabel='Frequency (Hz)',
         **loglog)
plt.savefig(filePath + "5.voss.bartlett" + fileExtension)
plt.close()

print(spectrum.estimate_slope().slope)
# -1.0009495578549672
