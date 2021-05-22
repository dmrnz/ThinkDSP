import numpy as np
import matplotlib.pyplot as plt
from code.thinkdsp import SinSignal, decorate, read_wave, normalize, unbias, Chirp

filePath = "../lab_captions/lab3/"
fileExtension = ".pdf"

"""
Exercise 3.1 Run and listen to the examples in chap03.ipynb, which is in
the repository for this book, and also available at http://tinyurl.com/
thinkdsp03.
In the leakage example, try replacing the Hamming window with one of
the other windows provided by NumPy, and see what effect they have on
leakage. See http://docs.scipy.org/doc/numpy/reference/routines.
window.html
"""

signal_3_1_1 = SinSignal(freq=440)
duration_3_1_1 = signal_3_1_1.period * 30.25
wave_3_1_1 = signal_3_1_1.make_wave(duration_3_1_1)
spectrum_3_1_1 = wave_3_1_1.make_spectrum()

spectrum_3_1_1.plot(high=880)
decorate(xlabel='Frequency (Hz)')

plt.savefig(filePath + "1.leakage.example" + fileExtension)
plt.close()

window_type_names = [np.bartlett, np.blackman, np.hamming, np.hanning]

for window_type in window_type_names:
    wave_3_1_2 = signal_3_1_1.make_wave(duration_3_1_1)
    wave_3_1_2.ys *= window_type(len(wave_3_1_2.ys))

    spectrum_3_1_2 = wave_3_1_2.make_spectrum()
    spectrum_3_1_2.plot(high=880, label=window_type.__name__)

decorate(xlabel='Frequency (Hz)')

plt.savefig(filePath + "1.hamm.window.replacing" + fileExtension)
plt.close()

"""
Exercise 3.2 Write a class called SawtoothChirp that extends Chirp and
overrides evaluate to generate a sawtooth waveform with frequency that
increases (or decreases) linearly.
Hint: combine the evaluate functions from Chirp and SawtoothSignal.
Draw a sketch of what you think the spectrogram of this signal looks like,
and then plot it. The effect of aliasing should be visually apparent, and if
you listen carefully, you can hear it.
"""


class SawtoothChirp(Chirp):
    """Represents a sawtooth signal with varying frequency."""

    def evaluate(self, ts):
        """Helper function that evaluates the signal.

        ts: float array of times
        """
        freqs = np.linspace(self.start, self.end, len(ts))
        dts = np.diff(ts, prepend=0)
        dphis = 2 * np.pi * freqs * dts
        phases = np.cumsum(dphis)
        cycles = phases / 2 / np.pi
        frac, _ = np.modf(cycles)
        ys = normalize(unbias(frac), self.amp)
        return ys


signal_3_2 = SawtoothChirp(start=220, end=880)
wave_3_2 = signal_3_2.make_wave(duration=1, framerate=4000)
wave_3_2.apodize()
wave_3_2.make_audio()

spectrum_3_2 = wave_3_2.make_spectrogram(256)
spectrum_3_2.plot()
decorate(xlabel='Time (s)', ylabel='Frequency (Hz)')

plt.savefig(filePath + "2.sawtooth.chirp" + fileExtension)
plt.close()
read_wave('../tos-redalert.wav').make_audio()

"""
Exercise 3.3 Make a sawtooth chirp that sweeps from 2500 to 3000 Hz, then
use it to make a wave with duration 1 s and framerate 20 kHz. Draw a
sketch of what you think the spectrum will look like. Then plot the spectrum
and see if you got it right.
"""

signal_3_3 = SawtoothChirp(start=2500, end=3000)
wave_3_3 = signal_3_3.make_wave(duration=1, framerate=20000)
wave_3_3.make_audio()

wave_3_3.make_spectrum().plot()
decorate(xlabel='Frequency (Hz)')

plt.savefig(filePath + "3.stc.2500-3000Hz" + fileExtension)
plt.close()

"""
Exercise 3.4 In musical terminology, a “glissando” is a note that slides from
one pitch to another, so it is similar to a chirp.
Find or make a recording of a glissando and plot a spectrogram of the first
few seconds. 
"""

wave_3_4 = read_wave('../153623__carlos-vaquero__violin-tenuto-non-vibrato-glissando-1.wav')
wave_3_4.make_audio()

wave_3_4.make_spectrogram(512).plot(high=5000)
decorate(xlabel='Time (s)', ylabel='Frequency (Hz)')

plt.savefig(filePath + "4.glissando" + fileExtension)
plt.close()

"""
Exercise 3.5 A trombone player can play a glissando by extending the trombone
slide while blowing continuously. As the slide extends, the total length
of the tube gets longer, and the resulting pitch is inversely proportional to
length.
Assuming that the player moves the slide at a constant speed, how does
frequency vary with time?
Write a class called TromboneGliss that extends Chirp and provides
evaluate. Make a wave that simulates a trombone glissando from C3 up
to F3 and back down to C3. C3 is 262 Hz; F3 is 349 Hz.
Plot a spectrogram of the resulting wave. Is a trombone glissando more like
a linear or exponential chirp?
"""


class TromboneGliss(Chirp):

    def evaluate(self, ts):
        l1, l2 = 1.0 / self.start, 1.0 / self.end
        lengths = np.linspace(l1, l2, len(ts))
        freqs = 1 / lengths

        dts = np.diff(ts, prepend=0)
        dphis = 2 * np.pi * freqs * dts
        phases = np.cumsum(dphis)
        ys = self.amp * np.cos(phases)
        return ys


low = 262
high = 349
signal_3_5_1 = TromboneGliss(high, low)
wave_3_5_1 = signal_3_5_1.make_wave(duration=1)
wave_3_5_1.apodize()
wave_3_5_1.make_audio()

signal_3_5_2 = TromboneGliss(low, high)
wave_3_5_2 = signal_3_5_2.make_wave(duration=1)
wave_3_5_2.apodize()
wave_3_5_2.make_audio()

wave_3_5 = wave_3_5_1 | wave_3_5_2
wave_3_5.make_audio()

spectrum_3_5 = wave_3_5.make_spectrogram(1024)
spectrum_3_5.plot(high=1000)
decorate(xlabel='Time (s)', ylabel='Frequency (Hz)')

plt.savefig(filePath + "5.tromb.gliss" + fileExtension)
plt.close()

"""
Exercise 3.6 Make or find a recording of a series of vowel sounds and look
at the spectrogram. Can you identify different vowels?
"""
wave_3_6 = read_wave('../523141__bryanr17__mjoven.wav')
wave_3_6.make_audio()

wave_3_6.make_spectrogram(1024).plot(high=1000)
decorate(xlabel='Time (s)', ylabel='Frequency (Hz)')
plt.savefig(filePath + "6.vowel.spectrum" + fileExtension)
plt.close()

high = 1000

a_segment = wave_3_6.segment(start=0, duration=0.25)
a_segment.make_spectrum().plot(high=high)
plt.savefig(filePath + "6.a.segment" + fileExtension)
plt.close()

e_segment = wave_3_6.segment(start=0.9, duration=0.25)
e_segment.make_spectrum().plot(high=high)
decorate(xlabel='Frequency (Hz)')
plt.savefig(filePath + "6.e.segment" + fileExtension)
plt.close()

i_segment = wave_3_6.segment(start=1.8, duration=0.25)
i_segment.make_spectrum().plot(high=high)
decorate(xlabel='Frequency (Hz)')
plt.savefig(filePath + "6.i.segment" + fileExtension)
plt.close()

o_segment = wave_3_6.segment(start=2.8, duration=0.25)
o_segment.make_spectrum().plot(high=high)
decorate(xlabel='Frequency (Hz)')
plt.savefig(filePath + "6.o.segment" + fileExtension)
plt.close()

u_segment = wave_3_6.segment(start=3.6, duration=0.25)
u_segment.make_spectrum().plot(high=high)
decorate(xlabel='Frequency (Hz)')
plt.savefig(filePath + "6.u.segment" + fileExtension)
plt.close()


