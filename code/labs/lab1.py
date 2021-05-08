from code.thinkdsp import read_wave, decorate, SinSignal, CosSignal
import matplotlib.pyplot as plt

filePath = "C:/Users/dimac/PycharmProjects/ThinkDSP/code/lab_captions/lab1/"

"""
Exercise 1.1

If you have Jupyter, load chap01.ipynb, read through it, and
run the examples. You can also view this notebook at https://tinyurl.com/thinkdsp01.
"""
wave_ex1 = read_wave('../92002__jcveliz__violin-origional.wav')
segment2_ex1 = wave_ex1.segment(0, 5)
spectrum_ex1 = segment2_ex1.make_spectrum()
spectrum_ex1.plot(color='0.7')
spectrum_ex1.low_pass(1000)
spectrum_ex1.plot(color='#045a8d')

plt.savefig(filePath + "1.last.ex.chap01.pdf")
plt.close()

"""
Exercise 1.2

Go to https://freesound.org and download a sound sample
that includes music, speech, or other sounds that have a well-defined pitch.
Select a roughly half-second segment where the pitch is constant. Compute
and plot the spectrum of the segment you selected. What connection can you
make between the timbre of the sound and the harmonic structure you see in
the spectrum?

Use high_pass, low_pass, and band_stop to filter out some of the harmonics.
Then convert the spectrum back to a wave and listen to it. How does the sound
relate to the changes you made in the spectrum?
"""
wave_1_2 = read_wave('../32158__zin__piano-2-140bpm.wav')
wave_1_2.normalize()
wave_1_2.make_audio()
wave_1_2.plot()
plt.savefig(filePath + "2.full.piano.pdf")
plt.close()

segment_1_2 = wave_1_2.segment(start=0, duration=0.22)
spectrum_1_2 = segment_1_2.make_spectrum()

spectrum_1_2.plot(high=4500)
plt.savefig(filePath + "2.piano.spectrum.pdf")
plt.close()

print(spectrum_1_2.peaks()[:10])

spectrum_1_2.plot(high=4500, color='0.7')
spectrum_1_2.low_pass(300)
spectrum_1_2.high_pass(220)
spectrum_1_2.plot(high=400, color='#045a8d')
decorate(xlabel='Frequency (Hz)')

spectrum_1_2.make_wave().make_audio()

plt.savefig(filePath + "2.piano.spectrum.filtered.pdf")
plt.close()

"""
Exercise 1.3

Synthesize a compound signal by creating SinSignal and CosSig-
nal objects and adding them up. Evaluate the signal to get a Wave, and listen
to it. Compute its Spectrum and plot it. What happens if you add frequency
components that are not multiples of the fundamental?
"""
signal_1_3 = (CosSignal(freq=1000, amp=1.0) +
              SinSignal(freq=600, amp=2.0) +
              CosSignal(freq=400, amp=0.33) +
              CosSignal(freq=2000, amp=1.5))
signal_1_3.plot()
plt.savefig(filePath + "3.compound.signal.pdf")
plt.close()

wave_1_3 = signal_1_3.make_wave(duration=3)
wave_1_3.apodize()
wave_1_3.make_audio()

spectrum_1_3 = wave_1_3.make_spectrum()
spectrum_1_3.plot(high=2100)
plt.savefig(filePath + "3.compound.spectrum.pdf")
plt.close()

signal_1_3 += SinSignal(freq=789)
signal_1_3.make_wave().make_audio()

"""
Exercise 1.4

Write a function called stretch that takes a Wave and a stretch
factor and speeds up or slows down the wave by modifying ts and framerate.
Hint: it should only take two lines of code.
"""

wave_1_4 = read_wave('../32158__zin__piano-2-140bpm.wav')
wave_1_4.normalize()
wave_1_4.make_audio()
wave_1_4.plot()
plt.savefig(filePath + "4.normal.pdf")
plt.close()


def stretch(wave, stretch_factor):
    wave.ts /= stretch_factor
    wave.framerate *= stretch_factor


stretch(wave_1_4, 2)
wave_1_4.make_audio()

wave_1_4.plot()
plt.savefig(filePath + "4.stretch.pdf")
plt.close()
