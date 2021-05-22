import matplotlib.pyplot as plt

from code.thinkdsp import read_wave, decorate

filePath = "../lab_captions/lab10/"
fileExtension = ".pdf"

"""
Exercise 1

In this chapter I describe convolution as the sum of shifted,
scaled copies of a signal.  Strictly speaking, this operation is
*linear* convolution, which does not assume that the signal
is periodic.

But when we multiply the
DFT of the signal by the transfer function, that operation corresponds
to *circular* convolution, which assumes that the signal is
periodic.  As a result, you might notice that the output contains
an extra note at the beginning, which wraps around from the end.

Fortunately, there is a standard solution to this problem.  If you
add enough zeros to the end of the signal before computing the DFT,
you can avoid wrap-around and compute a linear convolution.

Modify the example in `chap10soln.ipynb` and confirm that zero-padding
eliminates the extra note at the beginning of the output.
"""

response_1 = read_wave('../180960__kleeb__gunshot.wav')

start_1 = 0.12
response_1 = response_1.segment(start=start_1)
response_1.shift(-start_1)

response_1.truncate(2 ** 16)
response_1.zero_pad(2 ** 17)

response_1.normalize()
response_1.plot()
decorate(xlabel='Time (s)')
plt.savefig(filePath + "1.gunshot.graph" + fileExtension)
plt.close()

transfer_1 = response_1.make_spectrum()
transfer_1.plot()
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')
plt.savefig(filePath + "1.gunshot.spectrum" + fileExtension)
plt.close()

violin_1 = read_wave('../92002__jcveliz__violin-origional.wav')

violin_1 = violin_1.segment(start=start_1)
violin_1.shift(-start_1)

violin_1.truncate(2 ** 16)
violin_1.zero_pad(2 ** 17)

violin_1.normalize()
violin_1.plot()
decorate(xlabel='Time (s)')
plt.savefig(filePath + "1.violin.graph" + fileExtension)
plt.close()

spectrum_1 = violin_1.make_spectrum()

output_1 = (spectrum_1 * transfer_1).make_wave()
output_1.normalize()

output_1.make_audio()

output_1.plot()
plt.savefig(filePath + "1.violin.transformed.fixed" + fileExtension)
plt.close()

"""
## Exercise 2

The Open AIR library provides a "centralized... on-line resource for
anyone interested in auralization and acoustical impulse response
data" (https://www.openairlib.net).  Browse their collection
of impulse response data and download one that sounds interesting.
Find a short recording that has the same sample rate as the impulse
response you downloaded.

Simulate the sound of your recording in the space where the impulse
response was measured, computed two way: by convolving the recording
with the impulse response and by computing the filter that corresponds
to the impulse response and multiplying by the DFT of the recording.
"""

response_woman_club_2 = read_wave('../spokane_womans_club_ir.wav')

duration_2 = 5
response_woman_club_2 = response_woman_club_2.segment(duration=duration_2)

response_woman_club_2.normalize()
response_woman_club_2.plot()
decorate(xlabel='Time (s)')
plt.savefig(filePath + "2.woman.club.graph" + fileExtension)
plt.close()

response_woman_club_2.make_audio()

transfer_2 = response_woman_club_2.make_spectrum()
transfer_2.plot()
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')
plt.savefig(filePath + "2.woman.club.spectrum" + fileExtension)
plt.close()

transfer_2.plot()
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude',
         xscale='log', yscale='log')
plt.savefig(filePath + "2.woman.club.spectrum.log" + fileExtension)
plt.close()

wave_piano_2 = read_wave('../32158__zin__piano-2-140bpm.wav')

wave_piano_2 = wave_piano_2.segment(duration=duration_2)

wave_piano_2.truncate(len(response_woman_club_2))
wave_piano_2.normalize()

wave_piano_2.make_audio()

spectrum_2 = wave_piano_2.make_spectrum()

print(len(spectrum_2.hs), len(transfer_2.hs))
# 110251 110251

output_2 = (spectrum_2 * transfer_2).make_wave()
output_2.normalize()

wave_piano_2.plot()
plt.savefig(filePath + "2.piano" + fileExtension)
plt.close()
output_2.plot()
plt.savefig(filePath + "2.piano.transformed" + fileExtension)
plt.close()

output_2.make_audio()

convolved_2 = wave_piano_2.convolve(response_woman_club_2)
convolved_2.normalize()
convolved_2.make_audio()

convolved_2.plot()
plt.savefig(filePath + "2.piano.convolved" + fileExtension)
plt.close()
