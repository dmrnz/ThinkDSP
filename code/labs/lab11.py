import numpy as np
import matplotlib.pyplot as plt
from code.thinkdsp import decorate
from code.thinkdsp import Spectrum
from code.thinkdsp import read_wave, Wave

filePath = "../lab_captions/lab11/"
fileExtension = ".pdf"

"""
Exercise 11.1 
The code in this chapter is in chap11.ipynb. Read through it
and listen to the examples.
"""

# Все примеры были запущены успешно

"""
Exercise 11.2 
Chris “Monty” Montgomery has an excellent video called
“D/A and A/D | Digital Show and Tell”; it demonstrates the Sampling
Theorem in action, and presents lots of other excellent information about
sampling. Watch it at https://www.youtube.com/watch?v=cIQ9IXSUzuM.
"""

"""В видео Криса Монтгомери демонстрируется теорема о выборках на практике
и представляется множество другой информации о выборках. В частности было объяснено,
почему аналоговый звук в пределах человеческого слуха может воспроизводиться с идеальной точностью
с использованием 16-битного цифрового сигнала 44,1 кГц"""

"""
Exercise 11.3
As we have seen, if you sample a signal at too low a framerate,
frequencies above the folding frequency get aliased. Once that happens, it
is no longer possible to filter out these components, because they are indistinguishable
from lower frequencies.
It is a good idea to filter out these frequencies before sampling; a low-pass
filter used for this purpose is called an anti-aliasing filter.
Returning to the drum solo example, apply a low-pass filter before sampling,
then apply the low-pass filter again to remove the spectral copies
introduced by sampling. The result should be identical to the filtered signal.
"""

# Скопируем сигнал и построим для него график
wave_3 = read_wave('../263868__kevcio__amen-break-a-160-bpm.wav')
wave_3.normalize()
wave_3.plot()
plt.savefig(filePath + "3.wave.plot" + fileExtension)
plt.close()

# Сделаем запись для прослушивания
wave_3.make_audio()

# Теперь получим спектр для сигнала
spectrum_3 = wave_3.make_spectrum(full=True)
spectrum_3.plot()
plt.savefig(filePath + "3.wave.spectrum.plot" + fileExtension)
plt.close()

# Уменьшаем частоту дискретизации в 3 раза
framerate_3 = wave_3.framerate / 3
cutoff_3 = framerate_3 / 2 - 1

# Удаляем частоты выше новой частоты свертки
spectrum_3.low_pass(cutoff_3)
spectrum_3.plot()
plt.savefig(filePath + "3.wave.filtered.spectrum.plot" + fileExtension)
plt.close()

# Создаем отфильтрованный звук для прослушивания
filtered_signal_3 = spectrum_3.make_wave()
filtered_signal_3.make_audio()

# Напишем функцию для имитации выборки


def sample_imitation(wave):
    ys = np.zeros(len(wave))
    ys[::3] = np.real(wave.ys[::3])
    return Wave(ys, framerate=wave.framerate)


# Используем созданную функцию
sampled_signal_3 = sample_imitation(filtered_signal_3)
sampled_signal_3.make_audio()

# Создадим график для спектра полученного сигнала
sampled_signal_spectrum_3 = sampled_signal_3.make_spectrum(full=True)
sampled_signal_spectrum_3.plot()
plt.savefig(filePath + "3.wave.sampled.spectrum.plot" + fileExtension)
plt.close()

# Уберем спектральные копии
sampled_signal_spectrum_3.low_pass(cutoff_3)
sampled_signal_spectrum_3.plot()
plt.savefig(filePath + "3.wave.sampled.spectrum.without.spectral.copies" + fileExtension)
plt.close()

# Масштабируем полученный спектр и сравним с отфильтрованным
sampled_signal_spectrum_3.scale(3)
sampled_signal_spectrum_3.plot(color='gray')
spectrum_3.plot()
plt.savefig(filePath + "3.spectrum.vs.sampled.spectrum" + fileExtension)
plt.close()

# Посчитаем максимальную разницу между ними с помощью max_diff
spectrum_3.max_diff(sampled_signal_spectrum_3)

# Получим из полученного спектра сигнал
interpolated_signal_3 = sampled_signal_spectrum_3.make_wave()
interpolated_signal_3.make_audio()

# И построим график для нового сигнала
interpolated_signal_3.plot()
plt.savefig(filePath + "3.interpolated.signal" + fileExtension)
plt.close()

# Как можно видеть, новый сигнал почти не отличается от отфильтрованного
