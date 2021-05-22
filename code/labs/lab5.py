from code.thinkdsp import Wave, decorate, read_wave, TriangleSignal
from code.autocorr_fixed import autocorr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filePath = "../lab_captions/lab5/"
fileExtension = ".pdf"
"""
Упражнение 5.1
Блокнот Jupiter этой главы chap05.ipynb содержит приложение, в котором можно вычислить автокорреляции для различных lag.
Оцените высоты тона вокального чирпа для нескольких времен начала сегмента.
"""

"""
Exercise 5.1 
The Jupyter notebook for this chapter, chap05.ipynb, includes
an interaction that lets you compute autocorrelations for different lags. Use
this interaction to estimate the pitch of the vocal chirp for a few different
start times.
"""

# Скопируем запись вокального чирпа и выделим из нее один сегмент небольшой длительности
wave_1 = read_wave('../28042__bcjordan__voicedownbew.wav')
wave_1.normalize()
segment_1 = wave_1.segment(duration=0.01)

# Далее воспользуемся автокорреляцией и построим получившийся график зависимости
lags_1, correlation_1 = autocorr(segment_1)
plt.plot(lags_1, correlation_1)
decorate(xlabel='Lags', ylabel='Correlation')
plt.savefig(filePath + "1.lags.correlation.plot.for.first.segment" + fileExtension)
plt.close()

# Для уточнения значения пика необходимо воспользоваться функцией argmax
peak_1 = np.array(correlation_1[50:100]).argmax() + 50
print(peak_1)

# Вычисляем частоту полученного пика
frequency_1 = segment_1.framerate / peak_1
print(frequency_1)

# Повторяем предыдущие действия с автокорреляцией для другого фрагмента записи
segment_1 = wave_1.segment(start=1, duration=0.01)
lags_1, correlation_1 = autocorr(segment_1)
plt.plot(lags_1, correlation_1)
decorate(xlabel='Lags', ylabel='Correlation')
plt.savefig(filePath + "1.lags.correlation.plot.for.second.segment" + fileExtension)
plt.close()

# Уточняем пик
peak_1 = np.array(correlation_1[100:150]).argmax() + 100
print(peak_1)

# Вычисляем частоту для полученного значения
frequency_1 = segment_1.framerate / peak_1
print(frequency_1)

# Приходим к выводу что при увеличении значения для начала сегмента записи уменьшается частота для пиков

"""
Упражнение 5.2
Пример кода в chap05.ipynb показывает, как использовать автокорреляцию 
для оценки основной частоты периодического сигнала.
Инкапсулируйте этот код в функцию, названную estimate_fundamental, и используйте ее для отслеживания высоты тона
записанного звука.
Проверьте, насколько хорошо она работает, накладывая оценки высоты тона на спектрограмму записи.
"""

"""
Exercise 5.2
The example code in chap05.ipynb shows how to use autocorrelation
to estimate the fundamental frequency of a periodic signal. Encapsulate
this code in a function called estimate_fundamental, and use it to
track the pitch of a recorded sound.
"""

"""Поскольку задача определения границ для уточнения и определения пика достаточно трудоемкая, 
то эти границы будут указываться вручную при вызове функции"""


def estimate_fundamental(segment, low, high):
    lags, corr = autocorr(segment)
    lag = np.array(corr[low:high]).argmax() + low
    frequency = segment.framerate / lag
    return frequency


# Проверим функцию на ранее расчитанных данных
wave_2 = read_wave('../28042__bcjordan__voicedownbew.wav')
wave_2.normalize()
segment_2 = wave_2.segment(duration=0.01)
frequency_2 = estimate_fundamental(segment_2, 50, 100)
print(frequency_2)

# Построим спектрограмму записи, ограничим высоту графика для лучшего изображения
spectrogram_2 = wave_2.make_spectrogram(2048)
spectrogram_2.plot(high=1000)
decorate(xlabel='Time', ylabel='Frequency')
plt.savefig(filePath + "2.wave.spectrogram" + fileExtension)
plt.close()

# Сформируем списки для исходных и конечных данных
start_2 = 0.0
end_2 = 1.4
step_2 = (end_2 - start_2) / 100
values_2 = np.arange(start_2, end_2, step_2)
timestamps_2 = []
frequency_list_2 = []

# Заполним списки для выходных данных
for value in values_2:
    timestamps_2.append(value + step_2 / 2)
    segment = wave_2.segment(start=value, duration=0.01)
    frequency = estimate_fundamental(segment, 70, 150)
    frequency_list_2.append(frequency)

# Построим график для получившихся значений
spectrogram_2 = wave_2.make_spectrogram(2048)
spectrogram_2.plot(high=1000)
plt.plot(timestamps_2, frequency_list_2, color='white')
decorate(xlabel='Time', ylabel='Frequency')
plt.savefig(filePath + "2.wave.spectrogram.vs.peak.values" + fileExtension)
plt.close()

"""
Упражнение 5.3
Для упражнения из предыдущей главы нужны были исторические цены BitCoins, 
и надо было оценить спектр мощности изменения цен.
Используя те же данные, вычислите автокорреляции цен в платежной системе Bitcoin.
Быстро ли спадает автокорреляционная функция? Есть ли признаки периодичности процесса?
"""

"""
Exercise 5.3 
If you did the exercises in the previous chapter, you downloaded
the historical price of BitCoins and estimated the power spectrum
of the price changes. Using the same data, compute the autocorrelation of
BitCoin prices. Does the autocorrelation function drop off quickly? Is there
evidence of periodic behavior?
"""

# Сначала прочитаем полученные данные о Bitcoin и построим по ним график
data_frame_3 = pd.read_csv('../BTC_USD_2013-10-01_2021-05-08-CoinDesk.csv',
                           parse_dates=[0])
ys_3 = data_frame_3['Closing Price (USD)']
timestamps_3 = data_frame_3.index
wave_3 = Wave(ys_3, timestamps_3, framerate=1)
wave_3.plot()
decorate(xlabel='Days', ylabel='Price')
plt.savefig(filePath + "3.bitcoin.price.wave" + fileExtension)
plt.close()

# Затем вычислим автокоррекцию
lags_3, correlation_3 = autocorr(wave_3)
plt.plot(lags_3, correlation_3)
decorate(xlabel='Lags', ylabel='Correlation')
plt.savefig(filePath + "3.bitcoin.wave.autocorrection" + fileExtension)
plt.close()

# Как можно видеть, автокорреляция спадает не очень быстро, что указывает на то что перед нами розовый шум

"""
Упражнение 5.4
В репозитории этой книги есть блокнот Jupyter под названием saxophone.ipynb,
в котором исследуются автокорреляция, восприятие высоты тона и явление, называемое подавленная основная.
Прочтите этот блокнот и запустите примеры. Выберите другой сегмент записи и вновь поработайте с примерами.
У Ви Харт есть отличное видео под названием "Так что же там с шумами? (Наука и математика звука, частота и высота тона)"
Она демонстрирует феномен подавленной основной и объясняет, как воспринимается высота тона
(по крайней мере, насколько об этом известно).
См. https://www.youtube.com/watch?v=i_0DXxNeaQ0.
"""

"""
Exercise 5.4 
In the repository for this book you will find a Jupyter notebook
called saxophone.ipynb that explores autocorrelation, pitch perception,
and a phenomenon called the missing fundamental. Read through
this notebook and run the examples. Try selecting a different segment of
the recording and running the examples again.
Vi Hart has an excellent video called “What is up with Noises? (The Science
and Mathematics of Sound, Frequency, and Pitch)”; it demonstrates
the missing fundamental phenomenon and explains how pitch perception
works (at least, to the degree that we know). Watch it at https://www.youtube.com/watch?v=i_0DXxNeaQ0.
"""

# По примеру saxophone.ipynb сначала скопируем запись
wave_4 = read_wave('../100475__iluppai__saxophone-weep.wav')
wave_4.normalize()
wave_4.make_audio()

# Затем посмотрим гармоническую структуру записи
gram_4 = wave_4.make_spectrogram(seg_length=1024)
gram_4.plot(high=3000)
decorate(xlabel='Time (s)', ylabel='Frequency (Hz)')
plt.savefig(filePath + "4.harmonic.structure" + fileExtension)
plt.close()

# После этого выберем другой промежуток записи для рассмотрения

start_4 = 1.0
duration_4 = 0.5
segment_4 = wave_4.segment(start=start_4, duration=duration_4)
segment_4.make_audio()

# Построим спектр этого промежутка
spectrum_4 = segment_4.make_spectrum()
spectrum_4.plot(high=3000)
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')
plt.savefig(filePath + "4.segment.spectrum" + fileExtension)
plt.close()

# Затем рассмотрим значения пиков
print(spectrum_4.peaks()[:10])

# Основные частоты тут 1246 Гц, 416 Гц, 830Гц

"""Высота 416 Гц будет восприниматься как основная, хоть она и не доминирующая, 
для примера возьмем треугольную волну с такой же частотой"""

TriangleSignal(freq=416).make_wave(duration=0.5).make_audio()
segment_4.make_audio()

# Чтобы понять, почему мы воспринимаем основную высоту, не являющуюся доминирующей, используем автокорреляцию

lags_4, corrs_4 = autocorr(segment_4)
plt.plot(corrs_4[:200])
decorate(xlabel='Lag', ylabel='Correlation', ylim=[-1.05, 1.05])
plt.savefig(filePath + "4.segment.autocorrelation" + fileExtension)
plt.close()

# Первый пик находится между значениями 100 и 125, рассчитаем его частоту
estimate_fundamental(segment_4, 100, 125)

"""Это значение соответствует 416Гц, что значит что мы воспринимаем наивысший пик корреляционной функции,
а не самый высокий компонент спектра"""

# Теперь удалим основную частоту
spectrum_4_2 = segment_4.make_spectrum()
spectrum_4_2.high_pass(450)
spectrum_4_2.plot(high=3000)
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')
plt.savefig(filePath + "4.segment.without.fundamental" + fileExtension)
plt.close()

# Прослушаем полученную запись
segment_4_2 = spectrum_4_2.make_wave()
segment_4_2.make_audio()

"""Воспринимаемая высота звука до сих пор на 416 Гц, хотя мы убрали пик на этой частоте,
это явление называется "подавленная основная" """

# Обратимся к автокорреляции для того, чтобы понять, что происходит
lags_4_2, corrs_4_2 = autocorr(segment_4_2)
plt.plot(corrs_4_2[:200])
decorate(xlabel='Lag', ylabel='Correlation', ylim=[-1.05, 1.05])
plt.savefig(filePath + "4.segment.without.fundamental.autocorrelation" + fileExtension)
plt.close()

# Наивысший пик все еще соответствует 416Гц, помимо этого у нас есть и другие гармоники, которые надо рассчитать
estimate_fundamental(segment_4_2, 25, 50)
estimate_fundamental(segment_4_2, 50, 90)

# Если избавимся от гармоник выше 1200 Гц то эффект пропадет
spectrum_4_3 = segment_4_2.make_spectrum()
spectrum_4_3.high_pass(450)
spectrum_4_3.low_pass(1200)
spectrum_4_3.plot(high=3000)
decorate(xlabel='Frequency (Hz)', ylabel='Amplitude')
plt.savefig(filePath + "4.segment.without.fundamental.and.high.harmonics" + fileExtension)
plt.close()

segment_4_3 = spectrum_4_3.make_wave()
segment_4_3.make_audio()

lags_4_3, corrs_4_3 = autocorr(segment_4_3)
plt.plot(corrs_4_3[:200])
decorate(xlabel='Lag', ylabel='Correlation', ylim=[-1.05, 1.05])
plt.savefig(filePath + "4.segment.without.fundamental.and.high.harmonics.autocorrelation" + fileExtension)
plt.close()

# Теперь основная частота находится на 830 Гц
TriangleSignal(freq=830).make_wave(duration=0.5).make_audio()
