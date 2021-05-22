import math
import numpy as np
import matplotlib.pyplot as plt
from code.thinkdsp import normalize, unbias, decorate, Signal, TriangleSignal, SquareSignal, SinSignal

filePath = "../lab_captions/lab2/"
fileExtension = ".pdf"
"""
Exercise 2.1 
If you use Jupyter, load chap02.ipynb and try out the examples.
You can also view the notebook at https://tinyurl.com/thinkdsp02.
"""

"""
Упражнение 2.2

Пилообразный сигнал линейно нарастает от -1 до 1, а затем резко падает до -1 и повторяется.
См. https://en.wikipedia.org/wiki/Sawtooth_wave

Напишите класс, называемый SawtoothSignal, расширяющий класс Signal
и предоставляющий метод evaluate для оценки пилообразного сигнала.

Вычислите спектр пилообразного сигнала.
Как соотносится его гармоническая структура с треугольным и прямоугольным сигналами?
"""

"""
Exercise 2.2 
A sawtooth signal has a waveform that ramps up linearly from
-1 to 1, then drops to -1 and repeats. See https://en.wikipedia.org/wiki/
Sawtooth_wave
Write a class called SawtoothSignal that extends Signal and provides evaluate to evaluate a sawtooth signal.
Compute the spectrum of a sawtooth wave. How does the harmonic structure compare to triangle and square waves? 
"""

# Исходный код для класса SawtoothSignal и метода evaluate


class SawtoothSignal(Signal):

    def __init__(self, freq=440, amp=1.0, offset=0, func=np.sin):
        self.freq = freq
        self.amp = amp
        self.offset = offset
        self.func = func

    def period(self):
        return 1.0 / self.freq

    def evaluate(self, ts):
        ts = np.asarray(ts)
        cycles = self.freq * ts + self.offset / math.pi / 2
        frac, _ = np.modf(cycles)
        ys = normalize(unbias(frac), self.amp)
        return ys


# Создание пилообразного сигнала и построение его графика
sawtooth_signal_2 = SawtoothSignal()
sawtooth_wave_2 = sawtooth_signal_2.make_wave(sawtooth_signal_2.period() * 5, framerate=40000)
sawtooth_wave_2.plot()
plt.savefig(filePath + "2.sawtooth.signal" + fileExtension)
plt.close()
sawtooth_wave_2.make_audio()

# Создание пилообразного сигнала и построение графика для его спектра
sawtooth_wave_2 = sawtooth_signal_2.make_wave(0.5, framerate=40000)
sawtooth_spectrum_2 = sawtooth_wave_2.make_spectrum()
sawtooth_spectrum_2.plot()
decorate(xlabel='Frequency (Hz)')
plt.savefig(filePath + "2.sawtooth.spectrum" + fileExtension)
plt.close()

# Создание треугольного сигнала, построение графика для спектров пилообразного и треугольного сигналов
sawtooth_spectrum_2.plot(color='gray')
triangle_signal_2 = TriangleSignal(amp=0.79).make_wave(duration=0.5, framerate=40000)
triangle_signal_2.make_spectrum().plot()
decorate(xlabel='Frequency (Hz)')
plt.savefig(filePath + "2.sawtooth.vs.triangle.spectrum" + fileExtension)
plt.close()

# Создание прямоугольного сигнала, построение графика для спектров пилообразного и прямоугольного сигналов
sawtooth_spectrum_2.plot(color='gray')
square_signal_2 = SquareSignal(amp=0.5).make_wave(duration=0.5, framerate=40000)
square_signal_2.make_spectrum().plot()
decorate(xlabel='Frequency (Hz)')
plt.savefig(filePath + "2.sawtooth.vs.square.spectrum" + fileExtension)
plt.close()

"""
Упражнение 2.3
Создайте прямоугольный сигнал 1100 Гц и вычислите wave с выборками 10 000 кадров в секунду.
Постройте спектр и убедитесь, что большинство гармоник "завернуты" из-за биений.
Слышны ли последствия этого при проигрывании?
"""

"""
Exercise 2.3 
Make a square signal at 1100 Hz and make a wave that samples
it at 10000 frames per second. If you plot the spectrum, you can see that
most of the harmonics are aliased. When you listen to the wave, can you
hear the aliased harmonics? 
"""

# Создание прямоугольного сигнала 1100 Гц и построение графика для его спектра
square_signal_3 = SquareSignal(freq=1100).make_wave(framerate=10000)
square_signal_3.make_spectrum().plot()
decorate(xlabel='Frequency (Hz)')
plt.savefig(filePath + "3.square.signal.spectrum" + fileExtension)
plt.close()

"""Создание записи прямоугольного сигнала для прослушивания и синусоидального сигнала 200 Гц для проверки утверждения, 
что основной тон, который мы воспринимаем, является гармоникой биения на частоте 200 Гц"""
square_signal_3.make_audio()
SinSignal(200).make_wave(duration=0.5, framerate=10000).make_audio()

"""
Упражнение 2.4
Возьмите объект Spectrum и распечатайте несколько первых значений spectrum.fs.
Убедитесь, что они начинаются с нуля, то есть Spectrum.hs[0] - амплитуда компоненты с частотой 0.
Но что это значит? Проведите такой эксперимент:
    1. Создайте треугольный сигнал с частотой 440 Гц и wave длительностью 0.01 секунды. Распечатайте сигнал.
    2. Создайте объект Spectrum и распечатайте Spectrum.hs[0]. Каковы амплитуда и фаза этого компонента?
    3. Установите Spectrum.hs[0] = 100. Как эта операция повлияет на сигнал? 
       Подсказка: Spectrum дает метод, называемый make_wave, вычисляющий wave, соответствующий Spectrum.
"""

"""
Exercise 2.4 
If you have a spectrum object, spectrum, and print the first few
values of spectrum.fs, you’ll see that they start at zero. So spectrum.hs[0]
is the magnitude of the component with frequency 0. But what does that
mean?
Try this experiment:
1. Make a triangle signal with frequency 440 and make a Wave with duration
0.01 seconds. Plot the waveform.
2. Make a Spectrum object and print spectrum.hs[0]. What is the amplitude
and phase of this component?
3. Set spectrum.hs[0] = 100. Make a Wave from the modified Spectrum
and plot it. What effect does this operation have on the waveform?
"""

# Построение треугольного сигнала длительностью 0.01 и графика для него
triangle_signal_4 = TriangleSignal(freq=440).make_wave(duration=0.01)
triangle_signal_4.plot()
decorate(xlabel='Time (s)')
plt.savefig(filePath + "4.triangle.signal" + fileExtension)
plt.close()

# Построение спектра сигнала и вывод 0 значения из hs
spectrum_4 = triangle_signal_4.make_spectrum()
print(spectrum_4.hs[0])

# Изменение нулевой компоненты и построение графиков для сравнения сигнала до изменения и после
spectrum_4.hs[0] = 100
triangle_signal_4.plot(color='gray')
spectrum_4.make_wave().plot()
decorate(xlabel='Time (s)')
plt.savefig(filePath + "4.triangle.signal.before.vs.triangle.signal.after" + fileExtension)
plt.close()

"""
Упражнение 2.5
Напишите функцию, принимающую Spectrum как параметр 
и изменяющую его делением каждого элемента hs на соответствующую частоту из fs.
Подсказка: поскольку деление на 0 не определено, надо задать Spectrum.hs[0] = 0.
Проверьте эту функцию, используя прямоугольный, треугольный или пилообразный сигналы:
    1. Вычислите Spectrum и распечатайте его.
    2. Измените Spectrum, вновь используя свою функцию, и распечатайте его.
    3. Используйте Spectrum.make_wave, чтобы сделать wave из измененного Spectrum, и прослушайте его.
       Какая операция повлияла на сигнал?
"""

"""
Exercise 2.5
Write a function that takes a Spectrum as a parameter and
modifies it by dividing each element of hs by the corresponding frequency
from fs. Hint: since division by zero is undefined, you might want to set
spectrum.hs[0] = 0.
Test your function using a square, triangle, or sawtooth wave.
1. Compute the Spectrum and plot it.
2. Modify the Spectrum using your function and plot it again.
3. Make a Wave from the modified Spectrum and listen to it. What effect
does this operation have on the signal?
"""

# Код для функции, изменяющей спектр у сигнала


def spectrum_divide(spectrum):
    i = 0
    while i < len(spectrum):
        if i == 0:
            spectrum.hs[i] = 0
        else:
            spectrum.hs[i] = spectrum.hs[i] / spectrum.fs[i]
        i += 1


# Создание нового пиловидного сигнала длительностью 1 секунда
wave_5 = SawtoothSignal().make_wave(duration=1)

# Создание спектра для сигнала, изменение спектра и отрисовка графиков до и после вызова функции для проверки
spectrum_5 = wave_5.make_spectrum()
spectrum_5.plot(color='gray')
spectrum_divide(spectrum_5)
spectrum_5.high_pass(100)
spectrum_5.scale(440)
spectrum_5.plot()
decorate(xlabel='Frequency (Hz)')
plt.savefig(filePath + "5.sawtooth.spectrum.before.vs.sawtooth.spectrum.after" + fileExtension)
plt.close()

# Создание записей сигнала до и после изменений для сравнения
wave_5.make_audio()
changed_signal_5 = spectrum_5.make_wave()
changed_signal_5.make_audio()

"""
Упражнение 2.6
У треугольных и прямоугольных сигналов есть только нечетные гармоники; 
в пилообразном сигнале есть и четные и нечетные гармоники.
Гармоники прямоугольных и пилообразных сигналов уменьшаются пропорционально 1/f;
гармоники треугольных сигналов - пропорционально 1/f^2.
Можно ли найти сигнал, состоящий из четных и нечетных гармоник, спадающих пропорционально 1/f^2?
Подсказка: для этого есть два способа. Можно собрать желаемый сигнал из синусоид,
а можно взять сигнал со спектром, похожим на необходимый, и изменять его параметры.
"""

"""
Exercise 2.6 
Triangle and square waves have odd harmonics only; the sawtooth
wave has both even and odd harmonics. The harmonics of the square
and sawtooth waves drop off in proportion to 1/ f ; the harmonics of the triangle
wave drop off like 1/ f^2. Can you find a waveform that has even and
odd harmonics that drop off like 1/ f^2?
Hint: There are two ways you could approach this: you could construct the
signal you want by adding up sinusoids, or you could start with a signal
that is similar to what you want and modify it.
"""

# Создание пилообразного сигнала для изменения его гармоник на пропорциональные 1/f^2
sawtooth_signal_6 = SawtoothSignal().make_wave(duration=0.5, framerate=20000)

# Изменение гармоник с помощью метода, написанного в предыдущем упражнении,построение графика для сравнения спектров
spectrum_6 = sawtooth_signal_6.make_spectrum()
spectrum_6.plot(color='gray')
spectrum_divide(spectrum_6)
spectrum_6.scale(440)
spectrum_6.high_pass(100)
spectrum_6.plot()
decorate(xlabel='Frequency (Hz)')
plt.savefig(filePath + "6.sawtooth.spectrum.before.vs.sawtooth.spectrum.after" + fileExtension)
plt.close()

# Построение графика сигнала по спектру измененного сигнала
spectrum_6.make_wave().segment(duration=0.01).plot()
decorate(xlabel='Time (s)')
plt.savefig(filePath + "6.sawtooth.signal.after" + fileExtension)
plt.close()
