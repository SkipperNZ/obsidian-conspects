[[Радиолокация]]


# Введение в миллиметровое зондирование  FMCW  радары

**FMCW радар** - радиолокатор непрерывного излучения с частотной модуляцией (Frequency-Modulated Continuous Wave radar)


## Словарь

Chirp - ЛЧМ сигнал

$\large f_{c}$ - Начальная частота ЛЧМ сигнала

$\large T_{c}$ - длительность ЛЧМ сигнала

**B** (Bandwidth) - частотный диапазон ЛЧМ сигнала.

S (slope) - наклон ЛЧМ сигнала (скорость возрастания частоты от времени)

Tx - transfer - передача данных
Rx - receive - получение данных

IF (Instantaneous frequency) - ПЧ промежуточная частота.

##  Измерение дальности до объекта

### ЛЧМ сигнал 

![[Pasted image 20220709134035.png]]

$\large f_{c}$ - Начальная частота ЛЧМ сигнала
$\large T_{c}$ - длительность ЛЧМ сигнала
**B** (Bandwidth) - частотный диапазон ЛЧМ сигнала.
**S** (slope) = **B**/$\large T_{c}$  -  скорость возрастания частоты от времени на картинке частотный диапазон 4 ГГц и проходит он этот диапазон за 40 мкс. Тогда S в этом случае 100 МГц/мкс.

### 1TX-1RX FMCW radar
![[Pasted image 20220709135329.png]]

В миксере смешивается передаваемый в эфир сигнал, и принимаемое его отражение результирующий сигнал называется IF сигнал (Intermediate frequency) (в советских книжках ПЧ - промежуточная частота)
![[Pasted image 20220709142857.png]]

Входные сигналы в миксер
$\large x_{1}=\sin \left[w_{1} t+\phi_{1}\right]$
$\large x_{2}=\sin \left[w_{2} t+\phi_{2}\right]$

Выходной сигнал:
$\large x_{o u t}=\sin \left[\left(w_{1}-w_{2}\right) t+\left(\phi_{1}-\phi_{2}\right)\right]$

Видно что промежуточная частота (IF Instantaneous frequency) это разница между двумя пришедшими синусоидами.

Фаза это просто разность фаз двух пришедших синусоид.

$\large \tau$ - разность по времени между отправленным и отраженным сигналом.
S*$\large \tau$ - промежуточная частота, до этой частоты возрастает отправляемый сигнал, до того момента когда приходит отраженный и начинает происходить вычитание.

Сдедовательно объект перед радаром производит сигнал (IF signal) с постоянной частотой.
Частота у такого сигнала: 
$\large S\cdot\tau = \frac{S \cdot 2d}{c}$ , так как  $\large\tau = \frac{2d}{c}$ где d - расстояние до объекта а с - скорость света, а на два умножили, так как сигнал туда-сюда летит.

![[Pasted image 20220710050140.png]]

Вывод:
Один объект перед радаром порождает IF сигнал c константной частотой, которая равна $\large \frac{S \cdot 2d}{c}$

Так же обратим внимание, что $\large\tau$ - это обычно маленькая часть от всей длительности сигнала сдедовательно неперекрытая часть передаваемого сигнала отраженным - незначительна.
для радаров с максимальной дальнобойностью в 300 метров и длинной ЛЧМ Tc = 40us:   $\large \tau$/Tc = 5%


### Множественные объекты перед радаром

Много объектов перед радаром => много отраженных ЛЧМ чирпов на приемной RX антенне.

![[Pasted image 20220710055205.png]]

Частотный спектр такого IF сигнала выдаст несколько тонов, и частота каждого - пропорциональна  дистанции до каждого объекта.
![[Pasted image 20220710055406.png]]


### Разрешение по дальности в радаре. 
Тут видно, что 2 объекта расположены слишком близко, и на частотном спектре они сливаются в один объект.
![[Pasted image 20220710141240.png]]
![[Pasted image 20220710141347.png]]


Но если увеличить IF signal то мы уже сможем их различать. 

(Так же заметим, что пропорционально увеличилась полоса bandwidth отсюда вывод: больше bandwidth - лучше разрешение)

![[Pasted image 20220710141459.png]]
![[Pasted image 20220710141509.png]]


Вспомним:
* Объект на расстоянии d порождает IF tone с частотой S2d/c
* Два тона могут быть различены на спектре, до тех пока пока разность из частот Δf > 1/T (где T, длительность наблюдения сигналов.)

Воспользуемся этим, что бы вывести уравнение разрешения по дальности. 

Для 2х объектов разнесенных на расстояние Δd друг от друга, 
разница в их IF частотах определяется как:
$$\large
\Delta \mathrm{f}=\frac{\mathrm{S} 2 \Delta \mathrm{d}}{\mathrm{c}}
$$
Интервал наблюдения у нас $\large T_{c}$ Это означает что:
$$\large
\Delta \mathrm{f}>\frac{1}{\mathrm{~T}_{\mathrm{c}}} \Rightarrow \frac{\mathrm{S} 2 \Delta \mathrm{d}}{\mathrm{c}}>\frac{1}{\mathrm{~T}_{\mathrm{c}}} \Rightarrow \Delta \mathrm{d}>\frac{\mathrm{c}}{2 \mathrm{ST}_{\mathrm{c}}} \Rightarrow \frac{\mathrm{c}}{2 \mathrm{~B}}
$$
Мы считаем что B = STc так как тау в основном очень малая часть.

Тогда мы получаем что разрешение по дальности $\large d_{res}$ зависит только от ширины полосы Bandwidth ЛЧМ сигнала:
$$\large
d_{r e s}=\frac{c}{2 B}
$$
Ну и немного цифр из реального мира
![[Pasted image 20220710145617.png]]

### Оцифровка IF сигнала 
ADC - АЦП
DSP - ЦОС
![[Pasted image 20220710145853.png]]

* Интересуемая полоса пропускания зависит от желаемого максимального расстояния: $\large\mathrm{f}_{\mathrm{IF}_{-} \max }=\frac{\mathrm{S} 2 \mathrm{~d}_{\mathrm{max}}}{\mathrm{c}}$
* IF сигнал обычно оцифровывается с помощью ФНЧ+АЦП для дальнейшей обработки.
* Таким образом полоса пропускания IF ограничена частотой дискретизации АЦП ($\large F_{s}$)  $\large F_{s}\geq \frac{S2d_{max}}{c}$   

Частота дискретизации АЦП - $\large F_{s}$. Она ограничивает максимальную дальность радара до:
$$\large
d_{max} = \frac{F_{s}c}{2S}
$$

Быстрое фурье от IF сигнала называется "range-FFT"
![[Pasted image 20220710152101.png]]


### Полоса пропускания ЛЧМ vs IF полоса пропускания 
![[Pasted image 20220710152326.png]]


2 объекта равноудалены от радара. как выглядит для них range-FFT?
![[Pasted image 20220710152459.png]]

Как мы можем разделить эти объекты?
Если эти объекты имеют разную скорость, относительно радара, то они могут быть разделены с помощью обработки сигналов.
Надо взглянуть на фазу сигналов.

##  Фаза  IF сигнала

При преобразовании Фурье есть не только амплитудный, но и фазовый спектр:
![[Pasted image 20220710153245.png]]

Фазовый спектр - значение начальных фаз всех компонент в нулевой момент времени
![[Pasted image 20220710153438.png]]

![[Pasted image 20220710153551.png]]

Напоминаем, что начальная фаза сигнала на выходе смесителя - это разность начальных фаз двух входов.

Для объекта, на расстоянии d от радара, IF сигнал будет синусоидой.
![[Pasted image 20220710153905.png]]

Что случится, если полет ЛЧМ сигнало туда-сюда увеличится на небольшое значение $\large \Delta \tau$ 
Разность фаз между точками A и D расчитывается так:
$$\large
\Delta \Phi = 2 \pi f_{c}\Delta \tau =
\frac{4\pi\Delta d}{\lambda}
$$
Это также разность фаз между C и F.

![[Pasted image 20220710155646.png]]

Для объекта на расстоянии d от радара сигнал IF будет синусоидой:
$$\large
A\cdot \sin(2\pi ft + \phi_{0})
$$
![[Pasted image 20220710191308.png]]

Рассмотрим ЛЧМ сигнал с вот такими характеристиками:
![[Pasted image 20220710191901.png]]
* Что случится если объект перед радаром изменит свое положение на 1мм (для 77ГГц радара 1мм= λ/4) 
* Фаза IF сигнала изменится на $\large \Delta \phi = \frac{4\pi\Delta d}{\lambda} = \pi =180^{o}$ 
* Частота IF сигнала изменится на:
$$\large
\Delta f = \frac{S2\Delta d}{c}= \frac{50\cdot10^{12}\cdot2\cdot1\cdot10^{-3}}{3\cdot10^{8}} = 333 \ \  Hz
$$
Выглядит как довольно большая цифра но на рассматриваемом промежутке Tc:
$$\large 
\Delta f T_{c} = 333\cdot40\cdot10^{-6} = 0.013 \ \ \ \ цикла
$$
И на частотном спектре мы не увидим каких либо изменений 

**Фаза IF сигнала очень чувствительна к небольшим изменения расстояний**

Объект на определённой  дистанции порождает IF сигнал  на определённой частоте и фазе.

![[Pasted image 20220710193819.png]]

Небольшое изменение положения объекта меняет фазу IF сигнала, но не частоту.
![[Pasted image 20220710194207.png]]

### Как измерить скорость v объекта используя 2 ЛЧМ сигнала

* Отправляем 2 сигнала друг за другом на расстоянии Tc(прям сразу один за другим)
* range-FFT для каждого ЛЧМ сигнала будут одинаковыми в амплитудной части и разные в фазовой.
* Измеренная разность фаз (ω) соответсвует движению объекта $\large v T_{c}$  

![[Pasted image 20220710194432.png]]
![[Pasted image 20220710194956.png]]

Разность фаз измеренная для 2х последовательный ЛЧМ сигналов может использоваться для измерения скорости объекта.

### Измерения на вибрирующем объекте.
Тут нарисованы эволюции во времени колеблющегося объекта.
предполагается что колебания малы, поэтому максимальная дельта смещения  - доли длинны волны (миллиметры или меньше). 
Что будет если мы поставим радар перед таким объектом и будем передавать кучу равноудалённый ЛЧМ. 
Каждый из этих ЛЧМ приведёт к отраженному ЛЧМ. А у каждого отраженного сигнала будет пик в частотной области. 
Частота у него меняться особо не будет, а вот фаза будет.
![[Pasted image 20220710195406.png]]


![[Pasted image 20220710200118.png]]

**Изменения фазы во времени можно использовать для оценки как амплитуды, так и периодичности сигнала.**

Итак имеем много объектов равноудалённых от радара но движущихся с разной скоростью:
![[Pasted image 20220710201733.png]]

Как нам разделить  эти объекты?

Объекты равноудалённые от радара, но имеющие разные скорости, могут быть распознаны с помощью Doppler-FFT.

## Оценка скорости. 
### БПФ на комплексной последовательности( небольшой обзор)
Рассмотрим дискретный сигнал который соответствует вектору, вращающимся с постоянной скоростью ω радиан в отсчет. 
БПФ на этих выборках отсчетов выдаст пик на частоте ω.
![[Pasted image 20220710204128.png]]

Если сигнал состоит из суммы 2х векторов, FFT будет иметь 2 пика (каждый вектор вертится со своими $\large \omega_{1}$ и $\large \omega_{2}$ радиан в отсчёт соответственно) 
![[Pasted image 20220710204540.png]]


* $\large \omega_{1} = 0$ и $\large \omega_{2}= \frac{\pi}{N}$. За N семплов 2ой вектор прошел на пол цикла, ($\large \pi$ радиан) больше чем первый. И этого не достаточно для разрешения в частотной области.
![[Pasted image 20220710205549.png]]

* Через 2N отсчетов, 2ой вектор совершает на полный круг (2$\large \pi$ радиан) больше чем предыдущий вектор, и видно что эти 2 объекта разделимы в частотной области: 
![[Pasted image 20220710205915.png]]

Отсюда можно сделать вывод, что большая длинна выборки дает лучшее разрешение. В общем случае, последовательность длинны N может разделить угловые частоты ращнесенные более чем на 
2$\large \pi$/N радиан/отсчет.


Сравнение для дискретных и непрерывных сигналов:
![[Pasted image 20220710210631.png]]

### Как измерить скорость v объекта используя 2 ЛЧМ сигнала
(это со слайдов выше для лучшего понимания того что идёт дальше)
* Отправляем 2 сигнала друг за другом на расстоянии Tc(прям сразу один за другим)
* range-FFT для каждого ЛЧМ сигнала будут одинаковыми в амплитудной части и разные в фазовой.
* Измеренная разность фаз (ω) соответсвует движению объекта $\large v T_{c}$  

![[Pasted image 20220710194432.png]]
![[Pasted image 20220710194956.png]]

Разность фаз измеренная для 2х последовательный ЛЧМ сигналов может использоваться для измерения скорости объекта.

### Максимальная измеряемая скорость

![[Pasted image 20220710211414.png]]

![[Pasted image 20220710211500.png]]

* Видно что мы имеем неоднозначность если разность фаз  меньше 180 градусов (пи).
$$\large
\frac{4\pi v T_{c}}{\lambda} < \pi \Rightarrow 
v < \frac{\lambda}{4T_{c}}
$$
Отсюда вывод:
Максимальная распознаваемая скорость $\large v_{max}$ может быть измерена двумя ЛЧМ сигналами разделёнными $\large T_{c}$  и вычисляется как:
$$\large
v_{max} = \frac{\lambda}{4T_{c}}
$$
Отсюда, нам нужны более близкие ЛЧМ чирпы для измерения большей скорости.

### Измерение скорости с несколькими объектами на одном расстоянии.

Рассмотрим 2 равноудалёных объекта, приближающихся к радару со скоростями v1 и v2
![[Pasted image 20220710213336.png]]

Значение у вектора, тудет от 2х объектов, следовательно предыдущий подход не будет работать.

Решение: 
Трансмитим N чирпов на равноразнесенном расстоянии. (Это обычно называется фрейм)

![[Pasted image 20220710213928.png]]

БПФ последовательности векторов, соответствующих пикам диапазона БПФ, разрешает два объекта. это называется **doppler-FFT**
Более понятная картинка: 
![[Pasted image 20220710214702.png]]

ω1 и ω2 соответствуют разности фаз между последовательностями чирпов для соответствующих объектов.
![[Pasted image 20220710215036.png]]

### Разрешение по скорости
Какое разрешение по скорости$\large v_{res}$ у “doppler-FFT”?
*  т.е какая минимально различимася скорость между v1 и v2 что бы увидеть 2 пика на доплеровском fft.
![[Pasted image 20220710215601.png]]
Разрешение радара по скорости обратно пропорционально
пропорциональна времени кадра
![[Pasted image 20220710215548.png]]

### Визуализация 2-мерного FFT
Два объекта, равноудаленные от радара, приближаются к нему на разных скоростях 
![[Pasted image 20220710220309.png]]

![[Pasted image 20220710220415.png]]

### Двумерное преобразование фурье в 2х словах

Данные после АЦП для каждого чирпа хранятся в строках матрицы
![[Pasted image 20220710231738.png]]

Далее для каждой строки делается range-FFT которая показывает расстояния до объектов.
![[Pasted image 20220710231832.png]]

Далее делается доплеровское fft вдоль столбцов, разрешает какдый столбекц(range-bin) по скорости. 
![[Pasted image 20220710232025.png]]

В большинстве реализаций range-FFT  выполняется в блоке где идёт сохранение с АЦП
![[Pasted image 20220710232542.png]]

### Требования к ЛЧМ Chirp
Данны разрешение по расстоянию $\large d_{res}$ максимальная дальность действия $\large d_{max}$ разрешение по скорости $\large v_{res}$ и максимальная скорость $\large v_{max}$. Как нам задизайнить фрейм?
![[Pasted image 20220710232917.png]]

1. $\large T_{c}$ - определяется с помощью $\large v_{max}$
2. B - определяется с помощью $\large d_{res}$. (обратим внимание что B и $\large T_{c}$  уже выбрали и можем определить S = B/$\large T_{c}$)
3. Время всего фрейма $\large T_{f}$  определяется с помощью $\large v_{res}$ 
![[Pasted image 20220710233246.png]]

На практике этот процесс может быть более итеративным.
- Максимальная требуемая пропускная способность IF может не поддерживаться нашим железом.
- так как $\large f_{IFmax}=S2d_{max}/c$ может быть нужен компромисс между S и $\large d_{max}$ 
- Железо должно быть в стостоянии генерировать нужный нам S
- У железа могут быть спец. требования к промежутку между импулсами
- Может быть требованиек по памяти для преобразования фурье.

![[Pasted image 20220710233854.png]]


### Уравнение дальности действия радара 
![[Pasted image 20220710234102.png]]
![[Pasted image 20220710234123.png]]

Допустим два объекта, равноудаленные от радара и имеющие одинаковую скорость относительно радара. какой range-velocity график они будут иметь?

Будет один пик, так как у них одинаковая скорость и одинаковое расстояние.
![[Pasted image 20220710234344.png]]

Как разделить эти два объекта?
Для этого нужно несколько RX приемных антенн.


## Оценка угла

Тут мы рассмотрим следующие вопросы:

Как радар оценивает угол до объекта перед радаром?
![[Pasted image 20220710234747.png]]

Что если у нас несколько объектов под разными углами?
![[Pasted image 20220710234824.png]]

От чего зависит разрешение по углу?
![[Pasted image 20220710234856.png]]

От чего зависит максимальный угол обзора?
![[Pasted image 20220710234931.png]]

### Основа оценки угла прихода Angle of Arrival (AoA)
Напомним, что небольшое изменение расстояния до объекта приводит к изменению фазы (ω) в пике диапазона БПФ.
![[Pasted image 20220710235039.png]]

Для оценки угла требуется как минимум 2 антенны RX.

Разное расстояние от объекта до каждой из антенн
приводит к фазовому изменению пика 2D-БПФ, которое используется для оценки угла прихода.

![[Pasted image 20220710235203.png]]

Почему эта омега в 2 раза меньше чем на предыдущей картинке?

**Как измерить AoA объекта, используя 2 антенны RX**

* Антенна TX передает фрейм чирпа
* 2D-FFT, соответствующий каждой RX-антенне, будет иметь пики в том же месте, но с разной фазой
* Измеренная разность фаз (ω) может использоваться для оценки угла до объекта
![[Pasted image 20220710235540.png]]
![[Pasted image 20220710235559.png]]

Соответственно получаем такое выражение: 
![[Pasted image 20220710235639.png]]

### Точность оценки зависит от AoA

Чувствительность sin(θ) к θ ухудшается по мере увеличения θ
![[Pasted image 20220710235735.png]]

* Обратите внимание, что связь между ω и θ является нелинейной. (в отличие от случая скорости, когда $\large ω = \frac{4\pi v T_{c}}{\lambda}$ )
* При θ = 0 ω наиболее чувствителен к изменениям θ. Чувствительность ω к θ уменьшается по мере увеличения θ (и становится нулевой при угде в 90 гразусов на пике синусоиды)
* Следовательно, оценка θ более подвержена ошибкам по мере увеличения θ.

![[Pasted image 20220711000127.png]]

### Угловое поле зрения

![[Pasted image 20220711000236.png]]

Так же возникает неопределённость если модуль ω больше 180 градусов.
![[Pasted image 20220711000342.png]]
![[Pasted image 20220711000401.png]]

Максимальное поле зрения, которое могут давать две антенны, разнесенные на d, равно
![[Pasted image 20220711000435.png]]

Соответственно расстояние между антеннами d равное λ/2 даст наибольший угол обзора (+/- 90 градусов )


### Измерение углов прибытия AoA от нескольких объектов  которые находятся на одинаковом удалении и с одинаковой скоростью.

Рассмотрим два объекта, равноудаленные от радара, приближающиеся к радару с одинаковой относительной скоростью к радару.
![[Pasted image 20220711000742.png]]

Тут так же предыдущий подход работать не будет, надо что то новое. 

Решение: 
Нужно применить фазированную антенную решетку. из N антенн.
![[Pasted image 20220711001026.png]]

БПФ последовательности векторов, соответствующих пикам 2D-БПФ, разрешает два объекта. Это называется угловым БПФ (angle-FFT).

ω1 и ω2 соответствуют разности фаз между последовательными чирпами для соответствующих объектов.
![[Pasted image 20220711001230.png]]

Угловое разрешение ($\large θ_{res}$) — это минимальное расстояние между двумя объектами, чтобы они отображались как отдельные пики на изображении angle-FFT. Получается по формуле:
![[Pasted image 20220711001434.png]]
Тут видим зависимость от угла θ и лучшее при θ=0

Разрешение часто определяют при вот таких параметрах по такой формуле:
![[Pasted image 20220711001633.png]]

![[Pasted image 20220711001724.png]]

![[Pasted image 20220711001817.png]]

![[Pasted image 20220711001849.png]]

### Сравнение оценки угла и скорости

Как оценка угла, так и оценка скорости основаны на сходных концепциях, поэтому попробуем их сравнить. 
![[Pasted image 20220711002055.png]]


### Оценка угла в радаре FMCW
![[Pasted image 20220711002128.png]]
![[Pasted image 20220711002155.png]]

* Одна цепочка TX, RX может оценить дальность и скорость нескольких объектов.
* Помимо дальности, для определения местоположения необходима информация об угле.
* Для оценки угла необходимы несколько антенн RX.
	* Сетка 2D FFT создается в каждой цепочке RX (соответствующей каждой антенне)
	* БПФ на соответствующем пике между антеннами используется для оценки угла


### Диапазон, скорость и угловое разрешение
Range, Velocity and Angle Resolution

![[Pasted image 20220711002724.png]]

![[Pasted image 20220711002737.png]]
![[Pasted image 20220711002906.png]]





