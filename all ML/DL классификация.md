### AlexNet (2012)

-   [ImageNet Classification with Deep Convolutional Neural Networks](https://paperswithcode.com/paper/imagenet-classification-with-deep)
-   Первый победитель ILSVRC на основе нейросетей
-   60M параметров
-   2 ветви на 2 GPU
-   Свертки от 3х3 до 11х11
-   ReLU, Dropout (в FC)
-   Test-time Augmentation
    
![[Pasted image 20220329184621.png]]

![[Pasted image 20220329184640.png]]

* Аугментации на инференсе (Test-time Augmentation, TTA):

Агрегирование предсказаний для 1 изображения по 10 его "вариациям":
	* 5 кропов (углы + центр)
	* x2 с учетом отраженных


Другая улучшенная сеть 
### VGG (2014)
* Свертки 1х1 и 3х3
* 138M параметров у самой большой вариации (VGG-19)

* Семейство моделей
* Отличаются глубиной
	* VGG-11 / 13 / 16 / 19
* Огромная часть параметров - в выходных слоях для классификации

![Image](https://lh5.googleusercontent.com/H1X-89jGX5PNuYmgpJH6LMm5L3BX-fFkNph6qfUiRpLgwN4aNQxhaBnFsy7apAXXoXsEiPe5t_V9mg0uEoqp0Wa6CAiObW8n7XD_DJdHgoaLvg7Kmhlgi5jAKb4_KFhQG010RP6PfdAH)

Раньше можа была такая (три огромных полносвязных слоя в конце)

Сейчас уже пришло понимание, что в конце сети больше одного линейного слоя обычно ставить не нужно, а нужно сложность засовывать в часть свёрточную.


Оказалось, что предобученная VGG извлекает “хорошие” признаки, пригодные для переиспользования

Например в задаче style Transfer
![Image](https://lh4.googleusercontent.com/CKNtph6U8WPq02GkUINqyA9N7d5hQjDGFA4460dT_VX8DlYVDcKcMoT51sYOT69gWb-ajDSzpe_9UDaBR2hhU0tunhKPNpCkt4G0lMtsO-0Uh_A13-qAIcm6nA4Ji1Gn3EZfsDrhYKvv)


Следующая сеть 
### Inception / GoogLeNet (2014)

* [Going Deeper with Convolutions](https://paperswithcode.com/paper/going-deeper-with-convolutions)
* Одновременное извлечение признаков разного масштаба
	* Новая структурная единица - InceptionBlock
* Дополнительные выходы с функцией потерь
* Global Average Pooling перед классификацией
* Мало параметров (5M)


 InceptionBlock (“ванильный” вариант)
    
![Image](https://lh5.googleusercontent.com/Nfs68--luNj290C4ce13fvGAtZe_c0ZqIsSNX55YBwOwHmxdaBOaMppuLnlquhJbXf5_PFNWmOsDQTLTLl3vdv3a-4VIXD-Om61gwMwuxDX2wOQ5nEInFGxw4RYhruo9VmvRmBwgm16o)

В конце конкатенация этого добра.

**InceptionBlock (bottleneck-вариант)**
![Image](https://lh5.googleusercontent.com/QuPtKJ_QG4CpRmU01z3Wxzgn8hsM8MhnAugk3_EIRzADOZG4cv3Ogr9nxVfG2kqeKIHVPNhLLLU7jw3dEHNlFIZMAD-6oQ5QaAzol3dWt8Gyz9NVgJvdGU6de5dyoox7QhhY5dSx-BnE)

![Image](https://lh5.googleusercontent.com/JQYSdr6Zy7YKUoVjdHDCAqxcPhe3l1I7XNOgtGrfSrGjVHKkyGYwmTL8_GDjJqC6QC5nIVJJzqSzc_U4j_yFTDaT9TpUdrwjlk-qMO0SAMJK2PZ2ZvoAnNxgEXXiPKMMnTPdrJ1JeNT6)


![Image](https://lh6.googleusercontent.com/L2JCgG6hnKsnvA9I6YB9oxgBalEfHMwnoDUZJAIqtZY-FgxuLtKRs687xylCm87gghiwgfDqGFYAUyPxzHZuAhKYf_m21luCWzXK78_E1EaEzGqGFuTF3aR3p5aEEnUElkEdRoo_BvB-)

два выхода с  слева и по центру - только для обучения
(Такой подход в результате не выжил в истории)

### Inception-BN / Inception-V2 (2015)

Тут фишка в батчнорме


### ResNet
* [Deep Residual Learning for Image Recognition](https://paperswithcode.com/paper/deep-residual-learning-for-image-recognition)
* Наращивание глубины сети с помощью Residual Connections
	* До 152 слоев


Идея: позволить сети “пропускать” слои, если они, например, не улучшают качество
![[Pasted image 20220330184429.png]]

![[Pasted image 20220330185104.png]]

![[Pasted image 20220330185252.png]]


 2 типа базовых блоков
	* ResNet-18/34: “обычный” блок (слева)
	* ResNet-50/101/152: bottleneck-блок (справа)

![[Pasted image 20220330185707.png]]

### ResNeXt

![[Pasted image 20240622163513.png]]

### Squeeze-n-Excitation (SENet)
![[Pasted image 20240622164111.png]]
![[Pasted image 20240622164204.png]]

### EfficientNet

Идея:
Подобрать эфективный способ масштабирования размерности моделей.
* Размер входного изображения("разрешение")
* Число Фильтров в сверточных слоях ("ширина")
* Число сверточных слоев("глубина")
![[Pasted image 20240622212020.png]]
![[Pasted image 20240622212209.png]]
![[Pasted image 20240622212419.png]]

Базовый блок MBConv+SE
![[Pasted image 20240622222626.png]]
### MobileNet

Depthwise convilution
![[Pasted image 20240622223159.png]]
![[Pasted image 20240622223539.png]]

 


