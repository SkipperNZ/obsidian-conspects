
Upsampling
### Transposed Conv
Транспонированная свёртка 

![[Pasted image 20220503013726.png]]

* Вместо "сворачивания" с ядром (как в обычном ConvLayer) происходит "разворачивание" входного сигнала
* Значения входного сигнала выступают весами перед ядром транспонированной свертки

Тут ядро обучаемое, и ядро может быть вычислено более подходящим для нашей задачки образом. 


Есть свои нюансы:
При определённых комбинациях размера ядра свёртки и стрейда, может получатся шахматный паттерн который довольно сильно ухудшает качество модели:
[Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)

![[Pasted image 20220503014926.png]]


### SegNet (2015)

[SegNet: A Deep Convolutional Encoder-Decoder Architecture](https://paperswithcode.com/paper/segnet-a-deep-convolutional-encoder-decoder)

* Симметричная архитектура вида Encoder-Decoder
* Постепенный Upsampling


![[Pasted image 20220503015120.png]]

### UNet (2015)
Очень хорошо зарекомендовавший себя подход, по типу ResNet в классификации. 

[U-Net: CNNs for Biomedical Image Segmentation](https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical)

* Добавили горизонтальные связи к Encoder-Decoder

![[Pasted image 20220503015531.png]]

* Сильное улучшение сегментации на границах объектов
* Всего лишь 30 изображений 512х512 для обучения!


![[Pasted image 20220503015908.png]]

### UNet-like
Из конкретной архитектуры для сегментации UNet давно превратился в "подход" для задач image-to-image:
* Сегментация (subj)
* Колоризация (предсказание цветных каналов для grayscale-входа)
* InPainting ("закрашивание" пустот)
* ...

UNet-like сеть =
* encoder (resnet, efficientnet, …) +
* decoder

 image-to-image это когда на входе изображение и на выходе что то вроде изображения.

### Feature Pyramid Networks (FPN) (2017)
[Feature Pyramid Networks for Object Detection (2017)](https://paperswithcode.com/paper/feature-pyramid-networks-for-object-detection)

* Та самая пирамида признаков, что использовали в RetinaNet
* Идея подойдет и для улучшения сегментации

Собственно верхняя картинка это перевёрнутый U-net
Но если использовать не только последнюю карту но и другие то можно еще немного улучшить качество.
![[Pasted image 20220503021312.png]]

Из основных архитектур для сегментации вроде всё. 


Еще немного про альтернативные функции потерь для сегментации.
