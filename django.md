тут будут все заметки по джанго
#django #python #web

```toc
```

##  Гайд от ХаудиХо 

```
pip install django

python -m django --version
```



создание проекта NAME_PROJECT в директории из консоли.
```
django-admin startproject NAME_PROJECT
```


в папке появятся несколько файлов:

`__init__.py` - сообщает что данная директория является пакетом.

`settings.py` - настройки проекта

`urls.py` - корневая url привязка

`wsgi.py` -  _Web Server Gateway Interface_ — стандарт взаимодействия между Python-программой, выполняющейся на стороне сервера, и самим веб-сервером

Рассмотрим подробно `settings.py` :

режим дебага
```python
DEBUG = True
```

в целях защиты. суда прописываются все хосты которые django обслуживает, если пусто то обслуживает все
```python
ALLOWED_HOSTS = []
```

список установленных приложений 
```python
INSTALLED_APPS = [

 'django.contrib.admin',

 'django.contrib.auth',

 'django.contrib.contenttypes',

 'django.contrib.sessions',

 'django.contrib.messages',

 'django.contrib.staticfiles',

]
```

указание корневого роутинга
```python
ROOT_URLCONF = 'test1.urls'
```

настройки шаблонизатора
```python
TEMPLATES = [

 {

 'BACKEND': 'django.template.backends.django.DjangoTemplates',

 'DIRS': [],

 'APP_DIRS': True,

 'OPTIONS': {

 'context_processors': [

 'django.template.context_processors.debug',

 'django.template.context_processors.request',

 'django.contrib.auth.context_processors.auth',

 'django.contrib.messages.context_processors.messages',

 ],

 },

 },

]
```


настройка соединения с базой данных
```python
  
DATABASES = {

 'default': {

 'ENGINE': 'django.db.backends.sqlite3',

 'NAME': BASE_DIR / 'db.sqlite3',

 }

}
```

Больше автор ролика ни на что особых акцентов не делал. 

Для запуска данного сервера в консоли нужно прописать


Для запуска можно воспользоваться локальным тестовым сервером.
Используется команда:
```
python manage.py runserver
```

Для старта приложения прописывается:
```
python manage.py startapp NAME_APP
```

По умолчанию папка приложения появляется рядом с папкой проекта (там где лежит `manage.py`). Автор создал внутри проекта папку apps и переместил эту папку приложения туда.
А в файле `settings.py`  дописываем:

```python
import os, sys

PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'apps'))

```

Приложение - сущность которая выполняет какие то действия, в нашем случае приложение для добавления статей на сайт.

А проект - нечто что коллекционирует в себе список приложений, что бы получился готовый сайт.

Структура только что созданного приложения `articles`:

Директория `migtations` - в которой джанго коллекционирует миграции. 
#рассмотреть_позже 

`__init__.py` - сообщает что данная директория является пакетом.

`admin.py`  - указывать какие данные будут редактироваться в админке. 

`apps.py`  -  указывается конфиг приложения.

`models.py` - (важный файл) указывается модель своего приложения

`tests.py` - файл для тестов

`views.py` - (важный файл) логика приложения 

Так же ещё 2 файла, которые по умолчанию не создаются:

`urls.py` - так же url привязка, но локальная для приложения

`forms.py` - продвинутая техника в django для работы с формами, позволяет отображать, валидировать итд.

---
### views и urls

создадим `urls.py`. 

и запишем туда :
```python
from django.urls import path

  

from . import views

  

urlpatterns = [
 path('', views.index, name = 'index')
]
```

Дополним главный  `urls.py` всего проекта до такого состояния
```python
from django.contrib import admin

from django.urls import path, include # добавили инклюд

  

urlpatterns = [

 path('articles/', include('articles.urls')), # + эта строчка

 path('admin/', admin.site.urls),

]

```


Открываем `views.py`  приложения, там все удаляем и прописываем:
```python
from django.http import HttpResponse


def index(request):
 return HttpResponse("Hello, World!")
```

Ура! При запуске сервера и переходе на `http://127.0.0.1:8000/articles/` получаем страничку с записью 
Hello, World!

То есть когда мы запрашиваем `/articles`  джанго открывает главный файл url привязок и смотрит что записано в списке urlpatterns.
Он находит на там `/articles` и видит `include('articles.urls')`
где 'articles' - название приложения, а 'urls' файл привязки данного приложения.
В `urls.py` приложения мы продолжаем рассматривать путь привязок.
Джанго находит совподение в строке:
` path('', views.index, name = 'index')`
где первая часть `''` пустая строка. и джанго вызывает из файла `views.index`  вьюшку, которая называется "index". И уже в ней, мы просто возвращаем стандартный http ответ.

Для наглядности добавим еще одну вьюшку в приложение:
Добавляем в `urls.py` приложения:
`path('test/', views.test, name = 'test')`

Во `view.py` добавляем:
```python
def test(request):
 return HttpResponse("this is test")
```

Ура, всё заработало! по пути `http://127.0.0.1:8000/articles/test/`
выдаёт текст "this is test"

### Models
Разберёмся как работают модели и что это такое вообще.

Это файл - "объяснение" для джанги что есть/должно_быть в базе данных.  Объясняем из чего состоит приложение, что ей хранить в базе данных.

Для начала создадим модели, они представляют из себя python классы


`models.py`
```python
from django.db import models


class Article(models.Model):
 article_title = models.CharField('название статьи', max_length=200)
 article_text = models.TextField('текст статьи')
 pub_date = models.DateTimeField('дата публикации')


class Comment(models.Model):
 article = models.ForeignKey(Article, on_delete=models.CASCADE)
 author_name = models.CharField('имя автора', max_length=50)
 comment_text = models.CharField('текст комментария', max_length=200)
```

`CharField` - в базах данных небольшой текст на 200-300 символов.
`TextField` - большой текст.
`ForeignKey` - внешний ключ
`on_delete=models.CASCADE` - on_delete команда которая что то делает во время удаления комментария 

Теперь нужно синхронизировать модель с базой данных. 
Для этого в джанго есть такое понятие как миграции. 

Миграция это такая конструкция благодаря которому джанго может понимать какие изменения и когда он вносит в базу данных. 

Для того что бы создать миграцию в `settings.py` проекта в INSTALLED_APPS добавляем приложение в список установленных:
```python
INSTALLED_APPS = [
 'articles.apps.ArticlesConfig',
]
```

В консоли нужно вбить следующую команду:
```
python manage.py makemigrations articles
```

`articles` - название приложения

в ответ получаем сообщение о успешно созданной миграции 
```
Migrations for 'articles':
  test1\apps\articles\migrations\0001_initial.py
    - Create model Article
    - Create model Comment
```

что бы посмотреть как выглядят команды на sql в базу данных для этих миграций в консоли можно прописать:
```
python manage.py sqlmigrate articles 0001
```

Теперь мы можем применить созданную миграцию.
просто в консоли прописываем:
```
python manage.py migrate
```

Теперь все миграции применились.


### как работать с базами данных

В джанго есть свой ORM, свой шаблонизатор, система моделей, автосинхронизация

ORM - api для работы с базами данных. 
Попробуем работать с ним из консоли.

Воспользуемся утилитой в джанго которая называется shell:
```
python manage.py shell
```

откроется что то типа ipython только заточенный под джангу.
начнём писать:
```python 
from articles.models import Article, Comment
```

теперь допустим что нам нужно достать из базы данных все статьи что там есть:
```
Article.objects.all()
```

возвращается пустой список, так как статей у нас пока нет. Так что давайте добавим статью.

Заимпортируем недостающий модуль для работы со временем:
```
from django.utils import timezone
```

создаём статью:
```python
a = Article(
			article_title = "Какая-то статья",
			article_text = "Какой-то текст",
			pub_date = timezone.now(),
			)

```

и в одну строчку для копипасты:
`a = Article(article_title = "Какая-то статья", article_text = "Какой-то текст", pub_date = timezone.now())`

И теперь сохраняем её в бд:
```
a.save()
```


допишем немного магических методов для красивого отображения:
```python
from django.db import models


class Article(models.Model):
 article_title = models.CharField('название статьи', max_length=200)
 article_text = models.TextField('текст статьи')
 pub_date = models.DateTimeField('дата публикации')

 def __str__(self):
	 return self.article_title

  
  

class Comment(models.Model):
 article = models.ForeignKey(Article, on_delete=models.CASCADE)
 author_name = models.CharField('имя автора', max_length=50)
 comment_text = models.CharField('текст комментария', max_length=200)


 def __str__(self):
	return self.author_name
```

Теперь команда `Article.objects.all()` красивенько покажет все названия статей. 

Дополним модель своими методами:
```python
import datetime
from django.db import models
from django.utils import timezone

class Article(models.Model):
 article_title = models.CharField('название статьи', max_length=200)
 article_text = models.TextField('текст статьи')
 pub_date = models.DateTimeField('дата публикации')

 def __str__(self):
	return self.article_title

 def was_published_recently(self):
	return self.pub_date >= (timezone.now() - 
							 datetime.timedelta(days = 7))
```


Так можно получить в переменную статью с id 1 (в shell моде)
```
a = Article.objects.get(id = 1)
```

и у неё появится метод:
```
>>> a.was_published_recently()
True
```

Так же можно получить объекты по фильтру:
Пример(названия статей начинается с каких то символов/слов):
```
Article.objects.filter(article_title__startswith = "Какая")
```

так же можно изменять статью в объекте a и сохранять обновление в бд:
```
a.article_title = "Какое то обновлённое название"
a.save()
```

Для получения статей за определённый период(текущий или прошлый год):
```
from django.utils import timezone

current_yaer = timezone.now().year
Article.objects.filter(pub_date__year = current_yaer)
```

Если попробовать достать статью которой нет:
```
Article.objects.get(id = 2)
```

и получаем исключение:
```
    raise self.model.DoesNotExist(
articles.models.Article.DoesNotExist: Article matching query does not exist.
```

можем достать все комментарии к данное статье:
```
a.comment_set.all()
```

добавим комменты к статье:
```
a.comment_set.create(author_name = "Имя", comment_text = " текст комментария")
a.comment_set.create(author_name = "Имя2", comment_text = " текст комментария2")
```

можно посчитать комментарии к статье:
```
a.comment_set.count()
>>> 3
```

Удалить комментарии имена авторов начинающихся на "Дж":
```
cs = a.comment_set.filter(author_name__startswith = "Дж")
cs.delete()
```


### Админка
Создание учетки админа:
```
python manage.py createsuperuser
```

Для русификации админки в файле `settings.py` поменяем
'en-us' на 'ru-RU'

теперь в файле приложения `admin.py`  
```python 
from django.contrib import admin
from .models import Article, Comment


admin.site.register(Article)
admin.site.register(Comment)
```

Все, после этого в админке появились пункты для редактирования всего этого добра. 
Для того что бы русифицировать эти пункты в меню открываем файл `apps.py` приложения:
```python 
from django.apps import AppConfig

class ArticlesConfig(AppConfig):
 default_auto_field = 'django.db.models.BigAutoField'
 name = 'articles'
 verbose_name = 'Блог'

```

Таким образом русифицировалось только название блока.

Теперь заходим в файл `models.py`
И создаём подклассы вот так:
```python

class Article(models.Model):
 article_title = models.CharField('название статьи', max_length=200)
 article_text = models.TextField('текст статьи')
 pub_date = models.DateTimeField('дата публикации')

 def __str__(self):
	return self.article_title

 def was_published_recently(self):
	return self.pub_date >= (timezone.now() - 
							 datetime.timedelta(days = 7))

 class Meta:
	verbose_name = 'Статья'
	verbose_name_plural = 'Статьи'



class Comment(models.Model):
 article = models.ForeignKey(Article, on_delete=models.CASCADE)
 author_name = models.CharField('имя автора', max_length=50)
 comment_text = models.CharField('текст комментария', max_length=200)

 def __str__(self):
	return self.author_name

 class Meta:
	verbose_name = 'Коммент'
	verbose_name_plural = 'Комменты'
```

установим красивый модуль на админку:
называется django grappelli.
сайт: https://grappelliproject.com/

установка:
```
pip install django-grappelli
```

далее по инструкции из документации:
```
https://django-grappelli.readthedocs.io/en/latest/quickstart.html
```

что бы заработал 
```
 python manage.py collectstatic
```

Для этого в `settings.py` нужно создать константу 

```python
STATIC_ROOT = os.path.join(PROJECT_ROOT, 'static')
```

### Вывод статей на сайт
Сделаем 2 страницы и 3 вьюшки


создаём в главной директории проекта папку `templates`
В ней создадим папки `articles`
а уже в ней будут все html шаблоны приложения

В `settings.py` расширяем DIRS
```python
TEMPLATES = [
 {
 'BACKEND': 'django.template.backends.django.DjangoTemplates',
 'DIRS': [
 os.path.join(PROJECT_ROOT, 'templates')
 ],
```
Теперь джанго будет искать шаблоны не в папке приложения(подпапке темплейтс), а в свежесозданной папке.

Как устроен, как работает и зачем нужен шаблонизатор в джанго:

Сначала так же напишем просто код:

В только что созданной папке `templates` рядом с папкой `articles` создаём файл `base.html`

```django 
<!doctype html>

<html lang="en">
<head>
 <meta charset="utf-8" />
 <title>{% block title %}Мой сайт{% endblock %}</title>
</head>
<body>
 {% block content %}{% endblock %}
</body>
</html>
```

`{% block title %}Мой сайт{% endblock %}` - это конструкция шаблонизатора. Названия блоков могут быть любыми.

В папке `articles` создаём `list.html`

```django
{% extends 'base.html' %}

{% block title %} Последние статьи {% endblock %}

{% block content %}

Какая то надпись...

{% endblock %}
```

Теперь нужно создать рендер. открываем в папке приложения `views.py`:

меняем всё вот так:

```python
from django.http import HttpResponse
from django.shortcuts import render

def index(request):
	return render(request, 'articles/list.html')

def test(request):
	return HttpResponse("this is test")
```

Итак работает оно так:
Когда мы во `views.py` указываем render, он открывает `list.html` в нем он сначала видит extends и расширяет этот шаблон (что то типа наследования в ООП но с заменой блоков)

Теперь выведем сам список последних статей:
Во `views.py`:
```python
from django.http import HttpResponse
from django.shortcuts import render

from .models import Article, Comment

def index(request):
	latest_articles_list = Article.objects.order_by('-pub_date')[:5]
	return render(request, 'articles/list.html', {'latest_articles_list': latest_articles_list})

def test(request):
	return HttpResponse("this is test")
```

в `list.html`:
```django
{% extends 'base.html' %}
{% block title %} Последние статьи {% endblock %}
{% block content %}
 {% if latest_articles_list %}
 {% for a in latest_articles_list %}
 <a href="#">{{a.article_title}}</a>
 {% endfor %}
 {% else %}
 Статьи не найдены.
 {% endif %}
{% endblock %}
```

что бы ссылка заработала надо создать ссылку в файле `urls.py` нашего приложения

```python
from django.urls import path
from . import views

app_name = 'articles'
urlpatterns = [
 path('', views.index, name = 'index'),
 path('<int:article_id>/', views.detail, name = 'detail'),
 path('test/', views.test, name = 'test'),
]
```

теперь создадим новую вьюшку во `views.py`

```python
from django.http import HttpResponse
from django.shortcuts import render
from .models import Article, Comment

def index(request):
 latest_articles_list = Article.objects.order_by('-pub_date')[:5]
 return render(request, 'articles/list.html', {'latest_articles_list': latest_articles_list})
	
def test(request):
	return HttpResponse("this is test")

def detail(request, article_id):
	pass
```

и снова меняем `list.html`
```django
{% extends 'base.html' %}
{% block title %} Последние статьи {% endblock %}
{% block content %}
 {% if latest_articles_list %}
 {% for a in latest_articles_list %}
 <a href="{% url 'articles:detail' a.id %}">{{a.article_title}}</a>
 {% endfor %}
 {% else %}
 Статьи не найдены.
 {% endif %}
{% endblock %}
```

Теперь допишем вьюшку в `view.py`

```python
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.shortcuts import render
from .models import Article, Comment

def index(request):
 latest_articles_list = Article.objects.order_by('-pub_date')[:5]
	return render(request, 'articles/list.html', {'latest_articles_list': latest_articles_list})


def test(request):
	return HttpResponse("this is test")

def detail(request, article_id):
 try:
	a = Article.objects.get( id = article_id)
 except:
	raise Http404("Статья не найдена")
	return render(request, 'articles/detail.html', {'article': a})
```

создадим шаблон `detail.html` рядом c `list.html`

(во время написания шаблона сделали экшон добавления комментов)
В `urls.py` теперь добавили лив коммент:
```python
urlpatterns = [
 path('', views.index, name = 'index'),
 path('<int:article_id>/', views.detail, name = 'detail'),
 path('<int:article_id>/leave_comment/', views.leave_comment, name = 'leave_comment'),
 path('test/', views.test, name = 'test'),
]
```

Добавляем во  `vews.py`:
```python
def leave_comment(request, article_id):

 try:
	a = Article.objects.get( id = article_id)
 except:
	raise Http404("Статья не найдена")
 
 a.comment_set.create(author_name = request.POST['name'],
 comment_text = request.POST['text'])
 return HttpResponseRedirect(reverse('articles:detail',args = (a.id,) ))
```

`detail.html`:
```django
{% extends 'base.html' %}
{% block title %} {{article.article_title}} {% endblock %}

{% block content %}

<h2>{{article.article_title}}</h2>

<p>{{article.article_text}}</p>

<em>{{article.pub_date}}</em>

<hr>

<hr>

<form action="{% url 'articles:leave_comment' article.id %}" method="POST">
 {% csrf_token %}
 <input type="text" required placeholder="Ваше имя" name="name"><br>
 <textarea name="text" required="" placeholder="Текст комментария" cols="30" rows="10">
 </textarea><br>
 <button type="submit">Оставить комментарий</button>
</form>
{% endblock %}

```

Что бы показывать список коментов на страничке, подредактируем вьюшку:
```python
def detail(request, article_id):
 try:
	 a = Article.objects.get( id = article_id)
 except:
	 raise Http404("Статья не найдена")

 latest_comments_list = a.comment_set.order_by('-id')[:10]
  return render(request, 'articles/detail.html', {'article': a, 'latest_comments_list': latest_comments_list})

```

и в `detail.html`:
```django
<hr>

{% if latest_comments_list %}
 {% for c in latest_comments_list %}
 <p>
 <strong>{{c.author_name}}</strong>
 <p>
 {{c.comment_text}}
 </p>
 </p>
 {% endfor %}
{% else %}
Комметнтарии не найдены
{% endif %}
<hr>
```
