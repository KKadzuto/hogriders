# HogRiders - Система Анонимизации Данных

## Описание
HogRiders - это локальное приложение для защиты персональных данных. Система обеспечивает безопасную обработку и анонимизацию конфиденциальной информации без передачи данных в облако.

## Основные возможности
- Локальная обработка данных
- Анонимизация персональных данных:
  - ФИО
  - Телефоны
  - Email адреса
  - Адреса
  - ИИН
  - Номера банковских карт
  - Даты рождения
  - Паспортные данные
- Поддержка различных форматов:
  - Текстовые файлы (.txt)
  - Изображения (.jpg, .jpeg, .png)
- Размытие лиц на фотографиях
- Экспорт результатов в безопасном формате

## Требования
- Python 3.8+
- Tesseract OCR
- CUDA (опционально, для ускорения обработки)

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/hogriders.git
cd hogriders
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
.\venv\Scripts\activate 
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Установите Tesseract OCR:
- Windows: Скачайте и установите с [официального сайта](https://github.com/UB-Mannheim/tesseract/wiki)
- а так же установите ИИ через ссылку https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth а затем положите его в папку models
## Использование

1. Запустите приложение:
```bash
python app.py
```

2. Откройте браузер и перейдите по адресу:
```
http://localhost:5000
```

3. Загрузите файл или введите текст для анонимизации

4. Получите результат и скачайте обработанный файл

## Безопасность
- Все данные обрабатываются локально
- Нет передачи данных в облако
- Используется шифрование для хранения результатов
- Автоматическое удаление временных файлов

## Автор
KKadzuto

## Поддержка
При возникновении проблем создайте issue в репозитории проекта. 