from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import re
from PIL import Image, ImageFilter
import pytesseract
import io
import base64
import os
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
import random
import string
import hashlib
import sys
import tempfile
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Настройка пути к Tesseract OCR для Windows
if os.name == 'nt':  # для Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Инициализация RWKV модели
MODEL_DIR = "models"
MODEL_NAME = "RWKV-4-Pile-169M-20220807-8023.pth"
model_path = os.path.join(MODEL_DIR, MODEL_NAME)

# Создаем временную директорию для файлов
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Создаем папку для загрузок, если её нет
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Загружаем модель RWKV
try:
    strategy = 'cuda fp16' if torch.cuda.is_available() else 'cpu fp32'
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Проверяем структуру state_dict
    if isinstance(state_dict, dict):
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
    # Создаем модель
    model = RWKV(model=state_dict, strategy=strategy)
    pipeline = PIPELINE(model)
except Exception as e:
    print(f"Ошибка при загрузке RWKV модели: {str(e)}")
    model = None
    pipeline = None

def generate_random_name():
    first_names = ['Александр', 'Дмитрий', 'Максим', 'Иван', 'Артем', 'Никита', 'Михаил', 'Даниил', 'Егор', 'Андрей']
    last_names = ['Иванов', 'Смирнов', 'Кузнецов', 'Попов', 'Васильев', 'Петров', 'Соколов', 'Михайлов', 'Новиков', 'Федоров']
    middle_names = ['Александрович', 'Дмитриевич', 'Максимович', 'Иванович', 'Артемович', 'Никитич', 'Михайлович', 'Даниилович', 'Егорович', 'Андреевич']
    return f"{random.choice(first_names)} {random.choice(last_names)} {random.choice(middle_names)}"

def generate_random_iin():
    return ''.join(random.choices(string.digits, k=12))

def generate_random_card():
    groups = [''.join(random.choices(string.digits, k=4)) for _ in range(4)]
    return ' '.join(groups)

def generate_random_phone():
    return f"+7{''.join(random.choices(string.digits, k=10))}"

def generate_random_email():
    name = ''.join(random.choices(string.ascii_lowercase, k=8))
    domain = ''.join(random.choices(string.ascii_lowercase, k=6))
    return f"{name}@{domain}.com"

def process_with_rwkv(text):
    if not model or not pipeline:
        return text
    
    try:
        # Подготовка контекста для модели
        context = f"""Задача: Проанализировать текст и определить, является ли информация конфиденциальной.
Правила анализа:
1. Конфиденциальная информация включает:
   - Имена собственные (ФИО, названия организаций)
   - ИИН (12 цифр)
   - Номера банковских карт (16 цифр)
   - Номера телефонов (+7 или 8, затем 10 цифр)
   - Email адреса
   - Адреса (содержат: улица, дом, квартира, город)
   - Паспортные данные
   - Даты рождения
   - Названия городов
   - Названия учебных заведений (школы, университеты)
   - Названия мест работы
   - Номера учреждений

2. НЕ является конфиденциальной:
   - Общие слова (например: "город", "школа", "работа")
   - Названия месяцев, дней недели
   - Общие географические названия (страны, континенты)
   - Названия профессий
   - Названия документов

3. Для каждого найденного значения:
   - Определи, является ли оно конфиденциальным
   - Если да - создай хеш и замени
   - Если нет - оставь без изменений

Текст для анализа: {text}

Результат анализа:"""
        output = pipeline.generate(context, max_tokens=1000, temperature=0.1, top_p=0.9)
        result = output.split("Результат анализа:")[-1].strip()
        if not result or result == text:
            return anonymize_text(text, use_rwkv=False)
            
        return result
    except Exception as e:
        print(f"Ошибка при обработке RWKV: {str(e)}")
        return anonymize_text(text, use_rwkv=False)
#.\venv\Scripts\activate 
def anonymize_text(text, use_rwkv=False):
    if use_rwkv and model:
        return process_with_rwkv(text)
    
    # Словарь для хранения хешей
    hashes = {}
    
    # Функция для создания уникального хеша
    def get_hash(match, data_type):
        if match.group(0) not in hashes:
            # Создаем хеш на основе типа данных и значения
            value = match.group(0)
            hash_input = f"{data_type}:{value}"
            # Используем SHA-256 для создания хеша
            hash_obj = hashlib.sha256(hash_input.encode())
            # Берем первые 8 символов хеша для краткости
            hashes[match.group(0)] = f"{data_type.upper()}_{hash_obj.hexdigest()[:8]}"
        return hashes[match.group(0)]
    
    # Паттерны для поиска персональных данных
    patterns = {
        'full_name': r'\b[А-Я][а-я]+\s+[А-Я][а-я]+\s+[А-Я][а-я]+\b',  # Полное ФИО
        'name_prefix_full': r'(?:^|\n|^|\s)(?:меня зовут|мое имя|имя|зовут|я)\s+([А-Я][а-я]+\s+[А-Я][а-я]+\s+[А-Я][а-я]+)',  # Фраза "Меня зовут" + полное ФИО
        'name': r'\b[А-Я][а-я]+\s+[А-Я][а-я]+\b',  # Имя и фамилия
        'name_prefix': r'(?:^|\n|^|\s)(?:меня зовут|мое имя|имя|зовут|я)\s+([А-Я][а-я]+)',  # Фраза "Меня зовут" + имя
        'iin': r'\b\d{12}\b',  # ИИН
        'bank_account': r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Номер банковской карты
        'phone': r'\b\+?[78][-\(]?\d{3}\)?-?\d{3}-?\d{2}-?\d{2}\b',  # Телефон
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        'address': r'\b(?:ул\.|улица|дом|кв\.|квартира|город|г\.)\s+[А-Яа-я0-9\s,.-]+\b',  # Адрес
        'passport': r'(?:Паспорт|паспорт):\s*\d{4}\s*\d{6}|\b\d{4}\s*\d{6}\b',  # Паспорт
        'birth_date': r'\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}\s+года\b',  # Дата рождения
        'birth_date_short': r'\b\d{2}[./]\d{2}[./]\d{4}\b',  # Дата рождения в формате ДД.ММ.ГГГГ
        'city': r'\b(?:город|г\.)\s+[А-Я][а-я]+\b',  # Город
        'school': r'\b(?:школа|школу|школе|школы)\s+(?:№\s*)?\d+\b',  # Школа
        'workplace': r'\b(?:работает в|работает|работаю в|работаю)\s+[А-Яа-я0-9\s,.-]+\b',  # Место работы
        'number': r'\b№\s*\d+\b',  # Номер (после знака №)
        'class': r'\b\d+[А-Я]?\s+(класс|классе)\b',  # Класс
        'parent': r'\b(?:мама|папа|отец|мать)\s+—\s+[А-Я][а-я]+\s+[А-Я][а-я]+\b',  # Родитель
        'city_name': r'\b[А-Я][а-я]+\b'  # Название города без префикса
    }
    
    try:
        # Сначала обрабатываем фразы с "Меня зовут" и полное ФИО
        result = text
        for data_type, pattern in patterns.items():
            if data_type in ['name_prefix_full', 'full_name']:
                result = re.sub(pattern, lambda m: get_hash(m, 'name'), result)
            elif data_type == 'name_prefix':
                result = re.sub(pattern, lambda m: f"{m.group(0).split()[0]} {get_hash(m.group(1), 'name')}", result)
            elif data_type == 'passport':
                result = re.sub(pattern, lambda m: f"Паспорт: {get_hash(m, 'passport')}", result)
            elif data_type == 'city_name':
                # Проверяем, что это действительно город, а не часть другого слова
                result = re.sub(pattern, lambda m: get_hash(m, 'city') if m.group(0).lower() in ['новосибирск', 'москва', 'санкт-петербург'] else m.group(0), result)
            else:
                result = re.sub(pattern, lambda m: get_hash(m, data_type), result)
        
        return result
    except Exception as e:
        print(f"Ошибка в anonymize_text: {str(e)}")
        return text

def process_txt_file(file):
    try:
        # Читаем файл
        text = file.read().decode('utf-8')
        
        # Анонимизируем текст
        result = anonymize_text(text)
        
        # Проверяем, что результат не пустой
        if not result.strip():
            return "Ошибка: Результат анонимизации пуст"
            
        return result
    except Exception as e:
        print(f"Ошибка в process_txt_file: {str(e)}")
        return f"Ошибка обработки TXT файла: {str(e)}"

def process_image(image_data, use_rwkv=False):
    try:
        # Удаляем префикс base64, если он есть
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Декодируем base64 в изображение
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Конвертируем в оттенки серого для лучшего распознавания
        if image.mode != 'L':
            image = image.convert('L')
        
        # Увеличиваем контраст
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Размываем изображение для защиты конфиденциальности
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
        
        # Генерируем уникальное имя файла
        timestamp = int(time.time())
        result_filename = f'blurred_image_{timestamp}.png'
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        # Сохраняем размытое изображение
        blurred_image.save(temp_path, 'PNG')
        
        # Извлекаем текст из изображения
        try:
            # Используем улучшенные параметры для Tesseract
            custom_config = r'--oem 3 --psm 6 -l rus+eng'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            if text.strip():
                # Очищаем текст от мусора
                text = re.sub(r'[^\w\s.,!?-]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                # Удаляем короткие слова и цифры
                text = ' '.join(word for word in text.split() if len(word) > 2 and not word.isdigit())
                # Удаляем специфические артефакты
                text = re.sub(r'Notporante|tpasy|tif', '', text, flags=re.IGNORECASE)
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    text = anonymize_text(text, use_rwkv)
        except Exception as e:
            print(f"Ошибка при извлечении текста из изображения: {str(e)}")
            text = None
        
        return {
            'text': text,
            'image_path': temp_path,
            'filename': result_filename
        }
    except Exception as e:
        print(f"Ошибка в process_image: {str(e)}")
        return f"Ошибка обработки изображения: {str(e)}"

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/anonymize', methods=['POST'])
def anonymize():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'Файл не выбран'})
            
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext == '.txt':
                result = process_txt_file(file)
                if isinstance(result, str) and result.startswith('Ошибка'):
                    return jsonify({'error': result})
                
                # Создаем файл с результатом
                timestamp = int(time.time())
                result_filename = f'anonymized_{timestamp}_{filename}'
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write(result)
                
                return jsonify({
                    'result': result,
                    'download_url': f'/download/{result_filename}',
                    'filename': result_filename
                })
                
            elif file_ext in ['.jpg', '.jpeg', '.png']:
                # Читаем файл в base64
                image_data = base64.b64encode(file.read()).decode('utf-8')
                result = process_image(image_data, use_rwkv=False)
                
                if isinstance(result, dict):
                    return jsonify({
                        'result': f'/download/{result["filename"]}',
                        'text': result['text'] if result['text'] else None,
                        'download_url': f'/download/{result["filename"]}',
                        'filename': result['filename']
                    })
                else:
                    return jsonify({'error': result})
            else:
                return jsonify({'error': 'Неподдерживаемый формат файла'})
            
        elif request.is_json:
            text = request.json.get('text', '')
            if not text:
                return jsonify({'error': 'Текст не предоставлен'})
            
            result = anonymize_text(text)
            if not result.strip():
                return jsonify({'error': 'Результат анонимизации пуст'})
            
            # Создаем файл с результатом
            timestamp = int(time.time())
            result_filename = f'anonymized_text_{timestamp}.txt'
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(result)
            
            return jsonify({
                'result': result,
                'download_url': f'/download/{result_filename}',
                'filename': result_filename
            })
            
        else:
            return jsonify({'error': 'Неверный формат запроса'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )
    except Exception as e:
        return jsonify({'error': str(e)})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['txt', 'jpg', 'jpeg', 'png']
        
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/api/chat', methods=['POST'])
def process_chat_message():
    try:
        data = request.get_json()
        message = data.get('message', '')
        file = request.files.get('file')
        
        if file:
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext == '.txt':
                result = process_txt_file(file)
            elif file_ext in ['.jpg', '.jpeg', '.png']:
                # Читаем файл в base64
                image_data = base64.b64encode(file.read()).decode('utf-8')
                result = process_image(image_data, use_rwkv=False)
                
                if isinstance(result, dict):
                    return jsonify({
                        'success': True,
                        'result': f'/download/blurred_image.png',
                        'text': result['text'] if result['text'] else None,
                        'explanation': 'Изображение обработано. Лица размыты, текст анонимизирован.'
                    })
            else:
                return jsonify({'error': 'Неподдерживаемый формат файла'}), 400
                
            if isinstance(result, str) and result.startswith('Ошибка'):
                return jsonify({'error': result}), 400
                
            return jsonify({
                'success': True,
                'result': result,
                'explanation': 'Файл успешно обработан. Все персональные данные заменены на безопасные идентификаторы.'
            })
            
        elif message:
            # Обрабатываем текстовое сообщение
            result = anonymize_text(message)
            
            # Добавляем пояснение к результату
            explanation = """
Я обработал ваш текст и заменил персональные данные на безопасные идентификаторы:
Такие как NAME_XXXX,PHONE_XXXX,EMAIL_XXXX,ADDRESS_XXXX,IIN_XXXX,BANK_XXXX,DATE_XXXX, PASSPORT_XXX
Теперь ваш текст безопасен для публикации. Вы можете скопировать его или скачать как файл.
"""
            
            return jsonify({
                'success': True,
                'result': result,
                'explanation': explanation
            })
        else:
            return jsonify({'error': 'Сообщение пустое'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 