from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from model.get_result import process_image  # Импорт функции обработки изображения нейросетью

app = Flask(__name__)

# Настройки для загрузки файлов
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Проверка, был ли файл загружен и валиден ли он
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Обработка изображения нейросетью
            result = process_image(filepath)

            return render_template('result.html', result=result, image=filepath)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
