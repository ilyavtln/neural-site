from flask import Flask, request, render_template, redirect
import os
from werkzeug.utils import secure_filename
from model.get_result import process_image, process_expr, to_plot_expr
import templates.configs.advantages as cfg

app = Flask(__name__)

# Настройки для загрузки файлов
UPLOAD_FOLDER = 'static/uploads/'
PLOTTED_FOLDER = 'static/plotted/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOTTED_FOLDER'] = PLOTTED_FOLDER

menu_path = 'components/menu/menu.html'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html', menu_path=menu_path, config=cfg)


@app.route('/services/extract', methods=['GET', 'POST'])
def services_extract():
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

            return render_template('result.html',
                                   result=result,
                                   image='/' + filepath,
                                   from_page='services_extract',
                                   menu_path=menu_path)
    return render_template('services/extract.html', menu_path=menu_path)


@app.route('/services/solve_expr', methods=['GET', 'POST'])
def services_solve_expr():
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
            result = process_expr(filepath)

            return render_template('result.html', result=result, image='/' + filepath, from_page='services_solve_expr',
                                   menu_path=menu_path)
    return render_template('services/solve_expr.html', menu_path=menu_path)


@app.route('/services/plot', methods=['GET', 'POST'])
def services_plot_image():
    if request.method == 'POST':
        # Проверка, был ли файл загружен и валиден ли он
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['PLOTTED_FOLDER'] + '/saved', filename)
            file.save(filepath)

            # Обработка изображения нейросетью
            result = to_plot_expr(filepath)

            plot_path = str(os.path.join(app.config['PLOTTED_FOLDER'], result))

            return render_template('result.html', result=result, image='/' + plot_path, from_page='services_plot_image',
                                   menu_path=menu_path)
    return render_template('services/plot.html', menu_path=menu_path)


@app.route('/uploaded')
def show_uploaded():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('uploaded/uploaded.html', image_files=images, menu_path=menu_path)


@app.route('/plotted')
def show_plotted():
    images = os.listdir(app.config['PLOTTED_FOLDER'])
    return render_template('plotted/plotted.html', image_files=images, menu_path=menu_path)


if __name__ == '__main__':
    app.run(debug=True)
