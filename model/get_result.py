from utils.models import ConvNS, ConvNSExpr
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import cv2
from sympy import *
import re

# Путь до обученной модели
model_save_path = './model/trained/expr_convnet25.pth'

# loaded_model = ConvNS()
loaded_model = ConvNSExpr()
loaded_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
loaded_model = loaded_model.to('cpu')

loaded_root_classes = list(np.loadtxt('./utils/root_classes.txt', dtype=str))


def process_image(filepath):
    cropped_images = preprocess_image(filepath)
    digits = []
    for image in cropped_images:
        digits.append(process_image_by_img(image))

    digits_on_img = ''.join(map(str, digits))

    digits_on_img = digits_on_img.replace('--', '=').lower()

    return f"На картинке введено {digits_on_img}"


def add_mult_for_expr(expr: str, variable: str):
    variable = re.escape(variable)
    expr = re.sub(rf'(\d)({variable}|\()', r'\1*\2', expr)
    expr = re.sub(rf'(\))({variable}|\()', r'\1*\2', expr)
    return expr


def solve_expr(expression, variable):
    # Разделяем выражение на левую и правую части
    left, right = expression.split('=')

    left_expr = sympify(left)
    right_expr = sympify(right)

    equation = Eq(left_expr - right_expr, 0)

    my_var = symbols(variable)

    return solve(equation, my_var)


def process_expr(filepath):
    cropped_images = preprocess_image(filepath)
    recognized_expr = ''
    variable = ''
    for image in cropped_images:
        processed_symbol = process_image_by_img(image).lower()
        # Определяем переменную
        if processed_symbol in 'xyz':
            variable = processed_symbol
        recognized_expr += processed_symbol

    if variable == '':
        return 'Некорректный ввод уравнения'

    recognized_expr = recognized_expr.replace('--', '=')

    recognized_expr = add_mult_for_expr(recognized_expr, variable)

    # Решаем уравнение
    solution = solve_expr(recognized_expr, variable)
    return f'{variable} = {solution[0]}'


def to_plot_expr(filepath):
    cropped_images = preprocess_image(filepath)
    recognized_expr = ''
    variable = ''
    for image in cropped_images:
        processed_symbol = process_image_by_img(image).lower()
        # Определяем переменную
        if processed_symbol in 'xz':
            variable = processed_symbol
        recognized_expr += processed_symbol

    if variable == '':
        return 'Некорректный ввод уравнения'

    recognized_expr = recognized_expr.replace('--', '=')
    new_name = str(recognized_expr) + '.png'

    recognized_expr = add_mult_for_expr(recognized_expr, variable)

    # Разделяем выражение на левую и правую части
    left, right = recognized_expr.split('=')

    variable = symbols(variable)
    y_expr = sympify(right)

    y_func = lambdify(variable, y_expr, 'numpy')

    # Генерация значений x от -10 до 10
    x_values = np.linspace(-10, 10, 400)
    # Вычисление соответствующих значений y
    y_values = y_func(x_values)

    # Построение графика с использованием Matplotlib
    plt.plot(x_values, y_values, label=str(recognized_expr))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('График функции ' + recognized_expr)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.savefig('static/plotted/' + new_name)

    return new_name


def process_image_by_img(test_img):
    test_img = cv2.resize(test_img, (64, 64), interpolation=cv2.INTER_AREA)

    test_img = np.expand_dims(test_img, axis=0)
    test_img = np.expand_dims(test_img, axis=0)
    # Нормальзуем данные
    test_img = test_img.astype(np.float32) / 255.0

    torch_img = torch.from_numpy(test_img)
    test_model = loaded_model(torch_img)

    # Определяем значение
    idx = F.softmax(test_model.to("cpu")).detach().numpy().argmax()
    predict = loaded_root_classes[idx]

    return predict


def preprocess_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Фильтр Гаусса для сглаживания изображения
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Выделение контуров
    edges = cv2.Canny(blurred, 120, 150)

    # Находим контуры символов
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Сортируем контуры в порядке слева направо
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    cropped_img = []
    for contour in contours:
        # Получаем координаты контуров
        x, y, w, h = cv2.boundingRect(contour)
        # Отступы для символов
        x, y, w, h = x - 2, y - 2, w + 4, h + 4
        x = max(x, 0)
        y = max(y, 0)
        # Обрезка изображения
        cropped = img[y:y + h, x:x + w]
        # Размер для нейросети
        resized = cv2.resize(cropped, (64, 64), interpolation=cv2.INTER_AREA)
        cropped_img.append(resized)

    return cropped_img
