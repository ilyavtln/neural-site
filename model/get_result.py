from utils.models import ConvNS
import torch
import os
import torch.nn.functional as F
import numpy as np
import cv2

# Путь до обученной модели
model_save_path = 'model/trained/mnist_convnet_10.pth'

loaded_model = ConvNS()
loaded_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
loaded_model = loaded_model.to('cpu')


def process_image(filepath):
    cropped_images = preprocess_image(filepath)
    digits = []
    for image in cropped_images:
        digits.append(process_image_by_img(image))

    digits_on_img = ''.join(map(str, digits))
    return f"На картинке введено {digits_on_img}"


def process_image_by_img(test_img):
    test_img = cv2.resize(test_img, (28, 28), interpolation=cv2.INTER_AREA)
    # Инвертируем картинку
    test_img = cv2.bitwise_not(test_img)

    test_img = np.expand_dims(test_img, axis=0)
    test_img = np.expand_dims(test_img, axis=0)
    # Нормальзуем данные
    test_img = test_img.astype(np.float32) / 255.0

    torch_img = torch.from_numpy(test_img)
    test_model = loaded_model(torch_img)

    # Определяем число с картинки
    predict = F.softmax(test_model).detach().numpy().argmax()

    return predict


def preprocess_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # Фильтр для выделения чисел
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    # Найдите контуры на изображении
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Сортируем в порядке слева направо
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    cropped_img = []
    for contour in contours:
        # Получаем координаты контуров
        x, y, w, h = cv2.boundingRect(contour)
        # Отступы для цифр
        x, y, w, h = x - 2, y - 2, w + 4, h + 4
        x = max(x, 0)
        y = max(y, 0)
        # Обрезка изображения
        cropped = img[y:y + h, x:x + w]
        # Размер для нейросети
        resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
        cropped_img.append(resized)

    return cropped_img

