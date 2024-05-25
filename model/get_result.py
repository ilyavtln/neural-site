from model.models import ConvNS
import torch
import torch.nn.functional as F
import numpy as np
import cv2

# Путь до обученной модели
model_save_path = 'model/trained/mnist_convnet.pth'

loaded_model = ConvNS()
loaded_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
loaded_model = loaded_model.to('cpu')


def process_image(filepath):
    # Считываем полученное изображение
    test_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
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

    result = f'Я думаю, на картинке число {predict}'

    return result
