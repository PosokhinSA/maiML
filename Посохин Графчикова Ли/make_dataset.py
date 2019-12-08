import os

import cv2

import numpy as np

if __name__ == '__main__':
    path_to_images = 'edited_images/'
    images = []
    print('Файлы которые будут обработаны:')
    for file_ in os.listdir(path_to_images):
        print(f'\t {path_to_images}{file_}')
        images.append(cv2.imread(path_to_images + file_, cv2.IMREAD_GRAYSCALE))
    print('Данные загружены. Выполняю нормализацию...')
    for i in range(len(images)):
        images[i] //= 255
    print('Данные нормализованы. Создаю датасет...')
    y = np.zeros(shape=(300 * len(images)), dtype=np.uint8)
    X = np.zeros(shape=(300 * len(images), 32 * 32), dtype=np.uint8)
    for image_num, image in enumerate(images):
        sub_images = 0
        for y_c in range(0, image.shape[0], 32):
            for x_c in range(0, image.shape[1], 32):
                X[300 * image_num + sub_images] = image[y_c:y_c+32, x_c:x_c+32].reshape(32 * 32)
                y[300 * image_num + sub_images] = sub_images // 100
                sub_images += 1
        print(f'\tОбработано {(image_num + 1) * 300} из {len(images) * 300}')
    np.savetxt('features.data', X, fmt='%d')
    print('Матрица признаков записана в файл "features.data"')
    np.savetxt('target.data', y, fmt='%d')
    print('Целевая переменная записана в файл "target.data"')
