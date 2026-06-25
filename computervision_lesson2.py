# Відкрийте зображення data/Lenna.png. Виведіть на екран
# розмір зображення, тип даних, максимальну та мінімальну
# інтенсивність пікселів, саме зображення з підписом.

import cv2
import numpy as np
from pandas.core.dtypes import astype

image = cv2.imread(
    "data/lesson1/Lenna.png",
    cv2.IMREAD_GRAYSCALE,
)

cv2.imshow("Lenna", image)
# print(image.shape)
# print(image.dtype)
# print(image.max())
# print(image.min())
#
#cv2.waitKey(0)

# Відкрийте зображення data/Lenna.png. Виведіть на екран
# такі зображень:

#  Верхній лівий кут розміром 100х50
# segment = image[0:100, 0:50]
# cv2.imshow("Lenna", segment)
# cv2.waitKey(0)

#  Центральний квадрат розміром 100х100
# segment = image[78:178, 78:178]
# cv2.imshow("Lenna", segment)
# print(segment.shape)
# cv2.waitKey(0)

#  Верхню половину
# cv2.imshow("upside", image[: 128, :])
# cv2.waitKey(0)

#  Нижню половину
# cv2.imshow("bottom", image[129:255, :])
# cv2.waitKey(0)

#  Ліву половину
# cv2.imshow("LeftSide", image[:, :128])
# cv2.waitKey(0)

#  Праву половину
# cv2.imshow("RigtSide", image[:, 129:255])
# cv2.waitKey(0)

# Відкрийте зображення data/Lenna.png. Створіть наступні
# зображення
# image [:20, :] = 0
# image[235:256, :] = 255
# cv2.imshow("Lenna", image)
# cv2.waitKey(0)

# image[:, 0:20] = 0
# image[:, 240:257] = 0
# cv2.imshow("Lenna", image)
# cv2.waitKey(0)

# image [:20, :] = 0
# image [:, :20] = 0
# image[:, 240:257] = 0
# image[240:257, :] = 0
# cv2.imshow("Lenna", image)
# cv2.waitKey(0)

# Відкрийте зображення data/Lenna.png. Створіть маску для
# пік селів з інтенсивністю більше 128 та виведіть її. Також
# виведіть заперечення цієї маски.
# На оригінальному зображенні, усі пікселі які не
# відповідають масці замініть на 0 та виведіть результат
# mask = image > 128
#
# new_mask = mask.astype(np.uint8)
#
# print(mask)
#
# cv2.imshow("mask", new_mask * 255)
#
# image[~mask] = 0
#
# cv2.imshow("New_image", image)
#
#
# cv2.waitKey()

result = (image / 255) ** 0.4 * 255 # gamma param (operation with every pixel)
result = result.astype(np.uint8)

cv2.imshow("image", result)

cv2.waitKey()




