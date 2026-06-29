# Відкрийте зображення data/Lenna.png. Прочитайте маски
# data/mask1.png та data/mask2.png.
# Об’єднайте дві маски в одну, скористайтесь cv2.bitwise_or()
# та виведіть результат
# Виведіть ту частину зображення, яка відповідає:
#  mask1
#  mask2
#  mask1 і mask2
# Усі пікселі які не відповідають маскам замінити на 0, перед
# застосуванням змініть тип даних у масці на bool
import numpy as np
import cv2
image = cv2.imread(
    "data/lesson1/Lenna.png",
    cv2.IMREAD_GRAYSCALE,
)

mask1 = cv2.imread("data/lesson1/mask1.png")
mask2 = cv2.imread("data/lesson1/mask2.png")

mask_bool = mask1 > 0
mask_bool2 = mask2 > 0
print(mask_bool)
print(mask_bool2)

result =  cv2.bitwise_or(mask1, mask2)
print(result)

cv2.imshow(
    "original",
    result
)

# Виведіть зображення. Підберіть самостійно межі
# image = cv2.imread(
#     "data/lesson1/baboo.jpg",  # шлях до файлу
#     cv2.IMREAD_GRAYSCALE,   # прапорець як читати зображення(чорнобіле)
# )
#
# cv2.imshow(
#     "original",
#     image
# )
#
# segment = image[0:80, 30:200]
# print(segment.shape)
# print(segment.dtype)
# cv2.imshow("segment", segment)
#
cv2.waitKey(0)

