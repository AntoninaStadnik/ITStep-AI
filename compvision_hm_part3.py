# Завдання 1
# Відкрийте зображення data\lesson2\darken.png. Проведіть з
# ним наступні операції, переведіть його в HSV формат та
# обробіть канал Value наступними способами:
#  застосуйте вирівнювання гістограм
#  збільшіть значення десь на 20-50%, оскільки тут
# результат буде типу float32 та явно вийде за межі [0-255]
# застосуйте np.clip(value, 0, 255) та value.astype(np.uint8)
# Виведіть результати обох обробок на екран
import cv2
import numpy as np


image = cv2.imread(
    "data/lesson2/darken.png"
)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow("original", image )

h, s, v = cv2.split(hsv)

v = v.astype(np.float32)

v = v * 1.5

np.clip(v, 0, 255)

v = v.astype(np.uint8)

new_hsv = cv2.merge((h, s, v))

new_image = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("result_v", new_image)


v_eq = cv2.equalizeHist(v)

eq_hsv = cv2.merge((h, s, v_eq))
eq_image = cv2.cvtColor(eq_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow("equalized", eq_image)

cv2.waitKey(0)