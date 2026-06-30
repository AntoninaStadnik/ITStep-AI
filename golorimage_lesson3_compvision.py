# Завдання 1
# Відкрийте зображення data/lesson2/marbles.png.
# Використайте кольорову сегментацію для отримання масок до
# кульок:
#  усіх кульок
from email.mime import image

import cv2

#from lesson3 import hsv

# img = cv2.imread('data/lesson2/marbles.png')
# img = cv2.resize(img, (500, 500))
#
# cv2.imshow("image", img)

# lower = (100, 120, 120)
# upper = (130, 255, 255)
# mask = cv2.inRange(img, lower, upper)
#
# #  синього кольору
#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# mask = cv2.inRange(hsv, lower, upper)
#
# cv2.imshow("mask", mask)
#
# cv2.waitKey(0)

#  зеленого і червоного
# lower = (15, 100, 150)
# upper = (7, 255, 255) #червоного
# mask_red = cv2.inRange(hsv, lower, upper)
# cv2.imshow("mask_red", mask_red)
#
# lower = (35, 90, 80)
# upper = (90, 255, 255) #зеленого
# mask_green = cv2.inRange(hsv, lower, upper)
# cv2.imshow("mask_green", mask_green)

#  чорного
# lower = (0, 0, 0)
# upper = (100, 100, 40)
# mask_black = cv2.inRange(hsv, lower, upper)
#
# cv2.imshow("mask_black", mask_black)

#  білого
# lower = (0, 0, 200)
# upper = (180, 30, 255)
# mask_white = cv2.inRange(hsv, lower, upper)
#
# cv2.imshow("mask_white", mask_white)
#
#
# cv2.waitKey(0)


# Відкрийте зображення data/lesson2/cell.png. Покращте
# зображення за допомогою вирівнювання гістограми. Оскільки
# зображення кольорове, вам доведеться зробити наступні
# кроки:
#  перевести зображення в LAB
#  розбити зображення на канали l, a та b
#  вирівняти гістограму для l
#  зібрати канали назад в зображення
#  перевести результат назад в BGR
# Порівняйте результати для 2 алгоритмів.
img = cv2.imread('data/lesson2/cell.png')
img = cv2.resize(img, (500, 500))

cv2.imshow("image", img)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab)

new_l = cv2.equalizeHist(l)

new_lab = cv2.merge((new_l, a, b))

new_img = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)

cv2.imshow("updated_img", new_img)

cv2.waitKey(0)