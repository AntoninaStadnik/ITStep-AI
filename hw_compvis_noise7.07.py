# Завдання 1
# Відкрийте зображення data/lesson3/sonet.png. Проведіть
# бінарізацію.
# Обов’язково використайте:
#  розмиття або наведення різкості
#  адаптивну бінарізацію
#  очищеня шумів
import cv2

# image = cv2.imread("data/lesson3/sonet.png")
# cv2.imshow("Original", image)
#
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray", gray)
#
#
# blur = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow("Blur", blur)
#
#
# binary = cv2.adaptiveThreshold(
#     blur,
#     255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     21,
#     6
# )
# cv2.imshow("Adaptive Threshold", binary)
#
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# cv2.imshow("Noise Removed", clean)
#
# cv2.waitKey(0)




# Завдання 2
# Відкрийте зображення data/lesson3/sonnet_noised.png.
# Проведіть бінарізацію. Застосуйте код з завдання 1 та
# спробуйте покращити результат
image = cv2.imread("data/lesson3/sonet_noised.png")
cv2.imshow("Original", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (7, 7), 0)


binary = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    21,
    6
)
cv2.imshow("Adaptive Threshold", binary)


cv2.waitKey(0)