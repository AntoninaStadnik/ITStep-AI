# Завдання 1
# Виведіть відео з файлу data\lesson7\text.mp4 на екран та
# збережіть в новий файл.
# Змініть розмір зображення.

import cv2

# cap = cv2.VideoCapture(r'data\lesson7\text.mp4')
#
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# print(width)
# print(height)
#
# # FPS -- кількість кадрів у секунду
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# print(fps)
#
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out_writer = cv2.VideoWriter(
#     "result.mp4",  # файл куди зберігати відео
#     fourcc,  # кодек
#     fps,  # частота кадрів в секунду
#     (500, 500),  # розмір (ширина, висота)
#     isColor=True,  # чи є зображення кадрів кольоровими
# )
#
# while True:
#     success, frame = cap.read()
#
#     if not success:
#         break
#
#     new_frame =  cv2.resize(
#         frame,
#         (500, 500)
#                )
#
#     cv2.imshow("new_frame", new_frame)
#
#
#
#     out_writer.write(new_frame)
#
#     cv2.waitKey(40)
#
# out_writer.release()
# cap.release()

# Завдання 2
# Відкрийте відео з файлу data\lesson7\text.mp4. Проведіть
# бінарізацію кадрів та збережіть в новий файл.

# cap = cv2.VideoCapture(r"data\lesson7\text.mp4")
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(
#     r"data\lesson7\text_binary.mp4",
#     fourcc,
#     30,
#     (500, 500),
#     isColor = False
# )
#
# while True:
#     success, frame = cap.read()
#
#     if not success:
#         break
#
#
#     new_frame = cv2.resize(frame, (500, 500))
#     cv2.imshow("Original", new_frame)
#
#     gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
#
#
#     binary = cv2.adaptiveThreshold(
#         gray,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         3,
#         7
#     )
#     cv2.imshow("Binary", binary)
#
#
#     out.write(binary)
#
#     cv2.waitKey(40)
#
# cap.release()
# out.release()

# Завдання 3
# Відкрийте відео з файлу data\lesson7shapes.mp4.
# Проведіть виділення кольорів на кадрах та збережіть в новий
# файл.

cap = cv2.VideoCapture(r"data\lesson7\shapes.mp4")

while True:
    success, frame = cap.read()

    if not success:
        break

    new_frame = cv2.resize(frame, (600, 600))

    hsv = cv2.cvtColor(new_frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    lower_green = (40, 80, 50)
    upper_green = (65, 255, 255)

    mask = cv2.inRange(hsv, lower_green, upper_green)
    cv2.imshow("mask", mask)


    #cv2.imshow("new_frame", new_frame)
    cv2.waitKey(20)
