# Завдання 1
# Відкрийте зображення data/lesson_seg/crop3.jpg
# Проведіть сегментацію зображення використовуючи
# модель data/lesson_seg/crop-seg.jpg
# Покажіть усі маски рослин з підписами назви цієї
# рослини.
# Покажіть також самі рослини, для цього застосуйте
# маску, і всі зайві пікселі замініть на 255(зробити білий фон)

import cv2
from ultralytics import YOLO
import numpy as np

# img = cv2.imread('data/lesson_seg/crop3.jpg')
#
# cv2.imshow('original', img)
#
# model = YOLO("data/lesson_seg/crop-seg.pt")
#
# results = model.predict(
#     img,
#     device="cuda"
# )
# result = results[0]
#
# res = result.plot()
# cv2.imshow('result', res)
#
# print(result)
#
# masks = result.masks
# print(masks)
#
# masks_data = masks.data
# masks_data = masks_data.cpu().numpy()
#
#
# height, width, color = img.shape
#
# for i in range(len(masks_data)):
#     mask = masks_data[i]
#     mask = cv2.resize(mask, (width, height))
#
#     mask = mask.astype(bool)
#     # mask_uint = mask.astype(np.uint8)
#     # mask_uint *= 255
#
#     # cv2.imshow("mask", mask_uint)
#
#     plant = img.copy()
#     plant[~mask] = 255
#     cv2.imshow(f"Plant{i}", plant)
#
# cv2.waitKey(0)


# Завдання 2
# Відкрийте зображення data/lesson_seg/crop3.jpg
# Проведіть сегментацію зображення
# Порахуйте розмір кожної рослини(площа маски)
# Покажіть найбільшу рослину кожного виду

img = cv2.imread('data/lesson_seg/crop3.jpg')

model = YOLO("data/lesson_seg/crop-seg.pt")

results = model.predict(
    img,
    device="cuda"
)
result = results[0]

res = result.plot()
cv2.imshow('result', res)

# print(result)

cv2.imshow('original', img)

masks = result.masks
# print(masks)

masks_data = masks.data
masks_data = masks_data.cpu().numpy()
mask_list = []

for mask in masks_data:
    mask_sum = mask.sum()
    mask_list.append(mask_sum)

    # print(mask_sum)
print(mask_list)

biggest_mask = max(mask_list)
print(biggest_mask)

for i in range(len(mask_list)):
    if biggest_mask == mask_list[i]:
        break

print(i)

mask3 = masks_data[i]

mask3_uint = mask3.astype(np.uint8)
mask3_uint *= 255


cv2.imshow("mask", mask3_uint)

cv2.waitKey(0)