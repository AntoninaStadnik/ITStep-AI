# Відкрийте зображення data/lesson_seg/tumor1.jpg
# Проведіть сегментацію зображення використовуючи
# модель data/lesson_seg/brain-tumor-seg.jpg
# Визначте площу пухлини в пікселях.
# Визначте площу в
# (1 піксель – 0,0025)
# В залежності від площі присвойте пухлині певний тип
#  <10 – small
#  10-25 – middle
#  >25 – large
# Покажіть пухлину – за допомогою маски усі лишні
# пікселі зробіть 0, а як назву зображення використайте її тип

import cv2
import ultralytics
from ultralytics import YOLO
import numpy as np

model = YOLO("data/lesson_seg/brain-tumor-seg.pt")
img = cv2.imread("data/lesson_seg/tumor1.jpg")

cv2.imshow("orig", img)

results = model.predict(
    img,
    device="cuda"
)
result = results[0]

res = result.plot()
cv2.imshow("result", res)

print(result)


masks = result.masks
print(masks)

masks_data = masks.data
masks_data = masks_data.cpu().numpy()

mask = masks_data[0]

mask_bool = mask.astype(bool)

mask_uint = mask.astype(np.uint8)
mask_uint *= 255

# cv2.imshow("mask", mask_uint)

img[~mask_bool] = 0
cv2.imshow("with mask", img)

pixel_area = 0.0025
mask_area = np.count_nonzero(mask_bool)

area = pixel_area * mask_area

if area < 10:
    cv2.imshow("small", img)
elif area <= 25:
    cv2.imshow("middle", img)
else:
    cv2.imshow("large", img)

cv2.waitKey(0)
cv2.destroyAllWindows()