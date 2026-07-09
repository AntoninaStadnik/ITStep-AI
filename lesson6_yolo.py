# Завдання 1
# Отримайте перший кадр з файлу data\lesson8\animals.mp4
# та виведіть його на екран.
# Проведіть детекцію об’єктів зо допомогою YOLO та
# виведіть результати.
# Змініть параметри моделі conf та iou і подивіться як це
# впливає на результат.
# Отримайте рамки для кожного об’єкта, виріжіть їх та
# виведіть як окремі зображення

import cv2
import ultralytics


model = ultralytics.YOLO("yolo11s.pt")

cap = cv2.VideoCapture(r'data\lesson8\animals.mp4')

success, img = cap.read()
img = cv2.resize(img, None, fx=0.5, fy=0.5)

#cv2.imshow("img", img)

results = model.predict(
    img,
    device= "cuda",
    conf=0.25,
    iou=0.7,
)
#print(result)

result = results[0]

res = result.plot()

cv2.imshow("result", res)

boxes = result.boxes
#
# box1 = boxes[0]
#
# #print(box1)
#
# cls = box1.cls
# print(cls)
#
# conf = box1.conf
# print(conf)
#
# xyxy = box1.xyxy
# print(xyxy)
#
# cls = cls.cpu().numpy()
# conf = conf.cpu().numpy()
# xyxy = xyxy.cpu().numpy().astype(int)
#
#
# x1, y1, x2, y2 = xyxy[0]
#
# box1_img = img[y1:y2, x1:x2]
#
# cv2.imshow("result", box1_img)
#
#
# names = result.names
# name1 = names[cls[0]]
# print(name1)
# print(conf[0])
#
# cv2.imshow(f"{name1}, {conf[0]*100 :.2f}", box1_img)


# for box in boxes:
#     boxes = result.boxes
#
#     box1 = boxes[0]
#
#     # print(box1)
#
#     cls = box.cls
#     #print(cls)
#
#     conf = box.conf
#     #print(conf)
#
#     xyxy = box.xyxy
#     #print(xyxy)
#
#     cls = cls.cpu().numpy()
#     conf = conf.cpu().numpy()
#     xyxy = xyxy.cpu().numpy().astype(int)
#
#     x1, y1, x2, y2 = xyxy[0]
#
#     box1_img = img[y1:y2, x1:x2]
#
#     cv2.imshow("result", box1_img)
#
#     names = result.names
#     name1 = names[cls[0]]
#     #print(name1)
#     #print(conf[0])
#
#     cv2.imshow(f"{name1}, {conf[0] * 100 :.2f}", box1_img)

while True:
    success, img = cap.read()

    if not success:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    img = cv2.resize(img, None, fx=0.5, fy=0.5)

    results = model.predict(
        img,
        device="cuda",
        conf=0.25,
        iou=0.7,
    )

    result = results[0]
    boxes = result.boxes
    for box in boxes:
        boxes = result.boxes
        box1 = boxes[0]
        cls = box.cls
        conf = box.conf
        xyxy = box.xyxy
        cls = cls.cpu().numpy()
        conf = conf.cpu().numpy()
        xyxy = xyxy.cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy[0]
        box1_img = img[y1:y2, x1:x2]
        cv2.imshow("result", box1_img)



cv2.waitKey(1000)
cv2.destroyAllWindows()