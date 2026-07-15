# Завдання 1
# Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та виведіть результат, підберіть
# параметри
# Можете змінити розмір кадру для кращої візуалізації
# cv2.resize()

import cv2
import ultralytics

model = ultralytics.YOLO("yolo11s.pt")

cap = cv2.VideoCapture(r'data\lesson8\meetings.mp4')
#
#
# while True:
#     success, frame = cap.read()
#
#     if not success:
#         break
#
#     frame_resize = cv2.resize(frame, None, fx=0.3, fy=0.3)
#
#     results = model.predict(
#         frame_resize,
#         device="cpu",
#         conf=0.25,
#         iou=0.5,
#
#     )
#
#     result = results[0]
#
#     res_img = result.plot()
#     cv2.imshow("result", res_img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# Завдання 2
# Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та почніть показувати відео з
# моменту, коли людей стало 5

start_showing = False

while True:
    success, frame = cap.read()

    if not success:
        break

    frame_resize = cv2.resize(frame, None, fx=0.3, fy=0.3)

    results = model.predict(frame, verbose=False, conf=0.25)
    result = results[0]

    detected_classes = result.boxes.cls.int().tolist()

    people_count = detected_classes.count(0)

    if people_count >= 5:
        start_showing = True

    if not start_showing:
        continue

    cv2.imshow("Video Content", result.plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()