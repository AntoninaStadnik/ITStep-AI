# Відкрийте відео data/lesson_pose/squat.mp4
# Ваша задача рахувати кількість присідань.
# Отримайте перший кадр та виділіть основні точки.
# Отримайте координати 3-ох точок ноги
# Визначте кут між цими трьома точками. Скористайтесь
# функцією utils.get_angle(x1, y1, x2, y2, x3, y3) де x2, y2 –
# координати коліна(центральна точка)
# Запустіть відео та добавте на сам кадр кут згинання ніг.
# Визначіть нижню межу кута(якщо людина опустилась
# нижче вважаємо що вона достатньо опустилась) та верхню
# межу кута(якщо людина піднялась вище вважаємо що вона
# достатньо піднялась)
# Добавте кількість присідань та
# кут на кожен кадр.
import cv2
import ultralytics
import utils

cap = cv2.VideoCapture(r'data/lesson_pose/squat.mp4')
success, img = cap.read()

# cv2.imshow('frame', img)

model = ultralytics.YOLO("yolo11s-pose.pt")

results = model.predict(
    img,
    device="cpu"
)

result = results[0]

print(result)
result_img = result.plot()

# cv2.imshow('result', result_img)

frame = cv2.resize(result_img, None, fx=0.5, fy=0.5)

keypoints = result.keypoints
print(keypoints)

xy = keypoints.xy
xy = xy.cpu().numpy()

print(xy)
print(xy.shape)
print(xy.dtype)

xy = xy[0]
xy = xy.astype(int)

x_left_hip, y_left_hip = xy[11]
x_left_knee, y_left_knee = xy[13]
x_left_ankle, y_left_ankle = xy[15]

angle = utils.get_angle(
    x_left_hip,
    y_left_hip,
    x_left_knee,
    y_left_knee,
    x_left_ankle,
    y_left_ankle
)

print("Angle:", angle)

cv2.circle(
    img,
    center=(x_left_hip, y_left_hip),
    radius=15,
    color=(255, 0, 0),
    thickness=-1,
)

cv2.circle(
    img,
    center=(x_left_knee, y_left_knee),
    radius=15,
    color=(0, 255, 0),
    thickness=-1,
)

cv2.circle(
    img,
    center=(x_left_ankle, y_left_ankle),
    radius=15,
    color=(0, 0, 255),
    thickness=-1,
)


total_siting = 0

is_sitting = False

lower_angle = 90
upper_angle = 160


while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    results = model.predict(
        frame,
        device="cpu"

    )

    result = results[0]

    keypoints = result.keypoints

    xy = keypoints.xy
    xy = xy.cpu().numpy()

    if len(xy) == 0:
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        continue

    xy = xy[0]
    xy = xy.astype(int)

    x_left_hip, y_left_hip = xy[11]
    x_left_knee, y_left_knee = xy[13]
    x_left_ankle, y_left_ankle = xy[15]

    angle = utils.get_angle(
        x_left_hip,
        y_left_hip,
        x_left_knee,
        y_left_knee,
        x_left_ankle,
        y_left_ankle
    )

    cv2.circle(
        frame,
        center=(x_left_hip, y_left_hip),
        radius=10,
        color=(255, 0, 0),
        thickness=-1
    )

    cv2.circle(
        frame,
        center=(x_left_knee, y_left_knee),
        radius=10,
        color=(0, 255, 0),
        thickness=-1
    )

    cv2.circle(
        frame,
        center=(x_left_ankle, y_left_ankle),
        radius=10,
        color=(0, 0, 255),
        thickness=-1
    )

    if angle < lower_angle:
        is_sitting = True

    if angle > upper_angle and is_sitting:
        total_siting += 1
        is_sitting = False

    cv2.putText(
        frame,
        f"Angle: {int(angle)}",
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Sitting_total: {total_siting}",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()