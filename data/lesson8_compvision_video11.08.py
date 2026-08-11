# Відкрийте відео data/lesson_pose/sitting.mp4
# Отримайте перший кадр
# Покажіть його, за потреби змініть розмір
import cv2
import ultralytics

cap = cv2.VideoCapture(r'C:\Users\GamePC\Desktop\python lessons\AI\ITStep-AI\data\lesson_pose\sitting.mp4')
success, img = cap.read()

cv2.imshow('frame', img)


cv2.waitKey(0)


# Застосуйте модель YOLO Pose
# Отримайте результати (result) та виведіть їх на екран
# Використайте параметри device

model = ultralytics.YOLO("yolo11s-pose.pt")

results = model.predict(
    img,
    device= "cuda"
)

result = results[0]

print(result)

# Користуючись методом plot() отримайте зображення з
# рамками та підписами і покажіть його.

result_img = result.plot()

cv2.imshow('result', result_img)
cv2.waitKey(0)

# Отримайте інформацію про ключові точки(keypoints)
# ● Виведіть її на екран
# ● Отримайте координати точок(xy)
# ● Виведіть координати на екран разом з типом даних та
# розміром(позбудьтесь тензорів за допомогою cpu() та
# numpy())

keypoints = result.keypoints
print(keypoints)

xy = keypoints.xy
xy = xy.cpu().numpy()

print(xy)
print(xy.shape)
print(xy.dtype)

# Отримайте координати для лівого коліна, лівої руки,
# правої руки для першого об’єкта
# ● Намалюйте ці точки на зображенні:
# ○ ліве коліно – зелений
# ○ ліва рука – червоний
# ○ права рука – білий

xy = xy[0]

xy = xy.astype(int)

x_left_knee, y_left_knee = xy[14]
x_left_hand, y_left_hand = xy[10]
x_right_hand, y_right_hand = xy[9]
x_right_knee, y_right_knee = xy[13]

cv2.circle(
    img,   # зображення де малювати коло
    center=(x_left_hand, y_left_hand),   # координати центру
    radius=15,   # радіус в пікселях
    color=(0, 0, 255),  # колір в BGR(синій)
    thickness=-1,   # товщина ліній, -1 означає повністю заповнити кольором
)

cv2.circle(
    img,   # зображення де малювати коло
    center=(x_left_knee, y_left_knee),   # координати центру
    radius=15,   # радіус в пікселях
    color=(0, 255, 0),  # колір в BGR(синій)
    thickness=-1,   # товщина ліній, -1 означає повністю заповнити кольором
)

cv2.circle(
    img,   # зображення де малювати коло
    center=(x_right_knee, y_right_knee),   # координати центру
    radius=15,   # радіус в пікселях
    color=(255, 0, 0),  # колір в BGR(синій)
    thickness=-1,   # товщина ліній, -1 означає повністю заповнити кольором
)

cv2.imshow('img', img)
cv2.waitKey(0)


# Для кожного кадру на відео намалюйте координати для
# лівого коліна, лівої руки, правої руки
# Беріть координати для першого об’єкта

# Для кожного кадру на відео намалюйте координати для
# лівого коліна, лівої руки, правої руки
# Беріть координати для першого об’єкта

total_siting = 0
is_sitting = True


while True:
    success, frame = cap.read()

    if not success:
        break

    results = model.predict(
        frame,
        device="cuda"
    )

    result = results[0]

    keypoints = result.keypoints

    xy = keypoints.xy
    xy = xy.cpu().numpy()


    xy = xy[0].astype(int)


    x_left_knee, y_left_knee = xy[14]


    x_left_hand, y_left_hand = xy[10]


    x_right_hand, y_right_hand = xy[9]

    cv2.circle(
        frame,
        (x_left_knee, y_left_knee),
        15,
        (0, 255, 0),
        -1
    )

    cv2.circle(
        frame,
        (x_left_hand, y_left_hand),
        15,
        (0, 0, 255),
        -1
    )


    cv2.circle(
        frame,
        (x_right_hand, y_right_hand),
        15,
        (255, 255, 255),
        -1
    )

    if y_right_knee < y_right_hand and is_sitting:
        total_siting += 1

    if y_right_knee < y_right_hand:
        is_sitting = False
    else:
        is_sitting = True


    cv2.putText(
        frame,  # зображення де пишемо текст
        f"Sitting_total: {total_siting}, Sitting: {is_sitting}",  # текст
        (40, 40),  # позиція, лівий нижній кут
        cv2.FONT_HERSHEY_SIMPLEX,  # шрифт
        1,  # розмір шрифту
        (255, 255, 255),  # колір в BGR
        2  # товщина ліній
    )

    # Модифікуйте код щоб кількість присідань виводилась
    # правильно. Для цього вам потрібно визначати чи людина
    # зараз присідає чи піднімається за правилом:
    # ● якщо рука нижче коліна то людина встає
    # ● якщо рука вище коліна – присідає
    # Рахуйте лише ті присідання які відбулись коли людина
    # присідає та рука опинилась нижче коліна.
    # Разом з кількістю присідань відображайте чи людина
    # присідає чи встає

    


    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



