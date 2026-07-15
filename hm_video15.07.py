# Завдання 1
# Відкрийте відео з файлу data\lesson7\meter.mp4.
# Проведіть бінарізацію кадрів та збережіть в новий файл.
# Можливо очистіть від шуму або наведіть різкість через
# bilateralFilter

import cv2

cap = cv2.VideoCapture(r'data\lesson7\meter.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width)
print(height)


fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(
    "result.mp4",
    fourcc,
    fps,
    (width, height),
    isColor=False,
)

while True:
    success,frame = cap.read()

    if not success:
         break

    cv2.imshow('orig',frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)


    blur = cv2.bilateralFilter(gray, 11, 17, 17
    )
    #cv2.imshow("blur", blur)

    adapt = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    cv2.imshow("binarization", adapt)

    out_writer.write(adapt)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

out_writer.release()
cap.release()
cv2.destroyAllWindows()