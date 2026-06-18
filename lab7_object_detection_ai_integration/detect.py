from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    results = model(
        frame,
        classes=[0],   # chỉ person
        conf=0.5,
        verbose=False
    )
    annotated = results[0].plot()

    cv2.imshow("Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()