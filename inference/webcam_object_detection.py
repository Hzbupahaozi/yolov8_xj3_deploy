import cv2
import time
from yolov8 import YOLOv8

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv7 object detector
model_path = "./yolov8n.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

while True:
    # Read frame from the video
    start_time = time.time()
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)
    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)
    end_time = time.time()
    # Calculate FPS
    fps = 1.0 / (end_time - start_time)
    print("FPS: %.2f" % fps)
    # Press key q to stop
    if cv2.waitKey(10) == ord('q'):
        break

# 随时准备按q退出
cap.release()
cv2.destroyAllWindows()
