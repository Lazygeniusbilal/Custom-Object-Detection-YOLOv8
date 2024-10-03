from ultralytics import  YOLO
import  cv2
import math


cap = cv2.VideoCapture(0) # 0 to open default camera
# load the yolov11 model
model = YOLO("yolo11n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    ret, frame = cap.read()
    # capture frame via model
    results = model.detect(frame, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:

            # get the bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]
            # convert those values into integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # get the confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence ------->", confidence)

            # get the class
            cls = box.cls[0]
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            # now put all togethere
            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)


    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()