import cv2
import numpy as np
import pyttsx3
import time

# Initialize the text-to-speech engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "cell phone", "ship", "mouse"]

confidence_threshold = 0.2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    detected_objects = []  # Store detected objects and their positions

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            obj = classes[class_id]
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            color = (0, 255, 0)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15

            # Calculate object center
            obj_center_x = (startX + endX) // 2
            obj_center_y = (startY + endY) // 2

            # Determine object position relative to the frame
            if obj_center_x < width / 3:
                position = "left"
            elif obj_center_x > 2 * width / 3:
                position = "right"
            else:
                position = "center"

            detected_objects.append((obj, position))  # Store detected object and position

            # Display the position on the frame
            cv2.putText(frame, f"{obj} {position}", (startX, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Construct a string containing all detected objects and their positions
    feedback_text = ""
    for obj, position in detected_objects:
        feedback_text += f"{obj} detected on the {position}. "

    # Speak out all detected objects and their positions
    engine.say(feedback_text)
    engine.runAndWait()

    # Add a delay of 3 seconds
    time.sleep(3)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

