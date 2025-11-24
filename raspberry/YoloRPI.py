import cv2
import numpy as np
import os
import sys

# -------- EVITAR MULTIPLES VENTANAS --------
try:
    if cv2.getWindowProperty("Botellas", 0) >= 0:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
except:
    pass

# -------- LEER MODELO (TINY) --------
config = "../model/tiny/yolov3-tiny.cfg"
weights = "../model/tiny/yolov3-tiny.weights"
LABELS = open("../model/coco.names").read().strip().split("\n")

TARGET_CLASS = "bottle"
TARGET_ID = LABELS.index(TARGET_CLASS)
color = (0, 255, 0)

net = cv2.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# -------- CÁMARA --------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error cámara")
    sys.exit()

cv2.namedWindow("Botellas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Botellas", 640, 480)

# -------- LOOP PRINCIPAL --------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reducir tamaño para bajar temperatura
    frame_small = cv2.resize(frame, (320, 240))

    blob = cv2.dnn.blobFromImage(frame_small, 1/255.0, (256, 256), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    height, width = frame_small.shape[:2]
    boxes = []
    confidences = []

    botellaEncontrada = False

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)

            if classID != TARGET_ID:
                continue

            confidence = scores[classID]

            if confidence > 0.5:
                botellaEncontrada = True

                box = detection[:4] * np.array([width, height, width, height])
                (xc, yc, w, h) = box.astype("int")

                x = int(xc - w / 2)
                y = int(yc - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            text = f"{confidences[i]:.2f}"

            cv2.rectangle(frame_small, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame_small, text, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Botellas", frame_small)
    print("botellaEncontrada:", botellaEncontrada)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
