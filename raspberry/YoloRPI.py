import cv2
import numpy as np
import os
import sys

# ----------------- EVITAR EJECUCIONES MÚLTIPLES -----------------
try:
    if cv2.getWindowProperty("Detección Botellas - YOLOv3", 0) >= 0:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
except:
    pass

# ----------------- LEER MODELO -----------------
config = "../model/yolov3.cfg"
weights = "../model/yolov3.weights"
LABELS = open("../model/coco.names").read().strip().split("\n")

TARGET_CLASS = "bottle"

if TARGET_CLASS not in LABELS:
    print(f"Error: '{TARGET_CLASS}' no está en coco.names")
    sys.exit()

TARGET_ID = LABELS.index(TARGET_CLASS)
color = (0, 255, 0)

net = cv2.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# ----------------- ABRIR CÁMARA -----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    sys.exit()

cv2.namedWindow("Detección Botellas - YOLOv3", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detección Botellas - YOLOv3", 640, 480)

# ----------------- LOOP PRINCIPAL -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    classIDs = []

    botellaEncontrada = False

    # ----------------- PROCESAR DETECCIONES -----------------
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
                (x_center, y_center, w, h) = box.astype("int")

                x = int(x_center - w / 2)
                y = int(y_center - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # ----------------- DIBUJAR -----------------
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            text = f"BOTTLE: {confidences[i]:.2f}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Detección Botellas - YOLOv3", frame)

    print("botellaEncontrada:", botellaEncontrada)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# ----------------- LIMPIEZA -----------------
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
