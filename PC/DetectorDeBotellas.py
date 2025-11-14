import cv2
import numpy as np

# --------------- READ DNN MODEL ---------------
config = "../model/yolov3.cfg"
weights = "../model/yolov3.weights"
LABELS = open("../model/coco.names").read().strip().split("\n")

colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config, weights)

# Obtener nombres de las capas de salida
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# --------------- OPEN CAMERA ---------------
cap = cv2.VideoCapture(0)  # 0 = webcam por defecto

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame.")
        break

    height, width, _ = frame.shape

    # --------------- CREATE BLOB ---------------
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                 swapRB=True, crop=False)

    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    classIDs = []

    # --------------- PROCESS DETECTIONS ---------------
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:
                box = detection[:4] * np.array([width, height, width, height])
                (x_center, y_center, w, h) = box.astype("int")

                x = int(x_center - (w / 2))
                y = int(y_center - (h / 2))

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # --------------- APPLY NMS ---------------
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

    # --------------- DRAW RESULTS ---------------
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = colors[classIDs[i]].tolist()
            text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar en pantalla
    cv2.imshow("YOLOv3 - Detección en Cámara", frame)

    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
