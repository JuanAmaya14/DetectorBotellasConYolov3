import cv2
import numpy as np

# -------------------------------------------------
# CONFIGURACIÓN DEL MODELO
# -------------------------------------------------
config = "../model/yolov3.cfg"
weights = "../model/yolov3.weights"
LABELS = open("../model/coco.names").read().strip().split("\n")

colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config, weights)

# Para Raspberry Pi (CPU)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Obtener capas de salida
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# -------------------------------------------------
# CONFIGURACIÓN DE LA CÁMARA PARA RPi
# -------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # evita ventanas múltiples

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # resolución baja = más FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame.")
            break

        height, width, _ = frame.shape

        # --------- CREAR BLOB OPTIMIZADO PARA RPI ----------
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320),
                                     swapRB=True, crop=False)

        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        classIDs = []

        # --------- PROCESAR DETECCIONES ----------
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

        # --------- NMS ----------
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

        # --------- DIBUJAR ----------
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                color = colors[classIDs[i]].tolist()
                text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("YOLOv3 - Raspberry Pi", frame)

        # --------- CLAVE: aumentar waitKey para evitar ventanas múltiples ----------
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()