import cv2
import numpy as np
import sys
import time
import lgpio

# -------- CONFIGURAR SERVOMOTOR (lgpio) --------
SERVO_PIN = 17
CHIP = 0  # RP1 chip del Raspberry Pi 5

# Abrir chip GPIO
h = lgpio.gpiochip_open(CHIP)

# Configurar pin como salida PWM
lgpio.gpio_claim_output(h, SERVO_PIN)

def set_servo_angle(angle):
    """
    Convierte 0-180° a un pulso entre 500 y 2500us
    y lo envía usando PWM desde lgpio
    """
    pulse = int(500 + (angle * 11.11))   # microsegundos
    lgpio.tx_pwm(h, SERVO_PIN, 50, (pulse / 20000) * 100)  
    # 50Hz, duty = (pulse/periodo)*100

# -------- EVITAR MULTIPLES VENTANAS --------
try:
    if cv2.getWindowProperty("Botellas", 0) >= 0:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
except:
    pass

# -------- MODELO TINY --------
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

# -------- CAMARA --------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error camara")
    sys.exit()

cv2.namedWindow("Botellas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Botellas", 640, 480)

# -------- VARIABLES DEL SERVO --------
servo_tiempo_inicio = 0
servo_activo = False

# -------- LOOP PRINCIPAL --------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        # -------- CONTROL DEL SERVO --------
        tiempo_actual = time.time()

        if botellaEncontrada and not servo_activo:
            servo_tiempo_inicio = tiempo_actual
            servo_activo = True
            set_servo_angle(90)
            print("botellaEncontrada: True - Servo a 90°")

        elif servo_activo:
            if tiempo_actual - servo_tiempo_inicio >= 10:
                servo_activo = False
                set_servo_angle(0)
                print("10 segundos transcurridos - Servo a 0°")

        else:
            set_servo_angle(0)
            print("botellaEncontrada: False - Servo a 0°")

        cv2.imshow("Botellas", frame_small)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupción del usuario")

finally:
    set_servo_angle(0)
    cap.release()
    cv2.destroyAllWindows()
    lgpio.gpiochip_close(h)
    cv2.waitKey(1)
