import cv2
import numpy as np
import time
import lgpio

# ------------------- CONFIGURACION SERVO -------------------
SERVO_PIN = 4
SERVO_FREQ = 50  # 50 Hz → 20ms periodo
chip = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(chip, SERVO_PIN)

def set_servo_angle(angle):
    """
    Control manual del servo usando pulsos de 1–2 ms.
    """
    pulse_width = 1000 + (angle / 180.0) * 1000  # de 1000 a 2000 microsegundos
    period = 20000  # 20 ms

    lgpio.gpio_write(chip, SERVO_PIN, 1)
    time.sleep(pulse_width / 1_000_000)

    lgpio.gpio_write(chip, SERVO_PIN, 0)
    time.sleep((period - pulse_width) / 1_000_000)

# Inicializar servo en 0°
set_servo_angle(0)
print("Servo inicializado en 0 grados")

# ------------------- CARGAR MODELO YOLO -------------------
config = "../model/tiny/yolov3.cfg"
weights = "../model/tiny/yolov3.weights"
LABELS = open("../model/coco.names").read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(config, weights)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ------------------- INICIALIZAR VENTANA (UNA SOLA) -------------------
try:
    cv2.destroyAllWindows()
    cv2.waitKey(1)
except:
    pass

cv2.namedWindow("Deteccion Botellas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Deteccion Botellas", 640, 480)

# ------------------- CAMARA -------------------
camera = cv2.VideoCapture(0)

triggered = False
action_end = 0
cooldown = 5
non_bottle_counter = 0
rearm_frames = 5
servo_angle = 0

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        # Mostrar cAmara (WINDOW YA EXISTE)
        cv2.imshow("Deteccion Botellas", frame)

        # Preparar YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        found_bottle = False

        for out in layerOutputs:
            for detection in out:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5 and LABELS[classID] == "bottle":
                    found_bottle = True

        now = time.time()

        # ------------------- LOGICA DEL SERVO -------------------
        if found_bottle and not triggered and now >= action_end:
            triggered = True
            action_end = now + cooldown
            non_bottle_counter = 0
            servo_angle = 180
            set_servo_angle(servo_angle)
            print("Botella detectada → Servo a 180°")

        if triggered:
            if now < action_end:
                set_servo_angle(servo_angle)
            else:
                if not found_bottle:
                    non_bottle_counter += 1
                    if non_bottle_counter >= rearm_frames:
                        triggered = False
                        non_bottle_counter = 0
                        servo_angle = 0
                        set_servo_angle(servo_angle)
                        print("Rearmado → Servo a 0°")
                else:
                    non_bottle_counter = 0
        else:
            servo_angle = 0
            set_servo_angle(servo_angle)

        # Salir con ESC
        key = cv2.waitKey(1)
        if key == 27:
            break

finally:
    set_servo_angle(0)
    lgpio.gpiochip_close(chip)
    camera.release()
    cv2.destroyAllWindows()
    print("Finalizado – Servo a 0°")
