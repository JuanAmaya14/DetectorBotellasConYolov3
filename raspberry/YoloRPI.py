import cv2
import numpy as np
import time
import lgpio

# ------------------- CONFIGURACION SERVO -------------------
SERVO_PIN = 4
SERVO_FREQ = 50  # 50 Hz para servomotor estándar

chip = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(chip, SERVO_PIN, 0)

def set_servo_angle(angle):
    """
    Control del servo usando pulsos manuales con lgpio.
    Convierte ángulo (0-180) a ancho de pulso (1000-2000 µs).
    """
    pulse_width = 1000 + (angle / 180.0) * 1000
    lgpio.gpio_write(chip, SERVO_PIN, 1)
    time.sleep(pulse_width / 1_000_000)
    lgpio.gpio_write(chip, SERVO_PIN, 0)

servo_angle = 0
set_servo_angle(servo_angle)
print("Servomotor inicializado en 0 grados")

# ------------------- CARGAR MODELO YOLO -------------------
config = "../model/yolov3.cfg"
weights = "../model/yolov3.weights"
LABELS = open("../model/coco.names").read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(config, weights)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ------------------- INICIALIZAR VENTANA -------------------
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
action_end = 0.0
cooldown = 5.0
rearm_frames_needed = 5
non_bottle_counter = 0
servo_angle = 0

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        # Mostrar cámara
        cv2.imshow("Deteccion Botellas", frame)

        # Preparar YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
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
                    break
            if found_bottle:
                break

        now = time.time()

        # ------------------- LOGICA DEL SERVO -------------------
        if found_bottle and not triggered and now >= action_end:
            triggered = True
            action_end = now + cooldown
            non_bottle_counter = 0
            servo_angle = 180
            set_servo_angle(servo_angle)
            print("Botella detectada, servomotor a 180 grados")

        if triggered:
            if now < action_end:
                set_servo_angle(servo_angle)
            else:
                if not found_bottle:
                    non_bottle_counter += 1
                    if non_bottle_counter >= rearm_frames_needed:
                        triggered = False
                        non_bottle_counter = 0
                        set_servo_angle(0)
                        servo_angle = 0
                        print("Servo retornando a 0 grados")
                else:
                    non_bottle_counter = 0
        else:
            servo_angle = 0
            set_servo_angle(servo_angle)

        print("Botella detectada:", found_bottle, end="")
        print(f" | Servo: {servo_angle}°")

        keyboard_input = cv2.waitKey(1) & 0xFF
        if keyboard_input == 27 or keyboard_input in [ord("q"), ord("Q")]:
            break

finally:
    set_servo_angle(0)
    lgpio.gpiochip_close(chip)
    print("Programa interrumpido - Servo en 0 grados")
    camera.release()
    cv2.destroyAllWindows()
