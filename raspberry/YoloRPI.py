import cv2
import numpy as np
import time
import lgpio

# ------------------- CONFIGURACION SERVO -------------------
SERVO_PIN = 4
SERVO_FREQ = 50

ANGULOINICIO = 180
ANGULOFINAL = 0

DISPLAY_WIDTH = 320
DISPLAY_HEIGHT = 240

chip = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(chip, SERVO_PIN, 0)

def set_servo_angle(angle):
    """
    Control del servo usando pulsos manuales con lgpio.
    Convierte angulo (0-180) a ancho de pulso (1000-2000 µs).
    """
    pulse_width = 1000 + (angle / 180.0) * 1000
    lgpio.gpio_write(chip, SERVO_PIN, 1)
    time.sleep(pulse_width / 1_000_000)
    lgpio.gpio_write(chip, SERVO_PIN, 0)

# Mantener servo en ANGULOINICIO por defecto (180°).
servo_angle = ANGULOINICIO
set_servo_angle(servo_angle)
print(f"Servomotor inicializado en {servo_angle} grados")

# ------------------- CARGAR MODELO YOLO -------------------
config = "../model/tiny/yolov3-tiny.cfg"
weights = "../model/tiny/yolov3-tiny.weights"
LABELS = open("../model/coco.names").read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ------------------- INICIALIZAR VENTANA -------------------
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.namedWindow("Deteccion Botellas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Deteccion Botellas", DISPLAY_WIDTH, DISPLAY_HEIGHT)

# ------------------- CAMARA -------------------
camera = cv2.VideoCapture(0)
# Ajustar resolución de captura a la del display para menos carga
camera.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

triggered = False
action_end = 0.0
cooldown = 5.0
rearm_frames_needed = 5
non_bottle_counter = 0

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        # Reducir tamaño antes de mostrar para que la ventana sea pequeña
        frame_display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
        cv2.imshow("Deteccion Botellas", frame_display)

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
            servo_angle = ANGULOFINAL  # 0°
            set_servo_angle(servo_angle)
            print("Botella detectada, servomotor a 0 grados")

        if triggered:
            if now < action_end:
                set_servo_angle(servo_angle)
            else:
                if not found_bottle:
                    non_bottle_counter += 1
                    if non_bottle_counter >= rearm_frames_needed:
                        triggered = False
                        non_bottle_counter = 0
                        servo_angle = ANGULOINICIO  # volver a 180°
                        set_servo_angle(servo_angle)
                        print(f"Servo retornando a {servo_angle} grados")
                else:
                    non_bottle_counter = 0
        else:
            servo_angle = ANGULOINICIO  # 180° por defecto
            set_servo_angle(servo_angle)

        print("Botella detectada:", found_bottle, end="")
        print(f" | Servo: {servo_angle}°")

        keyboard_input = cv2.waitKey(1) & 0xFF
        if keyboard_input == 27 or keyboard_input in [ord("q"), ord("Q")]:
            break

finally:
    set_servo_angle(ANGULOINICIO)
    lgpio.gpiochip_close(chip)
    print(f"Programa interrumpido - Servo en {ANGULOINICIO} grados")
    camera.release()
    cv2.destroyAllWindows()
