# Importación de librerías necesarias
import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

def palm_centroid(coordinates_list):
    """
    Calcula el centroide de la palma usando las coordenadas de puntos clave.
    """
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    return int(centroid[0]), int(centroid[1])

# Inicialización de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Inicializar la captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Definición de puntos clave
thumb_points = [1, 2, 4]  # Pulgar
palm_points = [0, 1, 2, 5, 9, 13, 17]  # Palma
fingertips_points = [8, 12, 16, 20]  # Puntas de los dedos
finger_base_points = [6, 10, 14, 18]  # Base de los dedos

# Colores para cada dedo (BGR)
finger_colors = [
    (180, 229, 255),  # Pulgar - PEACH
    (128, 64, 128),   # Índice - PURPLE
    (0, 204, 255),    # Medio - YELLOW
    (48, 255, 48),    # Anular - GREEN
    (192, 101, 21)    # Meñique - BLUE
]

# Inicialización del detector de manos
with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while True:
        # Captura del frame de video
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        fingers_counter = "_"
        thickness = [2] * 5  # Grosor inicial para los dedos

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Coordenadas de cada punto clave
                coordinates_thumb = [
                    (int(hand_landmarks.landmark[idx].x * width),
                     int(hand_landmarks.landmark[idx].y * height))
                    for idx in thumb_points
                ]
                coordinates_palm = [
                    (int(hand_landmarks.landmark[idx].x * width),
                     int(hand_landmarks.landmark[idx].y * height))
                    for idx in palm_points
                ]
                coordinates_ft = [
                    (int(hand_landmarks.landmark[idx].x * width),
                     int(hand_landmarks.landmark[idx].y * height))
                    for idx in fingertips_points
                ]
                coordinates_fb = [
                    (int(hand_landmarks.landmark[idx].x * width),
                     int(hand_landmarks.landmark[idx].y * height))
                    for idx in finger_base_points
                ]

                # Detección del pulgar
                p1, p2, p3 = map(np.array, coordinates_thumb)
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                thumb_finger = angle > 150

                # Centroide de la palma
                nx, ny = palm_centroid(coordinates_palm)
                cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)
                centroid = np.array([nx, ny])
                coordinates_ft = np.array(coordinates_ft)
                coordinates_fb = np.array(coordinates_fb)

                # Detectar dedos levantados
                d_centrid_ft = np.linalg.norm(centroid - coordinates_ft, axis=1)
                d_centrid_fb = np.linalg.norm(centroid - coordinates_fb, axis=1)
                dif = d_centrid_ft - d_centrid_fb
                fingers = np.append(thumb_finger, dif > 0)
                fingers_counter = str(np.count_nonzero(fingers))

                # Actualizar grosor según dedos levantados
                for i, finger in enumerate(fingers):
                    thickness[i] = -1 if finger else 2

                # Dibujar los landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Visualizar contador principal de dedos
        cv2.rectangle(frame, (0, 0), (80, 80), (125, 220, 0), -1)
        cv2.putText(frame, fingers_counter, (15, 65), 1, 5, (255, 255, 255), 2)

        # Dibujar indicadores de cada dedo
        start_x = 100
        for i, color in enumerate(finger_colors):
            cv2.rectangle(frame, (start_x, 10), (start_x + 50, 60), color, thickness[i])
            cv2.putText(frame, ["Pulgar", "Indice", "Medio", "Anular", "Menique"][i],
                        (start_x, 80), 1, 1, (255, 255, 255), 2)
            start_x += 60

        # Mostrar el frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
