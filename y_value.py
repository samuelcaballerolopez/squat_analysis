import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

PUNTO_INTERES = mp_pose.PoseLandmark.RIGHT_HIP
ESPECIFICACION_PUNTO = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=6, circle_radius=6)

x_anterior = 0
y_anterior = 0
sistema_iniciado = False
conteo_frames = 0

ALPHA_X_ESTABLE = 0.02
ALPHA_Y_ESTABLE = 0.2

ALPHA_CALENTAMIENTO = 0.8
FRAMES_DE_CALENTAMIENTO = 45

def procesar_pose_inteligente(image, results):
    global x_anterior, y_anterior, sistema_iniciado, conteo_frames
    h, w, _ = image.shape
    landmark = results.pose_landmarks.landmark[PUNTO_INTERES.value]
    
    x_cruda = landmark.x
    y_cruda = landmark.y
    visibilidad = landmark.visibility
    
    if visibilidad > 0.5:
        if not sistema_iniciado:
            x_anterior = x_cruda
            y_anterior = y_cruda
            sistema_iniciado = True
            x_suavizada, y_suavizada = x_cruda, y_cruda
            color_punto = (0, 255, 255)
            
        else:
            if conteo_frames < FRAMES_DE_CALENTAMIENTO:
                alpha_x = ALPHA_CALENTAMIENTO
                alpha_y = ALPHA_CALENTAMIENTO
                color_punto = (0, 255, 255)
            else:
                alpha_x = ALPHA_X_ESTABLE
                alpha_y = ALPHA_Y_ESTABLE
                color_punto = (0, 255, 0)

            x_suavizada = (alpha_x * x_cruda) + ((1 - alpha_x) * x_anterior)
            y_suavizada = (alpha_y * y_cruda) + ((1 - alpha_y) * y_anterior)
            
            x_anterior = x_suavizada
            y_anterior = y_suavizada

        cx = int(x_suavizada * w)
        cy = int(y_suavizada * h)
        
        cv2.circle(image, (cx, cy), 8, (0, 0, 0), -1)      
        cv2.circle(image, (cx, cy), 6, color_punto, -1)   

        texto_valor = f"Y: {y_suavizada:.2f}"
        cv2.putText(image, texto_valor, (cx + 15, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    return image

VIDEO_ENTRADA = 'squat_2.mp4'
VIDEO_SALIDA = 'y_squat_analisis.mp4'

cap = cv2.VideoCapture(VIDEO_ENTRADA)
if not cap.isOpened(): exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(VIDEO_SALIDA, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

with mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=2,       
    smooth_landmarks=True     
    ) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        conteo_frames += 1

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            image = procesar_pose_inteligente(image, results)

        out.write(image)
        
        cv2.imshow('Analisis Final', image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video guardado en: {VIDEO_SALIDA}")
