import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

INTEREST_POINT = mp_pose.PoseLandmark.RIGHT_HIP
DRAWING_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=6, circle_radius=6)

prev_x = 0
prev_y = 0
system_initialized = False
frame_count = 0

ALPHA_X_STABLE = 0.02
ALPHA_Y_STABLE = 0.2

ALPHA_WARMUP = 0.8
WARMUP_FRAMES = 45

def process_pose_smart(image, results):
    global prev_x, prev_y, system_initialized, frame_count
    h, w, _ = image.shape
    landmark = results.pose_landmarks.landmark[INTEREST_POINT.value]
    
    raw_x = landmark.x
    raw_y = landmark.y
    visibility = landmark.visibility
    
    if visibility > 0.5:
        if not system_initialized:
            prev_x = raw_x
            prev_y = raw_y
            system_initialized = True
            smooth_x, smooth_y = raw_x, raw_y
            dot_color = (0, 255, 255)
            
        else:
            if frame_count < WARMUP_FRAMES:
                alpha_x = ALPHA_WARMUP
                alpha_y = ALPHA_WARMUP
                dot_color = (0, 255, 255)
            else:
                alpha_x = ALPHA_X_STABLE
                alpha_y = ALPHA_Y_STABLE
                dot_color = (0, 255, 0)

            smooth_x = (alpha_x * raw_x) + ((1 - alpha_x) * prev_x)
            smooth_y = (alpha_y * raw_y) + ((1 - alpha_y) * prev_y)
            
            prev_x = smooth_x
            prev_y = smooth_y

        cx = int(smooth_x * w)
        cy = int(smooth_y * h)
        
        cv2.circle(image, (cx, cy), 8, (0, 0, 0), -1)      
        cv2.circle(image, (cx, cy), 6, dot_color, -1)   

        text_value = f"Y: {smooth_y:.2f}"
        cv2.putText(image, text_value, (cx + 15, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    return image

INPUT_VIDEO = 'your_video_mp4'
OUTPUT_VIDEO = 'y_squat_analysis.mp4'

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened(): exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

with mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=2,       
    smooth_landmarks=True     
    ) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            image = process_pose_smart(image, results)

        out.write(image)
        
        cv2.imshow('Final Analysis', image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved at: {OUTPUT_VIDEO}")
