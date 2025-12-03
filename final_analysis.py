import cv2
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

class OneEuroFilter:
    def __init__(self, te, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self._x = None
        self._dx = 0
        self._te = te
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._alpha = self._alpha_smoothing(self._min_cutoff)

    def _alpha_smoothing(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / self._te)

    def predict(self, x, te=None):
        if te is not None:
            self._te = te
        
        if self._x is None:
            self._x = x
            self._dx = 0
            return x

        a_d = self._alpha_smoothing(self._d_cutoff)
        dx = (x - self._x) / self._te
        dx_hat = a_d * dx + (1 - a_d) * self._dx

        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        a = self._alpha_smoothing(cutoff)

        x_hat = a * x + (1 - a) * self._x

        self._x = x_hat
        self._dx = dx_hat
        return x_hat

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

INPUT_VIDEO = 'your_video_mp4'
OUTPUT_VIDEO = 'output_squat_analysis.mp4'
EXCEL_FILE = 'concentric_analysis.xlsx'

HIGH_THRESHOLD = 0.50
LOW_THRESHOLD = 0.569

REPS = 0
REP_STATE = 'Up'
rep_data = []
concentric_start_time = 0

system_initialized = False
frame_count = 0
WARMUP_FRAMES = 45

FPS_FREQUENCY = 30.0
FILTER_CONFIG = {
    'te': 1.0 / FPS_FREQUENCY,
    'min_cutoff': 0.5,
    'beta': 0.05,
    'd_cutoff': 1.0
}

filter_x = None
filter_y = None

INTEREST_POINT = mp_pose.PoseLandmark.RIGHT_HIP

def process_repetition(hip_y, current_video_time):
    global REPS, REP_STATE, concentric_start_time, rep_data
    
    if frame_count < WARMUP_FRAMES:
        return

    if hip_y > LOW_THRESHOLD and REP_STATE == 'Up':
        REP_STATE = 'Down'
        concentric_start_time = current_video_time
        print(f"--> BOTTOM REACHED ({hip_y:.2f})")
    
    elif hip_y < HIGH_THRESHOLD and REP_STATE == 'Down':
        REP_STATE = 'Up'
        REPS += 1
        duration = current_video_time - concentric_start_time
        
        rep_data.append({
            'Repetition': REPS,
            'Concentric': round(duration, 2)
        })
        print(f"REP {REPS}: Rise Time {duration:.2f}s")

def process_pose_smart(image, results, video_time):
    global system_initialized, frame_count, filter_x, filter_y
    h, w, _ = image.shape
    
    y_low_px = int(LOW_THRESHOLD * h)
    y_high_px = int(HIGH_THRESHOLD * h)
    cv2.line(image, (0, y_low_px), (w, y_low_px), (0, 0, 255), 1)
    cv2.line(image, (0, y_high_px), (w, y_high_px), (0, 255, 0), 1)

    landmark = results.pose_landmarks.landmark[INTEREST_POINT.value]
    
    raw_x = landmark.x
    raw_y = landmark.y
    visibility = landmark.visibility
    
    if visibility > 0.5:
        if filter_x is None:
            filter_x = OneEuroFilter(**FILTER_CONFIG)
            filter_y = OneEuroFilter(**FILTER_CONFIG)
            filter_x.predict(raw_x)
            filter_y.predict(raw_y)
            system_initialized = True
        
        smooth_x = filter_x.predict(raw_x)
        smooth_y = filter_y.predict(raw_y)

        if frame_count < WARMUP_FRAMES:
             dot_color = (0, 255, 255)
        else:
             dot_color = (0, 255, 0)

        cx = int(smooth_x * w)
        cy = int(smooth_y * h)
        
        cv2.circle(image, (cx, cy), 8, (0, 0, 0), -1)
        cv2.circle(image, (cx, cy), 6, dot_color, -1)   

        cv2.putText(image, f"Y: {smooth_y:.2f}", (cx + 15, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        process_repetition(smooth_y, video_time)

    return image

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print("Error opening video")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 30

FILTER_CONFIG['te'] = 1.0 / fps

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2,
    smooth_landmarks=True
    ) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            image = process_pose_smart(image, results, video_time)
        else:
            y_low_px = int(LOW_THRESHOLD * h)
            y_high_px = int(HIGH_THRESHOLD * h)
            cv2.line(image, (0, y_low_px), (w, y_low_px), (0, 0, 255), 1) 
            cv2.line(image, (0, y_high_px), (w, y_high_px), (0, 255, 0), 1)

        cv2.rectangle(image, (0, 0), (250, 80), (0, 0, 0), -1)
        cv2.putText(image, f'REPS: {REPS}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if rep_data:
            last_rep = rep_data[-1]
            cv2.putText(image, f"Con: {last_rep['Concentric']}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        out.write(image)

    if rep_data:
        df = pd.DataFrame(rep_data)
        
        try:
            df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')
            print(f"Data exported to {EXCEL_FILE}")
        except ModuleNotFoundError:
            print("Missing openpyxl. Saving as CSV...")
            df.to_csv('backup_data.csv', index=False)

        plt.figure(figsize=(10, 6))
        plt.plot(df['Repetition'], df['Concentric'], marker='o', linewidth=2, color='b')
        
        plt.xticks(df['Repetition'])
        
        max_time = df['Concentric'].max()
        plt.ylim(0, max_time * 1.5) 
        
        for x, y in zip(df['Repetition'], df['Concentric']):
            plt.text(x, y + (max_time*0.05), f'{y}s', ha='center', color='blue')

        plt.title('Rise Velocity')
        plt.xlabel('Repetition')
        plt.ylabel('Seconds')
        plt.grid(axis='y', linestyle='--', alpha=0.7) 
        
        plt.savefig('line_chart.png')
        print(f"Process finished. Video saved at: {OUTPUT_VIDEO}")
    else:
        print("No repetitions completed.")

cap.release()
out.release()
cv2.destroyAllWindows()
