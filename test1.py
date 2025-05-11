import cv2
import torch
import time
from gtts import gTTS
import os
import uuid
import speech_recognition as sr
from ultralytics import YOLO
from proactive import activate_proactive_system
import threading

# Load YOLOv10 model
model = YOLO("weights/yolov10s.pt")

# Constants
KNOWN_WIDTH = 0.5
FOCAL_LENGTH = 500
CHECK_INTERVAL = 10
DETECTION_THRESHOLD = 0.5

# Shared buffer for voice commands
command_buffer = []

# gTTS-based speech output
def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        filename = f"/tmp/speech_{uuid.uuid4()}.mp3"
        tts.save(filename)
        os.system(f"mpg123 -q {filename}")
        os.remove(filename)
    except Exception as e:
        print("Speech Error:", e)

# Continuous voice recognition in background thread
def listen_loop():
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=3)  # Set to your webcam mic card index
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    while True:
        with mic as source:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                command = recognizer.recognize_google(audio).lower()
                print("Heard:", command)
                command_buffer.append(command)
            except:
                continue

# Estimate object distance from width in pixels
def estimate_distance(pixel_width):
    return float("inf") if pixel_width == 0 else (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

# Classify detected objects by position
def classify_objects(results, frame_width):
    positions = {"left": [], "center": [], "right": []}
    distances = {}
    left, right = frame_width * 0.33, frame_width * 0.66

    for result in results:
        for cls, conf, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
            if conf < DETECTION_THRESHOLD:
                continue
            obj = result.names[int(cls)]
            x_min, _, x_max, _ = xyxy.tolist()
            pixel_width = x_max - x_min
            distance = estimate_distance(pixel_width)
            distances[obj] = distance

            if x_max < left:
                positions["left"].append(obj)
            elif x_min > right:
                positions["right"].append(obj)
            else:
                positions["center"].append(obj)
    return positions, distances, bool(distances)

# Generate audio feedback from object positions and distances
def generate_feedback(positions, distances):
    description = []

    for side, objs in positions.items():
        if objs:
            count_desc = ', '.join(f"{objs.count(o)} {o}{'s' if objs.count(o) > 1 else ''}" for o in set(objs))
            description.append(f"{count_desc} on the {side}.")

    for obj, dist in distances.items():
        if dist < 0.5:
            description.append(f"Warning! {obj} is too close.")
        elif dist < 1:
            description.append(f"{obj} is close.")

    final_msg = " ".join(description) or "No significant objects detected."
    speak(final_msg)

# Main object detection loop
def detect_objects_realtime():
    cap = cv2.VideoCapture(1)  # Use index 0 or 1 depending on your webcam
    is_active = False
    last_check = time.time()

    # Start voice recognition in background
    threading.Thread(target=listen_loop, daemon=True).start()

    speak("System is ready. Say start detection to begin.")

    while True:
        if command_buffer:
            command = command_buffer.pop(0)

            if "start detection" in command:
                is_active = True
                speak("Object detection activated.")
            elif "activate proactive" in command or "emergency" in command:
                speak("Activating proactive vision system.")
                activate_proactive_system()
            elif "stop detection" in command:
                is_active = False
                speak("Object detection stopped.")
            elif "exit" in command:
                speak("Exiting system.")
                break

        if is_active:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed.")
                continue

            frame_width = frame.shape[1]
            results = model.predict(frame, verbose=False)
            positions, distances, found = classify_objects(results, frame_width)

            if found:
                generate_feedback(positions, distances)

            if time.time() - last_check > CHECK_INTERVAL:
                if command_buffer:
                    command = command_buffer.pop(0)
                    if "stop detection" in command:
                        is_active = False
                        speak("Object detection stopped.")
                    elif "exit" in command:
                        speak("Exiting system.")
                        break
                last_check = time.time()

            time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

# Entry point
if __name__ == "__main__":
    try:
        detect_objects_realtime()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        cv2.destroyAllWindows()
