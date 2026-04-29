import cv2
import sys
import os
import numpy as np
import time
import joblib
from datetime import datetime
from collections import deque
from tkinter import Tk, Label, Button, Frame
from PIL import Image, ImageTk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.feature_extractor import extract_features

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# =======================
# LOAD MODELS
# =======================
print("[INFO] Loading models...")

super_svm = joblib.load("saved_models/super_svm_aug.pkl")
super_svm_scaler = joblib.load("saved_models/super_svm_scaler_aug.pkl")

fiber_svm = joblib.load("saved_models/fiber_svm_aug.pkl")
fiber_svm_scaler = joblib.load("saved_models/fiber_svm_scaler_aug.pkl")

rigid_svm = joblib.load("saved_models/rigid_svm_aug.pkl")
rigid_svm_scaler = joblib.load("saved_models/rigid_svm_scaler_aug.pkl")

super_knn = joblib.load("saved_models/super_knn.pkl")
super_knn_scaler = joblib.load("saved_models/super_knn_scaler.pkl")

fiber_knn = joblib.load("saved_models/fiber_knn_aug.pkl")
fiber_knn_scaler = joblib.load("saved_models/fiber_knn_scaler_aug.pkl")

rigid_knn = joblib.load("saved_models/rigid_knn_aug.pkl")
rigid_knn_scaler = joblib.load("saved_models/rigid_knn_scaler_aug.pkl")

print("[SUCCESS] Models loaded.\n")

# =======================
# CONFIG
# =======================
CAMERA_INDEX = 0
FPS_TARGET = 12
PREDICT_EVERY = 5

THRESHOLD_SVM = 0.60
THRESHOLD_KNN = 0.75

CLASS_NAMES = {
    0: "Glass",
    1: "Paper",
    2: "Cardboard",
    3: "Plastic",
    4: "Metal",
    5: "Trash",
    6: "Unknown"
}

# =======================
# HELPERS
# =======================
def center_crop(img, size=256):
    h, w, _ = img.shape
    cx, cy = w // 2, h // 2
    half = size // 2
    return img[cy-half:cy+half, cx-half:cx+half]


def predict_svm(features):
    X1 = super_svm_scaler.transform([features])
    p1 = super_svm.predict_proba(X1)[0]
    s = np.argmax(p1)
    conf = p1[s]

    if conf < THRESHOLD_SVM:
        return 6, conf

    if s == 0:  # Fiber: trained with ['paper', 'cardboard'] -> 0=paper, 1=cardboard
        X2 = fiber_svm_scaler.transform([features])
        p2 = fiber_svm.predict_proba(X2)[0]
        pred = np.argmax(p2)  # Returns 0 or 1
        # pred=0 -> Paper(1), pred=1 -> Cardboard(2)
        return 1 if pred == 0 else 2, p2[pred]

    if s == 1:  # Rigid: trained with ['plastic', 'metal'] -> 0=plastic, 1=metal
        X2 = rigid_svm_scaler.transform([features])
        p2 = rigid_svm.predict_proba(X2)[0]
        pred = np.argmax(p2)  # Returns 0 or 1
        # pred=0 -> Plastic(3), pred=1 -> Metal(4)
        return 3 if pred == 0 else 4, p2[pred]

    if s == 2:  # Transparent -> Glass(0)
        return 0, conf

    if s == 3:  # Garbage -> Trash(5)
        return 5, conf

    return 6, 0.0


def predict_knn(features):
    X1 = super_knn_scaler.transform([features])
    p1 = super_knn.predict_proba(X1)[0]
    s = np.argmax(p1)
    conf = p1[s]

    if conf < THRESHOLD_KNN:
        return 6, conf

    if s == 0:  # Fiber: trained with ['paper', 'cardboard'] -> 0=paper, 1=cardboard
        X2 = fiber_knn_scaler.transform([features])
        p2 = fiber_knn.predict_proba(X2)[0]
        pred = np.argmax(p2)  # Returns 0 or 1
        # pred=0 -> Paper(1), pred=1 -> Cardboard(2)
        return 1 if pred == 0 else 2, p2[pred]

    if s == 1:  # Rigid: trained with ['plastic', 'metal'] -> 0=plastic, 1=metal
        X2 = rigid_knn_scaler.transform([features])
        p2 = rigid_knn.predict_proba(X2)[0]
        pred = np.argmax(p2)  # Returns 0 or 1
        # pred=0 -> Plastic(3), pred=1 -> Metal(4)
        return 3 if pred == 0 else 4, p2[pred]

    if s == 2:  # Transparent -> Glass(0)
        return 0, conf

    if s == 3:  # Garbage -> Trash(5)
        return 5, conf

    return 6, 0.0


# =======================
# GUI APP
# =======================
class LiveCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Waste Classifier (SVM vs KNN)")
        self.root.configure(bg="#2b2b2b")

        Label(root, text="🗑️ WASTE CLASSIFIER",
              font=("Helvetica", 18, "bold"),
              fg="white", bg="#2b2b2b").pack(pady=10)

        self.video_label = Label(root)
        self.video_label.pack()

        self.svm_label = Label(root, text="SVM: ---",
                               font=("Helvetica", 14, "bold"),
                               fg="#4CAF50", bg="#2b2b2b")
        self.svm_label.pack(pady=5)

        self.knn_label = Label(root, text="KNN: ---",
                               font=("Helvetica", 14, "bold"),
                               fg="#2196F3", bg="#2b2b2b")
        self.knn_label.pack(pady=5)

        self.fps_label = Label(root, text="FPS: 0",
                               font=("Helvetica", 10),
                               fg="gray", bg="#2b2b2b")
        self.fps_label.pack()

        Button(root, text="Quit", command=self.quit,
               bg="#f44336", fg="white",
               font=("Helvetica", 12)).pack(pady=10)

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.frame_count = 0
        self.last_time = time.time()
        self.cached_features = None

        self.svm_buffer = deque(maxlen=5)
        self.knn_buffer = deque(maxlen=5)

        # Log file for predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(f"logs/predictions_{timestamp}.txt", "w")
        self.log_file.write("Timestamp,Frame,SVM_Class,SVM_Confidence,KNN_Class,KNN_Confidence\n")
        self.prediction_count = 0

        self.update()

    def update(self):
        start = time.time()
        ret, frame = self.cap.read()

        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop = center_crop(rgb)

            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.configure(image=img)
            self.video_label.image = img

            self.frame_count += 1

            if self.frame_count % PREDICT_EVERY == 0:
                self.cached_features = extract_features(crop)

            if self.cached_features is not None:
                s_cls, s_conf = predict_svm(self.cached_features)
                k_cls, k_conf = predict_knn(self.cached_features)

                self.svm_buffer.append(s_cls)
                self.knn_buffer.append(k_cls)

                svm_final = max(set(self.svm_buffer), key=self.svm_buffer.count)
                knn_final = max(set(self.knn_buffer), key=self.knn_buffer.count)

                self.svm_label.config(
                    text=f"SVM: {CLASS_NAMES[svm_final]} ({s_conf:.1%})")

                self.knn_label.config(
                    text=f"KNN: {CLASS_NAMES[knn_final]} ({k_conf:.1%})")

                # Write predictions to log file every 5 frames
                if self.frame_count % PREDICT_EVERY == 0:
                    self.prediction_count += 1
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_line = f"{timestamp},{self.prediction_count},{CLASS_NAMES[svm_final]},{s_conf:.4f},{CLASS_NAMES[knn_final]},{k_conf:.4f}\n"
                    self.log_file.write(log_line)
                    self.log_file.flush()  # Ensure data is written immediately
                    print(f"[LOG] Frame {self.prediction_count}: SVM={CLASS_NAMES[svm_final]} ({s_conf:.2%}), KNN={CLASS_NAMES[knn_final]} ({k_conf:.2%})")

        fps = 1 / (time.time() - self.last_time)
        self.last_time = time.time()
        self.fps_label.config(text=f"FPS: {fps:.1f}")

        delay = max(1, int((1000 / FPS_TARGET) - (time.time() - start) * 1000))
        self.root.after(delay, self.update)

    def quit(self):
        self.log_file.close()
        print(f"\n[INFO] Predictions saved to: {self.log_file.name}")
        print(f"[INFO] Total predictions logged: {self.prediction_count}")
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    LiveCameraApp(root)
    root.mainloop()
