"""
Streamlit Real-Time Emotion Recognition App
-----------------------------------------

Features
========
- Reuses your existing trained Keras/TensorFlow model (default: model_resnet_emotion.h5).
- Real-time webcam inference using `streamlit-webrtc` (recommended).
- Optional Snapshot mode using Streamlit's built-in `st.camera_input` (quick test).
- Rolling smoothing window (deque) to stabilize predictions across frames (default: 10).
- Face detection via OpenCV Haar Cascade.
- Works with grayscale 48x48 FER-style preprocessing (consistent with your script).

Usage
=====
1. Activate your existing virtual environment (you said it's `env/` in your project root):
   ```bash
   # Windows PowerShell
   .\env\Scripts\activate

   # Linux / macOS
   source env/bin/activate
   ```

2. Install any *additional* deps you don't already have in that env:
   ```bash
   pip install streamlit streamlit-webrtc av opencv-python-headless tensorflow
   ```
   *If you already have OpenCV + TF installed, pip will just confirm they're satisfied.*

3. Put this file in your project root (next to `model_resnet_emotion.h5`).

4. Run the app:
   ```bash
   streamlit run streamlit_emotion_app.py
   ```

5. Grant camera permissions in the browser when prompted.

Notes
=====
- If you're running on a system that already has desktop OpenCV (`opencv-python`), you can keep it; `opencv-python-headless` is suggested for servers. Use whichever fits your env.
- If your model file has a different name or path, change it in the sidebar or rename the file.
- Press the "Stop" button in the Streamlit-WebRTC component or just stop the Streamlit server (Ctrl+C in terminal) to end.

"""

import os
import tempfile
import operator
from collections import deque
from typing import Optional

import numpy as np
import cv2
import streamlit as st

# The "av" and webrtc imports are needed for real-time video streaming
import av  # type: ignore
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration  # type: ignore

import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------
# Constants & Config Defaults
# ---------------------------
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
DEFAULT_MODEL_PATH = "model_resnet_emotion.h5"  # update in sidebar if needed
DEFAULT_SMOOTH_WINDOW = 10

# Public STUN server; good enough for local dev / P2P fallback
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


# ---------------------------
# Cached Loaders (avoid reload on rerun)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_emotion_model(model_path: str):
    """Load and cache the Keras model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load_model(model_path)
    return model


@st.cache_resource(show_spinner=False)
def load_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")
    return cascade


# ---------------------------
# Emotion Smoother
# ---------------------------
class EmotionSmoother:
    def __init__(self, emotions, maxlen=10):
        self.emotions = emotions
        self.queue = deque(maxlen=maxlen)

    def update(self, prediction: np.ndarray) -> str:
        # prediction: shape (num_emotions,)
        values = {e: 0.0 for e in self.emotions}
        prob = float(np.max(prediction))
        idx = int(np.argmax(prediction))
        emo = self.emotions[idx]
        self.queue.appendleft((prob, emo))
        for p, e in self.queue:
            values[e] += p
        # pick the emotion with the highest aggregated prob
        return max(values.items(), key=operator.itemgetter(1))[0]


# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_face(face_gray: np.ndarray) -> np.ndarray:
    """Resize to 48x48, scale to [0,1], add batch + channel dims."""
    image_scaled = cv2.resize(face_gray, (48, 48))
    image_processed = image_scaled.astype("float32") / 255.0
    image_processed = np.expand_dims(image_processed, axis=(0, -1))  # (1,48,48,1)
    return image_processed


# ---------------------------
# Video Processor for WebRTC
# ---------------------------
class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Late-bind from session_state so user can change config in sidebar
        model_path = st.session_state.get("model_path", DEFAULT_MODEL_PATH)
        smooth_window = st.session_state.get("smooth_window", DEFAULT_SMOOTH_WINDOW)

        self.model = load_emotion_model(model_path)
        self.face_cascade = load_face_cascade()
        self.smoother = EmotionSmoother(EMOTIONS, maxlen=smooth_window)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        last_emotion = None

        for (x, y, w, h) in faces:
            roi_gray = gray[y : y + h, x : x + w]
            inp = preprocess_face(roi_gray)
            pred = self.model.predict(inp, verbose=0)
            emotion = self.smoother.update(pred[0])
            last_emotion = emotion

            # draw on frame
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                img,
                f"{emotion}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if last_emotion is not None:
            st.session_state["last_emotion"] = last_emotion

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------
# Snapshot (Camera Input) Processing
# ---------------------------
def run_snapshot_inference(image_bytes: bytes) -> Optional[str]:
    # decode image
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Failed to decode image.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = load_face_cascade()
    faces = cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected in snapshot.")
        return None

    model = load_emotion_model(st.session_state.get("model_path", DEFAULT_MODEL_PATH))
    smoother = EmotionSmoother(EMOTIONS, maxlen=st.session_state.get("smooth_window", DEFAULT_SMOOTH_WINDOW))

    # Process first face (or average across all)
    emotions_found = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y : y + h, x : x + w]
        inp = preprocess_face(roi_gray)
        pred = model.predict(inp, verbose=0)
        emo = smoother.update(pred[0])
        emotions_found.append(emo)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, emo, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display annotated snapshot
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Snapshot Result", use_column_width=True)

    # Return majority or first
    if emotions_found:
        # simple: choose first (faces small set)
        result = emotions_found[0]
        st.session_state["last_emotion"] = result
        return result
    return None


# ---------------------------
# Sidebar Config
# ---------------------------
def sidebar_controls():
    st.sidebar.header("Configuration")

    # Model file path / upload override
    model_path_default = st.session_state.get("model_path", DEFAULT_MODEL_PATH)

    st.sidebar.write("**Model File**")
    uploaded_model = st.sidebar.file_uploader("Upload Keras .h5 model (optional)", type=["h5", "hdf5"], accept_multiple_files=False)
    if uploaded_model is not None:
        # save temp
        tmp_dir = st.session_state.setdefault("_tmp_model_dir", tempfile.mkdtemp(prefix="st_emotion_model_"))
        tmp_path = os.path.join(tmp_dir, uploaded_model.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.session_state["model_path"] = tmp_path
        st.sidebar.success("Uploaded model loaded.")
    else:
        # manual text input in case model isn't in default location
        model_path_input = st.sidebar.text_input("Or enter model path", model_path_default)
        st.session_state["model_path"] = model_path_input

    # Smoothing window length
    smooth_window = st.sidebar.slider("Smoothing window (frames)", min_value=1, max_value=30, value=st.session_state.get("smooth_window", DEFAULT_SMOOTH_WINDOW))
    st.session_state["smooth_window"] = smooth_window

    # Show last predicted emotion
    st.sidebar.markdown("---")
    st.sidebar.write("**Last Emotion**")
    st.sidebar.info(st.session_state.get("last_emotion", "—"))


# ---------------------------
# Main App
# ---------------------------
def main():
    st.set_page_config(page_title="Emotion Recognition", layout="wide")
    st.title("Real-Time Emotion Recognition")
    st.caption("Webcam-based FER model demo (OpenCV + TensorFlow + Streamlit)")

    # init session defaults
    if "model_path" not in st.session_state:
        st.session_state["model_path"] = DEFAULT_MODEL_PATH
    if "smooth_window" not in st.session_state:
        st.session_state["smooth_window"] = DEFAULT_SMOOTH_WINDOW
    if "last_emotion" not in st.session_state:
        st.session_state["last_emotion"] = "—"

    sidebar_controls()

    st.markdown("## Live Webcam Mode")
    st.write("Click **Start** below to begin streaming your webcam feed and see real-time emotion predictions.")

    webrtc_ctx = webrtc_streamer(
        key="emotion-stream",
        video_processor_factory=EmotionVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown("---")
    st.markdown("## Snapshot Mode (Quick Test)")
    st.write("If live streaming doesn't work in your environment, capture a still image below.")

    img_file = st.camera_input("Take a picture")
    if img_file is not None:
        result = run_snapshot_inference(img_file.getvalue())
        if result is not None:
            st.success(f"Detected emotion: {result}")

    st.markdown("---")
    st.markdown("### Debug Info")
    st.json({
        "model_path": st.session_state.get("model_path"),
        "smooth_window": st.session_state.get("smooth_window"),
        "last_emotion": st.session_state.get("last_emotion"),
    })


if __name__ == "__main__":
    main()
