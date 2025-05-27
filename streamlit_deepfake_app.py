
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from tensorflow.keras.models import load_model

# Load the trained CNN model
@st.cache_resource
def load_cnn_model():
    model = load_model("cnn_model.h5")
    return model

model = load_cnn_model()

st.title("Deepfake Detection - Frame Based (CNN)")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

def extract_frames_from_video(video_path, num_frames=15, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frame = cv2.resize(frame, resize)
            frames.append(frame)
    cap.release()
    return frames

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    st.video(video_path)

    st.write("Extracting frames and making prediction...")
    frames = extract_frames_from_video(video_path)

    if not frames:
        st.error("Could not extract frames from the video.")
    else:
        frames_np = np.array(frames) / 255.0
        preds = model.predict(frames_np)
        avg_pred = np.mean(preds)
        label = "Fake" if avg_pred > 0.5 else "Real"
        st.success(f"Prediction: **{label}** ({avg_pred:.2f} confidence)")

    os.unlink(video_path)
