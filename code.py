import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import urllib.request
import numpy as np
import time

# --- CONFIGURATION ---
# Your verified dynamic Ngrok development domain path
HTTP_STREAM_URL = "https://bulk-boxer-handiness.ngrok-free.dev/video"

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.set_page_config(page_title="Live Camera Analysis", layout="wide")
st.title("📹 Deployed Live Video Analysis Dashboard")

if "running" not in st.session_state:
    st.session_state.running = False
if "log_data" not in st.session_state:
    st.session_state.log_data = []

# Controls
st.sidebar.header("Controls")
if st.sidebar.button("▶️ Start Stream"):
    st.session_state.running = True
if st.sidebar.button("⏹️ Stop Stream"):
    st.session_state.running = False

col1, col2 = st.columns([2, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    metric_placeholder = st.empty()
    table_placeholder = st.empty()

# --- BOUNDARY-SAFE HTTP FRAME PROCESSING ---
if st.session_state.running:
    try:
        # Open a direct web stream pipeline to your local Ngrok tunnel
        stream = urllib.request.urlopen(HTTP_STREAM_URL, timeout=10)
        bytes_buffer = bytes()
        
        while st.session_state.running:
            # Read chunks of image data from the stream payload
            bytes_buffer += stream.read(1024 * 8)
            
            # Look for standard JPEG start (0xffd8) and end (0xffd9) markers
            a = bytes_buffer.find(b'\xff\xd8')
            b = bytes_buffer.find(b'\xff\xd9')
            
            if a != -1 and b != -1:
                jpg_bytes = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]
                
                # Turn raw image bytes straight into a frame OpenCV can draw on
                frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # ML Inference Engine
                results = model(frame, verbose=False, classes=0)
                human_count = len(results[0].boxes) if results else 0

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Dashboard Rendering Matrix
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                current_time = datetime.now().strftime("%H:%M:%S")
                metric_placeholder.metric(label="Humans Detected", value=human_count)

                if human_count > 0:
                    st.session_state.log_data.insert(0, {"Timestamp": current_time, "Count": human_count})
                    st.session_state.log_data = st.session_state.log_data[:50]
                    table_placeholder.dataframe(pd.DataFrame(st.session_state.log_data), use_container_width=True)
                
                # Control frame loop pacing
                time.sleep(0.01)
                
    except Exception as e:
        st.error(f"Cloud Connection Timeout or Reset: {e}")
        st.session_state.running = False
else:
    frame_placeholder.info("Stream offline. Press 'Start Stream' to connect.")
