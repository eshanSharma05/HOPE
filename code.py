import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Cloud Live Human Detection", layout="wide")
st.title("📹 Cloud Live Video Analysis")

# Public STUN servers allow WebRTC to bypass network firewalls/NAT layers
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- BACKEND: LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Initialize log data in session state
if "log_data" not in st.session_state:
    st.session_state.log_data = []

# --- MULTITHREADED FRAME VISUALIZATION & INFERENCE ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert incoming WebRTC frame to standard NumPy BGR format for OpenCV/YOLO
    img = frame.to_ndarray(format="bgr24")
    
    # Run YOLOv8 on the live frame (Filter class=0 for human detection)
    results = model(img, verbose=False, classes=0)
    
    human_count = 0
    for result in results:
        boxes = result.boxes
        human_count = len(boxes)
        
        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Human: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Note: Streamlit's typical 'st.write' elements won't directly update inside this callback 
    # thread, but it successfully outputs the processed video back to the user interface.
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- FRONTEND UI LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Live Browser Video Feed")
    # Native WebRTC streaming component replaces the loop architecture
    ctx = webrtc_streamer(
        key="human-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("### Analytics Panel")
    if ctx.state.playing:
        st.success("Stream active! Tracking frames...")
    else:
        st.info("Click 'START' on the left panel to allow camera permissions and stream.")
