import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import time

# --- CONFIGURATION ---
# Replace with your DVR RTSP URL. Use 0 for local webcam testing.
RTSP_URL = ""

# --- BACKEND: MODEL INITIALIZATION ---
@st.cache_resource
def load_model():
    # Loads the open-source YOLOv8 nano model (lightweight & fast for live streams)
    return YOLO("yolov8n.pt")

model = load_model()

# --- FRONTEND: UI SETUP ---
st.set_page_config(page_title="Live Human Detection Dashboard", layout="wide")
st.title("📹 Live Video Analysis Dashboard")
st.subheader("Real-time Human Tracking & Analytics")

# Initialize session state for tracking application execution and logs
if "running" not in st.session_state:
    st.session_state.running = False
if "log_data" not in st.session_state:
    st.session_state.log_data = []

# Sidebar Controls
st.sidebar.header("Controls")
start_button = st.sidebar.button("▶️ Start Stream", use_container_width=True)
stop_button = st.sidebar.button("⏹️ Stop Stream", use_container_width=True)

# Handle control logic
if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

# Layout columns: Left for live feed, Right for analytics metrics/tables
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Live Video Feed")
    frame_placeholder = st.empty()  # UI placeholder for the video frames

with col2:
    st.markdown("### Analytics Panel")
    metric_placeholder = st.empty()  # UI placeholder for live count
    table_placeholder = st.empty()   # UI placeholder for timestamps log

# --- CORE PROCESSING LOOP ---
if st.session_state.running:
    # Initialize DVR Stream
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        st.error("Error: Could not connect to the video stream. Please check your RTSP URL/Camera index.")
        st.session_state.running = False
    
    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame from stream.")
            break
        
        # Run YOLOv8 inference on the frame
        # classes=0 limits tracking strictly to the 'person' class in COCO dataset
        results = model(frame, verbose=False, classes=0)
        
        # Count detected humans
        human_count = 0
        for result in results:
            boxes = result.boxes
            human_count = len(boxes)
            
            # Draw bounding boxes on the frame for visual confirmation
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Human: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert BGR (OpenCV format) to RGB (Streamlit format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Update metrics and historical logs
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metric_placeholder.metric(label="Current Human Count", value=human_count)
        
        # Log only if humans are detected to save memory/clutter
        if human_count > 0:
            st.session_state.log_data.insert(0, {"Timestamp": current_time, "Persons Detected": human_count})
            # Keep only the last 100 detections to prevent memory bloat
            st.session_state.log_data = st.session_state.log_data[:100]
        
        # Display updated logs as a clean pandas DataFrame
        if st.session_state.log_data:
            df = pd.DataFrame(st.session_state.log_data)
            table_placeholder.dataframe(df, use_container_width=True)
            
        # Tiny sleep to yield control slightly and maintain smooth UI updates
        time.sleep(0.01)
        
    # Clean up resources once 'Stop' is pressed
    cap.release()
    frame_placeholder.info("Stream stopped.")
else:
    frame_placeholder.info("Stream is currently offline. Press 'Start Stream' to begin tracking.")
    # Persistent display of logs even when stream is idle
    if st.session_state.log_data:
        df = pd.DataFrame(st.session_state.log_data)
        table_placeholder.dataframe(df, use_container_width=True)
