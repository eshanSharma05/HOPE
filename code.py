import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import urllib.request
import numpy as np
import time

# --- CONFIGURATION FOR WHITELISTED ROUTER ACCESS ---
# 1. Provide the building's Public WAN IP address
# BUILDING_PUBLIC_IP = "122.160.x.x" 

# 2. Provide the external port forwarded on the router (e.g., 8085)
#FORWARDED_PORT = "8085" 

#USERNAME = "admin"
#PASSWORD = "your_dahua_password"
#CHANNEL = "4"

# Direct API snapshot endpoint string
#HTTP_STREAM_URL = f"http://{BUILDING_PUBLIC_IP}:{FORWARDED_PORT}/cgi-bin/snapshot.cgi?channel={CHANNEL}"


# --- CONFIGURATION ---
# Ensure this matches your active Ngrok URL exactly
HTTP_STREAM_URL = "https://bulk-boxer-handiness.ngrok-free.dev/video"

@st.cache_resource
def load_model():
    # Using YOLOv8 nano for real-time cloud inference speed
    return YOLO("yolov8n.pt")

model = load_model()

# Full list of common COCO dataset classes for selection
AVAILABLE_CLASSES = {
    "Person (Human)": 0,
    "Bicycle": 1,
    "Car": 2,
    "Motorcycle": 3,
    "Backpack": 24,
    "Umbrella": 25,
    "Handbag": 26,
    "Laptop": 63,
    "Cell Phone": 67
}

st.set_page_config(page_title="Object & Human Detection Dashboard", layout="wide")
st.title("📹 Live Object & Human Detection Dashboard")

if "running" not in st.session_state:
    st.session_state.running = False
if "log_data" not in st.session_state:
    st.session_state.log_data = []

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Tracking Settings")
selected_labels = st.sidebar.multiselect(
    "Select Objects to Detect:",
    options=list(AVAILABLE_CLASSES.keys()),
    default=["Person (Human)"]
)

# Map selected string labels to their respective YOLO integer class IDs
selected_class_ids = [AVAILABLE_CLASSES[label] for label in selected_labels]

st.sidebar.markdown("---")
if st.sidebar.button("▶️ Start Stream", use_container_width=True):
    st.session_state.running = True
if st.sidebar.button("⏹️ Stop Stream", use_container_width=True):
    st.session_state.running = False

# Layout Structure
col1, col2 = st.columns([2, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    metric_placeholder = st.empty()
    table_placeholder = st.empty()

# --- PROCESSING LOOP ---
if st.session_state.running and selected_class_ids:
    try:
        stream = urllib.request.urlopen(HTTP_STREAM_URL, timeout=30)
        bytes_buffer = bytes()
        
        while st.session_state.running:
            bytes_buffer += stream.read(1024 * 8)
            a = bytes_buffer.find(b'\xff\xd8')
            b = bytes_buffer.find(b'\xff\xd9')
            
            if a != -1 and b != -1:
                jpg_bytes = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]
                
                frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                
                # Run YOLO inference filtering strictly by the selected UI classes
                results = model(frame, verbose=False, classes=selected_class_ids)
                
                human_count = 0
                total_objects = 0
                
                for result in results:
                    boxes = result.boxes
                    total_objects = len(boxes)
                    
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Increment specific human counter if the class ID is 0
                        if cls_id == 0:
                            human_count += 1
                        
                        # Fetch the text label name from the model dictionary
                        label_name = model.names[cls_id].upper()
                        
                        # Draw dynamic bounding boxes and labels
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label_name} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Render output to Streamlit Web Dashboard
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                current_time = datetime.now().strftime("%H:%M:%S")
                metric_placeholder.metric(label="Humans Tracked", value=human_count)

                # Keep a running log if any monitored objects are present
                if total_objects > 0:
                    st.session_state.log_data.insert(0, {
                        "Timestamp": current_time, 
                        "Humans Present": human_count,
                        "Total Target Objects": total_objects
                    })
                    st.session_state.log_data = st.session_state.log_data[:50]
                    table_placeholder.dataframe(pd.DataFrame(st.session_state.log_data), use_container_width=True)
                
                time.sleep(0.01)
                
    except Exception as e:
        st.error(f"Cloud Connection Timeout or Reset: {e}")
        st.session_state.running = False
elif not selected_class_ids:
    frame_placeholder.warning("Please select at least one target class object from the sidebar selection box.")
else:
    frame_placeholder.info("Stream offline. Click 'Start Stream' to engage.")
