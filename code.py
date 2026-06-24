import streamlit as st
import sqlite3
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import os
import time
import threading
from datetime import datetime

# --- CONFIGURATION ENGINE ---
CAMERA_CONFIG = {
    "Channel3": "http://bulk-boxer-handiness.ngrok-free.dev/video?channel=3",
    "Channel4": "http://bulk-boxer-handiness.ngrok-free.dev/video?channel=4"
}
VIDEO_DURATION_SECS = 10  # Cut exactly 10-second videos
GAP_DURATION_SECS = 30    # Wait 30 seconds before cutting the next video
DB_FILE = "surveillance_vault.db"

# Initialize SQLite database schema
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS video_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            camera_id TEXT,
            timestamp TEXT,
            video_blob BLOB,
            processed INTEGER DEFAULT 0,
            humans INTEGER DEFAULT 0,
            objects INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

# Global Native Python Event for Thread Signaling
if "stop_signal" not in st.session_state:
    st.session_state.stop_signal = threading.Event()

# --- BACKGROUND RECORDER & VISUAL LABELER WORKER ---
def record_camera_worker(cam_name, rtsp_url, stop_event):
    """Headless loop that captures video, draws YOLO annotations, saves data, then sleeps."""
    # Load an isolated model instance inside the thread to prevent cross-thread memory corruption
    thread_model = YOLO("yolov8n.pt")
    
    while not stop_event.is_set():
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            time.sleep(5)
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 20
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 0 else 640
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > 0 else 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        while cap.isOpened() and not stop_event.is_set():
            timestamp_str = datetime.now().strftime("%d%m%y_%H%M%S")
            filename = f"{cam_name}_{timestamp_str}.mp4"
            temp_filepath = f"temp_{filename}"
            
            out = cv2.VideoWriter(temp_filepath, fourcc, fps, (frame_width, frame_height))
            start_time = time.time()
            
            max_humans = 0
            max_total_objects = 0
            ret = True
            
            # --- Active 10-Second Recording & Inference Phase ---
            while (time.time() - start_time) < VIDEO_DURATION_SECS:
                if stop_event.is_set():
                    ret = False
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLO Object Inference on the frame matrix
                results = thread_model(frame, verbose=False)
                
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    total_objs = len(boxes)
                    class_ids = boxes.cls.cpu().numpy()
                    human_objs = int((class_ids == 0).sum())
                    
                    # Track peak counts found within this 10s video block
                    if human_objs > max_humans:
                        max_humans = human_objs
                    if total_objs > max_total_objects:
                        max_total_objects = total_objs
                    
                    # --- Render Bounding Boxes & Text Overlays ---
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label_name = thread_model.names[cls_id]
                        
                        # Set color: Neon green for humans, neon blue for other objects
                        color = (0, 255, 0) if cls_id == 0 else (255, 255, 0)
                        
                        # Draw bounding box rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Construct label banner text (e.g., "person: 84%")
                        caption = f"{label_name}: {int(conf * 100)}%"
                        
                        # Draw background banner text rectangle for clarity
                        cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(caption) * 10, y1), color, -1)
                        cv2.putText(frame, caption, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Write the annotated frame directly to the video container
                out.write(frame)
                
            out.release()
            
            # Save the processed video chunk to the SQLite database
            if ret and os.path.exists(temp_filepath):
                with open(temp_filepath, "rb") as f:
                    blob_data = f.read()
                
                try:
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    # Directly store data with processed metrics calculated during recording
                    cursor.execute('''
                        INSERT INTO video_chunks (filename, camera_id, timestamp, video_blob, processed, humans, objects)
                        VALUES (?, ?, ?, ?, 1, ?, ?)
                    ''', (filename, cam_name, timestamp_str, blob_data, max_humans, max_total_objects))
                    conn.commit()
                    conn.close()
                except sqlite3.OperationalError:
                    pass
                
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
            
            if not ret:
                break
                
            # --- 30-Second Passive Idle Phase ---
            gap_start = time.time()
            while (time.time() - gap_start) < GAP_DURATION_SECS:
                if stop_event.is_set():
                    break
                time.sleep(1)
                
        cap.release()
        time.sleep(1)

# --- DASHBOARD LAYOUT & INITIALIZATION ---
st.set_page_config(page_title="Dynamic Surveillance Fleet Dashboard", layout="wide")
st.title("🛡️ Automated Multi-Camera Analytics Fleet Dashboard")

init_db()

if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False

# --- DOWNLOAD PARAMETERS INTERCEPTOR ---
query_params = st.query_params
if "download_id" in query_params:
    target_id = query_params["download_id"]
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, video_blob FROM video_chunks WHERE id = ?", (target_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        filename, blob_data = row
        st.download_button(
            label=f"💾 Click to Confirm Download: {filename}",
            data=blob_data,
            file_name=filename,
            mime="video/mp4",
            type="primary"
        )
        st.stop()

# --- INTERACTIVE CONTROL BUTTONS ---
st.sidebar.header("Pipeline Controls")
if st.session_state.pipeline_running:
    if st.sidebar.button("⏹️ Stop Stream & Processing", type="primary"):
        st.session_state.pipeline_running = False
        st.session_state.stop_signal.set()
        st.rerun()
else:
    if st.sidebar.button("▶️ Start Stream & Processing", type="secondary"):
        st.session_state.pipeline_running = True
        st.session_state.stop_signal.clear()
        
        for camera_id, stream_path in CAMERA_CONFIG.items():
            t = threading.Thread(
                target=record_camera_worker, 
                args=(camera_id, stream_path, st.session_state.stop_signal), 
                daemon=True
            )
            t.start()
        st.rerun()

status_placeholder = st.sidebar.empty()
if st.session_state.pipeline_running:
    status_placeholder.success("🟢 System Active: Processing Logs...")
else:
    status_placeholder.error("🔴 System Offline.")

# --- UI DATA RENDERING MATRIX ---
st.subheader("📋 Historical Processing Log Registry")

if os.path.exists(DB_FILE):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query('''
        SELECT id,
               filename AS [Video Name], 
               camera_id AS [Camera Origin], 
               timestamp AS [Recording Timestamp], 
               humans AS [Number of Humans], 
               objects AS [Number of Objects] 
        FROM video_chunks 
        WHERE processed = 1 
        ORDER BY id DESC
    ''', conn)
    conn.close()
    
    if not df.empty:
        df["Download Link"] = df["id"].apply(lambda x: f"/?download_id={x}")
        df = df.drop(columns=["id"])
        columns_order = ["Download Link", "Video Name", "Camera Origin", "Recording Timestamp", "Number of Humans", "Number of Objects"]
        df = df[columns_order]
        
        st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Download Link": st.column_config.LinkColumn(
                    "Action",
                    display_text="📥 Download MP4"
                )
            },
            disabled=True
        )
    else:
        st.info("Awaiting clip segments. Records will appear once the first 10-second block closes.")
else:
    st.info("Database initializing.")

if st.session_state.pipeline_running:
    time.sleep(5)
    st.rerun()
