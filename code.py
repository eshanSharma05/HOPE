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
VIDEO_DURATION_SECS = 10
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

# --- GLOBAL THREAD SIGNAL CONTROL ---
# Use a global native Python Event instead of Streamlit Session State for the threads
if "stop_signal" not in st.session_state:
    st.session_state.stop_signal = threading.Event()

# --- HEADLESS BACKGROUND RECORDER THREAD ---
def record_camera_worker(cam_name, rtsp_url, stop_event):
    """Headless background loop that records for 10s, then sleeps for 30s."""
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
            
            # Loop 1: Record actively for 10 seconds
            ret = True
            while (time.time() - start_time) < VIDEO_DURATION_SECS:
                if stop_event.is_set():
                    ret = False
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                
            out.release()
            
            # Commit the 10-second clip to the SQLite database
            if ret and os.path.exists(temp_filepath):
                with open(temp_filepath, "rb") as f:
                    blob_data = f.read()
                
                try:
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO video_chunks (filename, camera_id, timestamp, video_blob)
                        VALUES (?, ?, ?, ?)
                    ''', (filename, cam_name, timestamp_str, blob_data))
                    conn.commit()
                    conn.close()
                except sqlite3.OperationalError:
                    pass
                
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
            
            if not ret:
                break
                
            # Loop 2: Idle countdown gap. Check stop_event every second to keep the STOP button responsive.
            gap_start = time.time()
            while (time.time() - gap_start) < GAP_DURATION_SECS:
                if stop_event.is_set():
                    break
                time.sleep(1)
                
        cap.release()
        time.sleep(1)
# --- YOLO DETECTION INFERENCE PASS ---
def process_stored_blobs(model):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, video_blob FROM video_chunks WHERE processed = 0")
    unprocessed_jobs = cursor.fetchall()
    conn.close()
    
    for job_id, filename, video_blob in unprocessed_jobs:
        temp_file = f"process_target_{filename}"
        with open(temp_file, "wb") as f:
            f.write(video_blob)
            
        cap = cv2.VideoCapture(temp_file)
        max_humans = 0
        max_total_objects = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model(frame, verbose=False)
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                total_objs = len(boxes)
                class_ids = boxes.cls.cpu().numpy()
                human_objs = int((class_ids == 0).sum())
                
                if human_objs > max_humans:
                    max_humans = human_objs
                if total_objs > max_total_objects:
                    max_total_objects = total_objs
                    
        cap.release()
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE video_chunks SET processed = 1, humans = ?, objects = ? WHERE id = ?
        ''', (max_humans, max_total_objects, job_id))
        conn.commit()
        conn.close()

# --- DASHBOARD LAYOUT & INITIALIZATION ---
st.set_page_config(page_title="Dynamic Surveillance Fleet Dashboard", layout="wide")
st.title("🛡️ Automated Multi-Camera Analytics Fleet Dashboard")

init_db()
model = YOLO("yolov8n.pt")

if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False

# --- DOWNLOAD LOGIC ENGINE ---
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
        # Set the signal to force all background threads to exit cleanly
        st.session_state.stop_signal.set()
        st.rerun()
else:
    if st.sidebar.button("▶️ Start Stream & Processing", type="secondary"):
        st.session_state.pipeline_running = True
        # Clear the signal so threads can run freely
        st.session_state.stop_signal.clear()
        
        for camera_id, stream_path in CAMERA_CONFIG.items():
            # Pass the native Python event object directly into the thread parameters
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
    process_stored_blobs(model)
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
        st.info("Awaiting clip segments. Records will appear once the first 40-second block closes.")
else:
    st.info("Database initializing.")

if st.session_state.pipeline_running:
    time.sleep(5)
    st.rerun()
