import streamlit as st
import cv2
import sqlite3
import os
import time
import threading
from datetime import datetime

# --- CONFIGURATION ENGINE ---
CAMERA_CONFIG = {
    "Camera_1": "rtsp://admin:password@your_public_or_ngrok_endpoint:554/cam/realmonitor?channel=1&subtype=1",
    "Camera_2": "rtsp://admin:password@your_public_or_ngrok_endpoint:554/cam/realmonitor?channel=2&subtype=1"
}
VIDEO_DURATION_SECS = 40
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

def record_camera_loop(cam_name, rtsp_url):
    while True:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            time.sleep(5)
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 20
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 0 else 640
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > 0 else 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        while cap.isOpened():
            timestamp_str = datetime.now().strftime("%d%m%y_%H%M%S")
            filename = f"{cam_name}_{timestamp_str}.mp4"
            temp_filepath = f"temp_{filename}"
            
            out = cv2.VideoWriter(temp_filepath, fourcc, fps, (frame_width, frame_height))
            start_time = time.time()
            
            ret = True
            while (time.time() - start_time) < VIDEO_DURATION_SECS:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                
            out.release()
            
            # If the recording window completed successfully, stream binary to database
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
                    pass # Database locked temporarily, skip chunk to prevent buffer locks
                
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                    
            if not ret:
                break
                
        cap.release()
        time.sleep(2)

# --- HEADLESS UI INTERFACE ---
st.set_page_config(page_title="Headless Camera Worker", layout="centered")
st.title("⚙️ Cloud Camera Recording Engine")
st.info("This application runs headlessly in the cloud background. No streaming playback is rendered here.")

init_db()

if "threads_initialized" not in st.session_state:
    st.session_state.threads_initialized = True
    for camera_id, stream_path in CAMERA_CONFIG.items():
        t = threading.Thread(target=record_camera_loop, args=(camera_id, stream_path), daemon=True)
        t.start()

# Keep instance warm
status_placeholder = st.empty()
while True:
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM video_chunks")
        count = cursor.fetchone()[0]
        conn.close()
        status_placeholder.metric("Total Video Chunks Stored in Cloud DB", count)
    except Exception:
        pass
    time.sleep(5)
    st.rerun()
