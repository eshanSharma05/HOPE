import streamlit as st
import sqlite3
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import os
import time
from datetime import datetime

# --- CONFIGURATION ENGINE ---
# Maintain the list of camera channels available on your local bridge network
CHANNELS_LIST = ["Channel 3", "Channel 4"]
PLAYBACK_BASE_URL = "https://bulk-boxer-handiness.ngrok-free.dev/playback"
DB_FILE = "surveillance_vault.db"

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

# --- REUSABLE ANNOTATION ENGINE ---
import requests

def fetch_and_process_mjpeg(playback_url, cam_name, start_timestamp_str, duration_limit):
    """Fetches MJPEG frames via requests (bypassing Ngrok interstitials), runs YOLO, and saves MP4."""
    headers = {"ngrok-skip-browser-warning": "true"}
    
    try:
        response = requests.get(playback_url, headers=headers, stream=True, timeout=30)
        if response.status_code != 200:
            return False, f"Bridge returned HTTP {response.status_code}"
    except Exception as e:
        return False, f"Connection error: {e}"

    thread_model = YOLO("yolov8n.pt")
    filename = f"{cam_name}_{start_timestamp_str}.mp4"
    temp_filepath = f"temp_{filename}"
    
    out = None
    fps = 15  # Standard playback stream rate over HTTP
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    start_time = time.time()
    max_humans = 0
    max_total_objects = 0
    frames_written = 0
    byte_buffer = b""

    try:
        for chunk in response.iter_content(chunk_size=8192):
            if (time.time() - start_time) >= duration_limit:
                break
            if not chunk:
                continue

            byte_buffer += chunk
            # Locate JPEG start (0xff 0xd8) and end (0xff 0xd9) markers
            a = byte_buffer.find(b'\xff\xd8')
            b = byte_buffer.find(b'\xff\xd9')

            while a != -1 and b != -1 and b > a:
                jpg = byte_buffer[a:b+2]
                byte_buffer = byte_buffer[b+2:]

                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    if out is None:
                        h, w, _ = frame.shape
                        out = cv2.VideoWriter(temp_filepath, fourcc, fps, (w, h))

                    # Run YOLO inference & draw bounding boxes
                    results = thread_model(frame, verbose=False)
                    if results and len(results[0].boxes) > 0:
                        boxes = results[0].boxes
                        total_objs = len(boxes)
                        class_ids = boxes.cls.cpu().numpy()
                        human_objs = int((class_ids == 0).sum())

                        max_humans = max(max_humans, human_objs)
                        max_total_objects = max(max_total_objects, total_objs)

                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            label_name = thread_model.names[cls_id]

                            color = (0, 255, 0) if cls_id == 0 else (255, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            caption = f"{label_name}: {int(conf * 100)}%"
                            cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(caption) * 10, y1), color, -1)
                            cv2.putText(frame, caption, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    out.write(frame)
                    frames_written += 1

                a = byte_buffer.find(b'\xff\xd8')
                b = byte_buffer.find(b'\xff\xd9')
    finally:
        response.close()
        if out is not None:
            out.release()

    if frames_written > 0 and os.path.exists(temp_filepath):
        with open(temp_filepath, "rb") as f:
            blob_data = f.read()
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO video_chunks (filename, camera_id, timestamp, video_blob, processed, humans, objects)
                VALUES (?, ?, ?, ?, 1, ?, ?)
            ''', (filename, cam_name, start_timestamp_str, blob_data, max_humans, max_total_objects))
            conn.commit()
            conn.close()
        except sqlite3.OperationalError:
            pass
        os.remove(temp_filepath)
        return True, f"Processed {frames_written} frames."

    if os.path.exists(temp_filepath):
        os.remove(temp_filepath)
    return False, "No video frames received. Check DVR timestamp or Ngrok status."

# --- INITIALIZATION ---
st.set_page_config(page_title="DVR Playback Extraction Engine", layout="wide")
st.title("⏳ DVR Historical Playback & Analytics Dashboard")
init_db()

# --- QUERY LAYER FOR DOWNLOADS ---
if "download_id" in st.query_params:
    target_id = st.query_params["download_id"]
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, video_blob FROM video_chunks WHERE id = ?", (target_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        st.download_button(label=f"💾 Confirm Download: {row[0]}", data=row[1], file_name=row[0], mime="video/mp4", type="primary")
        st.stop()

# --- MAIN CONTROLS: SIDEBAR EXTRACTOR ---
st.sidebar.header("📥 Fetch Historical Footage")
selected_channel = st.sidebar.selectbox("Select Target Channel", CHANNELS_LIST)

# Date/Time User inputs
start_date = st.sidebar.date_input("Target Date", datetime.now())
start_time = st.sidebar.time_input("Start Time (24h format)", datetime.now().time())
duration = st.sidebar.number_input("Extraction Duration (Seconds)", min_value=5, max_value=300, value=20, step=5)

if st.sidebar.button("🧠 Extract and Process Video", type="primary", use_container_width=True):
    dt_start = datetime.combine(start_date, start_time)
    dt_end = datetime.fromtimestamp(dt_start.timestamp() + duration)
    
    start_str = dt_start.strftime("%d_%m_%Y_%H_%M_%S")
    end_str = dt_end.strftime("%d_%m_%Y_%H_%M_%S")
    file_timestamp_format = dt_start.strftime("%d%m%y_%H%M%S")
    
    channel_num = "".join(filter(str.isdigit, selected_channel))
    clean_cam_name = f"Channel{channel_num}"
    playback_url = f"{PLAYBACK_BASE_URL}?channel={channel_num}&start={start_str}&end={end_str}"
    
    with st.spinner(f"Fetching & analyzing Channel {channel_num} from DVR..."):
        success, msg = fetch_and_process_mjpeg(playback_url, clean_cam_name, file_timestamp_format, duration)
        if success:
            st.sidebar.success(f"{clean_cam_name}: {msg}")
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error(msg)

# --- MAIN UI TELEMETRY REGISTRY ---
st.subheader("📋 Historical Processing Log Registry")
if os.path.exists(DB_FILE):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query('''
        SELECT id, filename AS [Video Name], camera_id AS [Camera Origin], 
               timestamp AS [Recording Timestamp], humans AS [Number of Humans], objects AS [Number of Objects] 
        FROM video_chunks ORDER BY id DESC
    ''', conn)
    conn.close()
    
    if not df.empty:
        df["Download Link"] = df["id"].apply(lambda x: f"/?download_id={x}")
        df = df.drop(columns=["id"])
        df = df[["Download Link", "Video Name", "Camera Origin", "Recording Timestamp", "Number of Humans", "Number of Objects"]]
        
        st.data_editor(df, hide_index=True, use_container_width=True, disabled=True,
                       column_config={"Download Link": st.column_config.LinkColumn("Action", display_text="📥 Download MP4")})
    else:
        st.info("No records present. Configure target time boundaries on the left panel to fetch historical data from the DVR storage sectors.")
else:
    st.info("Database initializing.")
