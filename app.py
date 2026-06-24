import streamlit as st
import sqlite3
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import os
import time

DB_FILE = "surveillance_vault.db"

@st.cache_resource
def load_yolo_network():
    return YOLO("yolov8n.pt")

model = load_yolo_network()

st.set_page_config(page_title="Surveillance Analytics Dashboard", layout="wide")
st.title("🛡️ Automated Multi-Camera Analytics Fleet Dashboard")

def process_stored_blobs():
    """Fetches unprocessed video data arrays directly out of database memory."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, video_blob FROM video_chunks WHERE processed = 0")
    unprocessed_jobs = cursor.fetchall()
    conn.close()
    
    for job_id, filename, video_blob in unprocessed_jobs:
        # Write binary chunk back to a temporary local disk sector for OpenCV to parse
        temp_file = f"process_target_{filename}"
        with open(temp_file, "wb") as f:
            f.read(video_blob) if hasattr(video_blob, 'read') else f.write(video_blob)
            
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
            
        # Update entry with processing metrics
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE video_chunks 
            SET processed = 1, humans = ?, objects = ? 
            WHERE id = ?
        ''', (max_humans, max_total_objects, job_id))
        conn.commit()
        conn.close()

# --- RUN PROCESSING PASSTHROUGH ---
if os.path.exists(DB_FILE):
    process_stored_blobs()

# --- UI DATA RENDERING MATRIX ---
st.subheader("📋 Historical Processing Log Registry")

if os.path.exists(DB_FILE):
    conn = sqlite3.connect(DB_FILE)
    # Fetch processed tracking metadata logs cleanly
    df = pd.read_sql_query('''
        SELECT filename AS [Video Name], 
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
        # Display the text-based summary metrics
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Awaiting file outputs. Records will appear once the recording background engine processes chunks.")
else:
    st.info("Database initializing. Run background_recorder.py to start processing telemetry data rows.")

# Automated refresh pace configuration
time.sleep(5)
st.rerun()
