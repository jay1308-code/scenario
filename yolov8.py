import streamlit as st
import time
import uuid
from ultralytics import YOLO
import tempfile
import os

# Load the YOLO model
model = YOLO("yolov8s-seg.pt")

# Define indoor and outdoor items
indoor_items = [
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard' ,'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'hair drier', 
    'person', 'chair', 'couch',
]

outdoor_items = [
    'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'boat', 'bird','kite', 'baseball bat', 
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'hot dog', 'traffic light', 'stop sign', 'parking meter', 
    'bench', 'person'
]

def save_results(result, timestamp, unique_id, context, person_detected):
    if context == 'indoor':
        folder = "indoor_person" if person_detected else "indoor"
    elif context == 'outdoor':
        folder = "outdoor_person" if person_detected else "outdoor"
    
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/result_{timestamp}_{unique_id}.jpg"
    result.save(filename=filename)
    print(f"Saved {context} {'person' if person_detected else 'object'} result to {filename}")

# Streamlit app
st.title("YOLO Video Processing App")
st.write("Upload a video and the app will process it using YOLOv8.")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.flush()  # Ensure data is written to the file
    
    st.video(tfile.name)  # Display the uploaded video
    
    # Run the YOLO model on the uploaded video
    results = model.predict(source=tfile.name, conf=0.50)
    
    for result in results:
        unique_id = str(uuid.uuid4())
        timestamp = int(time.time())
        append_list = []
        item_classes = result.boxes.cls.int().tolist()
       
        indoor_detected = False
        outdoor_detected = False
        person_detected = False

        for i in item_classes:
            append_list.append(result.names[i])

        for item in append_list:
            if item in indoor_items and item != 'person':
                indoor_detected = True
            if item in outdoor_items and item != 'person':
                outdoor_detected = True
            if item == 'person':
                person_detected = True

        if person_detected:
            if indoor_detected:
                save_results(result, timestamp, unique_id, 'indoor', True)
            elif outdoor_detected:
                save_results(result, timestamp, unique_id, 'outdoor', True)
        else:
            if indoor_detected:
                save_results(result, timestamp, unique_id, 'indoor', False)
            if outdoor_detected:
                save_results(result, timestamp, unique_id, 'outdoor', False)
    
    # Display success message
    st.success("Processing complete!")
else:
    st.info("Please upload a video file.")
