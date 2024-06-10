# YOLO Video Processing App

This Streamlit app allows users to upload a video and process it using the YOLOv8 model. The app detects objects in the video and saves the results into appropriate folders.

## Requirements

Ensure you have Python installed. Then, install the necessary dependencies using:

1-pip install -r requirements.txt

Running the App
Save the provided script as yolov8.py.
Run the Streamlit app using the following command:
2-streamlit run yolov8.py

Once the app is running, upload a video file (supported formats: mp4, avi, mov).
How It Works

Upload a video file through the Streamlit interface.

The app will process the video using the YOLOv8 model.
Based on the detections, the app will save the results into specific folders:

indoor_person for indoor scenes with people.
indoor for indoor scenes without people.
outdoor_person for outdoor scenes with people.
outdoor for outdoor scenes without people.


