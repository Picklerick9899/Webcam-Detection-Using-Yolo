import streamlit as st

import cv2

import cvzone

import numpy as np

from ultralytics import YOLO

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase



# Load YOLO model

model = YOLO("Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",

       "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",

       "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",

       "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",

       "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",

       "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",

       "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",

       "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",

       "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",

       "teddy bear", "hair drier", "toothbrush"]



# Streamlit Title

st.title("Live YOLO Object Detection")

st.sidebar.header("Webcam Settings")



# Define Video Processor

class YOLOVideoProcessor(VideoProcessorBase):

  def recv(self, frame):

    img = frame.to_ndarray(format="bgr24")



    # Perform YOLO Detection

    results = model(img, stream=True)



    for r in results:

      boxes = r.boxes

      for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h))

        conf = round(float(box.conf[0]), 2)

        cls = int(box.cls[0])

        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (x1, max(35, y1)), scale=1, thickness=1)



    return frame.from_ndarray(img, format="bgr24")



# Start Webcam Stream

webrtc_streamer(key="webcam", video_processor_factory=YOLOVideoProcessor)

