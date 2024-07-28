import streamlit as st
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepface import DeepFace

def detect_faces(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def analyze_image(image, faces):
    rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = DeepFace.analyze(rgb_image, actions=['age', 'gender', 'emotion'], enforce_detection=False)
    
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 18)
    
    for face in results:
        (x, y, w, h) = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        age = face['age']
        gender = face['gender']
        emotion = face['dominant_emotion']
        
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        draw.text((x, y - 20), f"{gender}, {age}", fill="red", font=font)
        draw.text((x, y + h + 5), emotion, fill="blue", font=font)
        
    return image

st.title("Advanced Face Detection App")

option = st.sidebar.selectbox(
    "Choose an option",
    ("Image Upload", "Webcam", "Video Upload")
)

if option == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        faces = detect_faces(image)
        if faces is not None:
            result_image = analyze_image(image, faces)
            st.image(result_image, caption='Processed Image', use_column_width=True)

elif option == "Webcam":
    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    while run:
        _, frame = camera.read()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = detect_faces(image)
        if faces is not None:
            result_image = analyze_image(image, faces)
            FRAME_WINDOW.image(result_image)
    else:
        st.write('Stopped')

elif option == "Video Upload":
    uploaded_file = st.file_uploader("Choose a video...", type="mp4")
    if uploaded_file is not None:
        video = cv2.VideoCapture(uploaded_file.name)
        stframe = st.empty()
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            faces = detect_faces(image)
            if faces is not None:
                result_image = analyze_image(image, faces)
                stframe.image(result_image)

