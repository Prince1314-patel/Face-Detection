import streamlit as st
import cv2
from PIL import Image, ImageDraw
import numpy as np
import tempfile
import io

# Function to detect faces and draw rectangles without changing the image color
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    for (x, y, w, h) in faces:
        draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
    return pil_image


# Streamlit app
st.title("Face Detection App")

# Option selection
option = st.selectbox("Choose an option:", ["Image Upload", "Webcam", "Video Upload"])

if option == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        # Detect faces
        processed_image = detect_faces(image_array)

        # Display the processed image
        st.image(processed_image, caption='Processed Image', use_column_width=True)

        # Save the processed image to a BytesIO object
        img_bytes = io.BytesIO()
        processed_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        # Provide download link
        st.download_button(label='Download Image', data=img_bytes, file_name='detected_faces.jpg', mime='image/jpeg')

elif option == "Webcam":
    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    
    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = detect_faces(frame)
        FRAME_WINDOW.image(processed_frame)
    else:
        st.write('Stopped')

elif option == "Video Upload":
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        vf = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = detect_faces(frame)
            stframe.image(processed_frame, channels='RGB')
