import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import numpy as np
import os

# Load the pre-trained model
model_path = os.path.join("models", "happy-sad-imageclassifier.h5")
new_model = load_model(model_path)

# Prediction function
def predict_image(image_array):
    # Resize the image to the model's expected input shape
    resized_image = tf.image.resize(image_array, (256, 256))
    # Normalize the image
    normalized_image = resized_image / 255.0
    # Expand dimensions to match model input
    prediction = new_model.predict(np.expand_dims(normalized_image, axis=0))
    return prediction

# Video transformer class for live webcam
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None

    def transform(self, frame):
        self.latest_frame = frame.to_ndarray(format="bgr24")
        return self.latest_frame

# Streamlit app title
st.title("Webcam Image Classifier: Happy or Sad")

# Start webcam stream
ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if ctx.video_transformer:
    # Capture the latest frame from the webcam
    frame = ctx.video_transformer.latest_frame

    if frame is not None:
        # Display the current frame
        st.image(frame, channels="BGR", caption="Live Webcam Frame")

        # Add a button to capture and predict
        if st.button("Capture & Predict"):
            # Convert the BGR frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL image
            pil_image = Image.fromarray(rgb_frame)
            st.image(pil_image, caption="Captured Image")

            # Predict using the model
            prediction = predict_image(rgb_frame)
            st.write(prediction)

            # Interpret the prediction
            if prediction > 0.5:
                st.success("Predicted class is Sad")
            else:
                st.success("Predicted class is Happy")
