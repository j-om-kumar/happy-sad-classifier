import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import numpy as np

# Set the title of the app
st.title("Upload an Image")

# Add a file uploader to allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    resize = tf.image.resize(image,(256,256))
    new_model = load_model(os.path.join('models','happy-sad-imageclassifier.h5'))
    output = new_model.predict(np.expand_dims(resize/255,0))
    st.write(output)
    if output > 0.5:
        st.success(f'Predicted class is Sad')
    else:
        st.success(f'Predicted class is Happy')
    
else:
    st.info("Please upload an image file.")
