# app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="✍️", layout="wide")

# --- LOAD THE TRAINED MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('digit_recognizer.h5')
model = load_model()

# --- WEB APP INTERFACE ---
st.title("✍️ Handwritten Digit Recognizer")
st.markdown("Draw a digit (0-9) on the canvas, and the AI will predict what it is.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Drawing Canvas")
    # Create a canvas component for drawing
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    if st.button('Predict'):
        if canvas_result.image_data is not None:
            # 1. Get the drawing from the canvas
            img_data = canvas_result.image_data

            # 2. Process the image to match the model's training data format. This is a crucial step.
            image = Image.fromarray(img_data.astype('uint8'), 'RGBA').convert('L')
            image_resized = image.resize((28, 28))
            img_array = np.array(image_resized) / 255.0
            img_reshaped = img_array.reshape(1, 28, 28, 1)

            # 3. Make a prediction
            prediction = model.predict(img_reshaped)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)

            # 4. Display the result
            st.metric(label="Predicted Digit", value=f"{predicted_digit}")
            st.metric(label="Confidence", value=f"{confidence:.2%}")
            st.image(image_resized, caption='Processed 28x28 Image')