# app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Handwritten Recognizer", page_icon="✍️", layout="wide")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    digit_model = tf.keras.models.load_model('digit_recognizer.h5')
    alphabet_model = tf.keras.models.load_model('alphabet_recognizer.h5')
    return digit_model, alphabet_model

digit_model, alphabet_model = load_models()

# --- WEB APP INTERFACE ---
st.title("✍️ Handwritten Digit & Alphabet Recognizer")

# --- USER SELECTION ---
st.sidebar.header("Select Mode")
mode = st.sidebar.radio("Choose what you want to recognize:", ('Digit', 'Alphabet'))

# --- DRAWING CANVAS ---
st.subheader(f"Draw a single {mode.lower()} on the canvas")
canvas_result = st_canvas(
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# --- PREDICTION LOGIC ---
if st.button(f'Recognize {mode}'):
    if canvas_result.image_data is not None:
        # Preprocess the image
        image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
        image_resized = image.resize((28, 28))
        img_array = np.array(image_resized) / 255.0
        img_reshaped = img_array.reshape(1, 28, 28, 1)

        # --- CHOOSE MODEL AND PREDICT ---
        if mode == 'Digit':
            prediction = digit_model.predict(img_reshaped)
            result = np.argmax(prediction)
        else: # Alphabet
            prediction = alphabet_model.predict(img_reshaped)
            # Map the prediction index to a letter (0=A, 1=B, etc.)
            alphabet_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            result = alphabet_map[np.argmax(prediction)]
        
        confidence = np.max(prediction)

        # --- DISPLAY RESULT ---
        st.success(f'The model predicts: {result}')
        st.metric(label="Confidence", value=f"{confidence:.2%}")