import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. UI Configuration
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")
st.title("🌱 Plant Disease CNN Engine")
st.write("Upload a leaf image to identify the crop and potential diseases.")

# 2. Load the Bundled Model (Cached to prevent reloading on every interaction)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('./plant_disease_cnn.keras')

model = load_model()

# 3. Define the Class Names (Must match the exact alphabetical order of your Colab folders)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
]

# 4. Input Stream
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Target Image', use_column_width=True)
    
    # 5. Preprocessing Pipeline (Matches the Colab training transforms)
    image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0         
    img_batch = np.expand_dims(img_array, axis=0) 
    
    # 6. Inference Engine
    st.write("Running inference...")
    predictions = model.predict(img_batch)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    # 7. Output Routing
    st.success(f"**Classification:** {CLASS_NAMES[predicted_class_index]}")
    st.info(f"**Confidence Score:** {confidence:.2f}%")