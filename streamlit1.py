import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import tensorflow

# Load the model
def load_model():
    model = tensorflow.keras.models.load_model('best_model1.keras')
    return model

model = load_model()

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize the image
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    img_array = preprocess_input(img_array)  # Preprocess the image
    return img_array

# Streamlit app
st.title("Medical Image Diagnosis")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Display uploaded image and perform diagnosis
if uploaded_file is not None:
    # Read the image file
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(img)

    # Make predictions
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    class_confidence = prediction[0][predicted_class]

    # Get the class label
    class_label = "Class " + str(predicted_class)

    # Display diagnosis result
    st.write("Diagnosis:", class_label)
    st.write("Confidence:", class_confidence)
