import streamlit as st
from PIL import Image
import io
import torch
from ultralytics import YOLO
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# Placeholder for your YOLO model's inference function
def identify_disease(image):
    model = YOLO(f'best.pt')
    result = model.predict("/home/rohan-verma/prerit.jpg")
    # For demonstration, let's return a dummy disease name
    return result

# Streamlit UI
st.title("Disease Detection Using YOLO")

# Section for image upload
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform disease identification
    with st.spinner("Identifying the disease..."):
        disease = identify_disease(image)

    # Display the result
    st.success("Disease Identified!")
    st.write(f"**Disease:** {disease}")
