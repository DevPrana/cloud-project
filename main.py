import streamlit as st
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO

# Print torch setup information
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# Function to identify disease and return relevant results
def identify_disease(image_path):
    model = YOLO('best.pt')  # Load your YOLO model
    results = model.predict(image_path)  # Perform prediction
    return results[0]  # Return the first result (assuming single image input)

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, names):
    draw = ImageDraw.Draw(image)
    for box in boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        cls_name = names[int(cls)]
        label = f"{cls_name} ({conf:.2f})"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), label, fill="red")
    return image

# Streamlit UI
st.title("Disease Detection Using YOLO")

# Section for image upload
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image to a temporary location
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Perform disease identification
    with st.spinner("Identifying the disease..."):
        result = identify_disease(temp_image_path)

    # Extract and display details from result
    st.success("Disease Identified!")
    boxes = result.boxes
    disease_names = result.names

    # Extract disease with highest confidence
    if boxes.data.size(0) > 0:  # Check if boxes exist
        top_conf_idx = torch.argmax(boxes.conf)  # Get the index of the highest confidence
        top_cls = int(boxes.cls[top_conf_idx])  # Class of the highest confidence box
        top_disease = disease_names[top_cls]
        top_confidence = boxes.conf[top_conf_idx].item()

        st.write(f"**Top Disease:** {top_disease}")
        st.write(f"**Confidence:** {top_confidence:.2f}")

        # Draw bounding boxes on the image
        image = Image.open(temp_image_path)
        annotated_image = draw_boxes(image.copy(), boxes, disease_names)

        # Display the image with bounding boxes
        st.image(annotated_image, caption="Detected Diseases", use_column_width=True)
    else:
        st.write("No diseases detected.")
