import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
import io

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Face Mask Detection", layout="wide")
st.title("😷 Face Mask Detection with YOLO + FastAPI + Streamlit")

uploaded_file = st.file_uploader(
    "Upload an image (JPG/JPEG only)", 
    type=["jpg", "jpeg", "JPG", "JPEG"]
)

if uploaded_file is not None:
    # Normalize filename extension (to avoid case issues)
    filename = uploaded_file.name.lower()
    if not (filename.endswith(".jpg") or filename.endswith(".jpeg")):
        st.error("❌ Invalid file type. Please upload a JPG or JPEG image.")
    else:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to bytes for API request
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        # Call FastAPI backend
        with st.spinner("Analyzing image..."):
            response = requests.post(API_URL, files={"file": ("image.jpg", img_bytes, "image/jpeg")})

        if response.status_code == 200:
            predictions = response.json().get("predictions", [])

            if predictions:
                st.subheader("✅ Predictions")
                for pred in predictions:
                    st.write(
                        f"**Class**: {pred['class']} | "
                        f"**Confidence**: {pred['confidence']:.2f} | "
                        f"**BBox**: {pred['bbox']}"
                    )

                # Draw bounding boxes on image
                np_img = np.array(image)
                for pred in predictions:
                    x1, y1, x2, y2 = map(int, pred["bbox"])
                    label = f"{pred['class']} {pred['confidence']:.2f}"

                    cv2.rectangle(np_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        np_img,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                st.subheader("Detected Image")
                st.image(np_img, caption="YOLO Predictions", use_column_width=True)

            else:
                st.warning("No objects detected.")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
