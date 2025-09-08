from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained YOLO model
model = YOLO("../models/model.pt")

app = FastAPI()

# Define class names (consistent with training)
CLASS_NAMES = {
    0: "Not Wearing a Mask Correctly 😷",
    1: "Wearing a Mask",
    2: "Not Wearing a Mask"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.lower().endswith((".jpg", ".jpeg")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .jpg or .jpeg images are allowed."
        )
        
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLO inference
    results = model.predict(image)

    predictions = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            predictions.append({
                "class": CLASS_NAMES.get(cls_id, f"class{cls_id}"),
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()
            })

    return JSONResponse(content={"predictions": predictions})
