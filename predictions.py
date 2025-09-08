from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("/Users/faiq/Desktop/STUDY MATERIAL/MLOPs/Projects/Face Mask Detection/models/model.pt")

# Run prediction on an image
results = model.predict(
    source="/Users/faiq/Desktop/STUDY MATERIAL/MLOPs/Projects/Face Mask Detection/random_images/test_image_2.jpg",
    save=True,               # saves predictions with bounding boxes
    conf=0.25,               # confidence threshold
    show=False               # set True if you want OpenCV windows popup
)

# Print results summary
for r in results:
    print(r.boxes)   # bounding box info
    print(r.names)   # class names mapping

# If you want to visualize using OpenCV
for r in results:
    im_array = r.plot()  # BGR numpy array with annotations
    cv2.imshow("Prediction", im_array)
    cv2.waitKey(0)

cv2.destroyAllWindows()
