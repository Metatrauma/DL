

import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
# Import cv2_imshow from google.colab.patches
#from google.colab.patches import cv2_imshow
import numpy as np

# Pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO dataset class names
COCO_Instance_Category_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'toilet', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define the classes you want to detect
desired_classes = ['traffic light', 'car', 'motorcycle']

def detect_objects(image_path, confidence_threshold=0.6):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}.")
        return None

    original_image = image.copy()
    image_tensor = F.to_tensor(image)  # Convert image to tensor

    # Run object detection
    with torch.no_grad():
        predictions = model([image_tensor])

    # Extract predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    # Draw bounding boxes for detections of desired classes
    for i, box in enumerate(boxes):
        if scores[i] >= confidence_threshold:
            label = COCO_Instance_Category_names[labels[i]]
            if label in desired_classes:  # Filter detections by class
                start_point = (int(box[0]), int(box[1]))
                end_point = (int(box[2]), int(box[3]))
                color = (0, 255, 0)  # Green bounding box
                cv2.rectangle(original_image, start_point, end_point, color, 2)

                # Draw label and confidence score
                cv2.putText(original_image, f"{label}: {scores[i]:.2f}",
                            (start_point[0], start_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return original_image

# Test the function
if __name__ == "__main__":
    image_path = "/Users/samahita/Documents/GitHub/DL/traffic.jpg"  
    detected_image = detect_objects(image_path)

    # Display the resulting image
    if detected_image is not None:

        output_path = "/Users/samahita/Documents/GitHub/DL/output_image.jpg"

        cv2.imwrite(output_path, detected_image)
        print("Image saved as 'output_image.jpg'.")

        cv2.imshow("Detected Image", detected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: No detected image to display.")

