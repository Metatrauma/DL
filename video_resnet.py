import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import os

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

desired_classes = ['traffic light', 'car', 'motorcycle']

def detect_objects(image, confidence_threshold=0.6):
    original_image = image.copy()
    image_tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    for i, box in enumerate(boxes):
        if scores[i] >= confidence_threshold:
            label = COCO_Instance_Category_names[labels[i]]
            if label in desired_classes:
                start_point = (int(box[0]), int(box[1]))
                end_point = (int(box[2]), int(box[3]))
                color = (0, 255, 0)  # Green bounding box
                cv2.rectangle(original_image, start_point, end_point, color, 2)
                cv2.putText(original_image, f"{label}: {scores[i]:.2f}",
                            (start_point[0], start_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return original_image

# Open video stream
video_path = "/Users/samahita/Downloads/vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4"
cap = cv2.VideoCapture(video_path)

# Create a folder to save output images
output_folder = "//Users/samahita/Documents/GitHub/DL/outframes"
os.makedirs(output_folder, exist_ok=True)

frame_count = 0
max_frames = 10  # Limit to 10 frames

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the current frame
    detected_frame = detect_objects(frame)

    # Save the processed frame
    output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, detected_frame)

    frame_count += 1

cap.release()
print(f"Processed frames saved to {output_folder}")
