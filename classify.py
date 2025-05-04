import cv2
import json
import os
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from centroid_tracker import CentroidTracker  # 🧠 Your own tracker

# Load models
yolo_model = YOLO("yolov8n.pt")  # Use yolov8n.pt or your preferred model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").cpu()

# Caption generator
def get_blip_caption(cropped_image):
    pil_image = Image.fromarray(cropped_image)
    inputs = blip_processor(pil_image, return_tensors="pt").to("cpu")
    output = blip_model.generate(**inputs)
    return blip_processor.decode(output[0], skip_special_tokens=True)

# Create directory to save JSON outputs
os.makedirs("frame_jsons", exist_ok=True)

# Main function
def track_and_describe(video_path, interval_seconds=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * interval_seconds)

    tracker = CentroidTracker()
    frame_idx = 0
    described_ids = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval_frames == 0:
            results = yolo_model(frame, device="cpu")[0]
            detections = []
            boxes_for_tracking = []

            for box in results.boxes:
                if box.conf < 0.4:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = yolo_model.names[int(box.cls)]
                detections.append({'bbox': (x1, y1, x2, y2), 'label': label})
                boxes_for_tracking.append((x1, y1, x2, y2))

            objects = tracker.update(boxes_for_tracking)

            print(f"\n🧠 Frame {frame_idx}:")
            frame_data = {
                "frame": frame_idx,
                "objects": []
            }

            for object_id, centroid in objects.items():
                try:
                    (x1, y1, x2, y2) = boxes_for_tracking[object_id]
                    label = detections[object_id]['label']
                except IndexError:
                    continue

                if object_id not in described_ids:
                    cropped = frame[y1:y2, x1:x2]
                    description = get_blip_caption(cropped)
                    described_ids[object_id] = description
                else:
                    description = described_ids[object_id]

                print(f"Object ID {object_id} ({label}) at ({x1}, {y1}, {x2}, {y2}) - Description: {description}")

                frame_data["objects"].append({
                    "id": object_id,
                    "label": label,
                    "position": [x1, y1, x2, y2],
                    "description": description
                })

            # Save this frame's data to JSON
            json_path = f"frame_jsons/frame_{frame_idx}.json"
            with open(json_path, "w") as f:
                json.dump(frame_data, f, indent=4)
            print(f"💾 Saved frame data to {json_path}")

        frame_idx += 1

    cap.release()

# Run it
track_and_describe("video.mp4", interval_seconds=2)
