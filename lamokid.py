import base64
import os
from datetime import datetime

import cv2
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Configuration ---
SAVE_DIRECTORY = "uploaded_images"
SAVE_DIRECTORY1 = "processed_images"
model_path = r"runs\train\exp\weights\last.pt"  # Update to your actual path

# --- YOLOv5 Model Initialization ---
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    print(f"Loaded YOLOv5 model from {model_path}")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    exit(1)  # Exit if model loading fails

# --- FastAPI Setup ---
app = FastAPI()


class ImagePayload(BaseModel):
    image: str
    device_id: str
    
class SensorData(BaseModel):
    device_id: str
    water_level: int
    humidity: float
    temperature: float
    waterTemp: float


# --- Upload and Object Detection Endpoint ---
@app.post("/upload")
async def upload_image(payload: ImagePayload):
    try:
        # Create device-specific directory
        device_dir = os.path.join(SAVE_DIRECTORY, payload.device_id)
        os.makedirs(device_dir, exist_ok=True)
        device_dir1 = os.path.join(SAVE_DIRECTORY1, payload.device_id)
        os.makedirs(device_dir1, exist_ok=True)

        # Unique filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{payload.device_id}_{timestamp}.jpg"
        filepath = os.path.join(device_dir, filename)

        # Decode and save image
        image_data = base64.b64decode(payload.image)
        with open(filepath, "wb") as f:
            f.write(image_data)

        # --- YOLOv5 Object Detection ---
        image = cv2.imread(filepath)
        results = model(image)

        # Draw bounding boxes on the image
        for *xyxy, conf, cls in results.xyxy[0]:
            label = f'{results.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(image, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the image with bounding boxes
        processed_filename = f"processed_{payload.device_id}_{timestamp}.jpg"
        processed_filepath = os.path.join(device_dir1, processed_filename)
        cv2.imwrite(processed_filepath, image)

        detected_objects = [
            {"confidence": float(conf), "class": results.names[int(cls)]}
            for *_, conf, cls in results.xyxy[0]
        ]

        response_data = {
            "detections": detected_objects,
            "count": len(detected_objects),
            "status": "success",
            "message": "Image uploaded and processed successfully",
            "filename": filename,
            "processed_filename": processed_filename
        }
        # Print the response data as a list of key-value pairs
        for key, value in response_data.items():
            print(f"{key}: {value}")

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/sensor-data")
async def receive_sensor_data(sensor_data: SensorData):
    try:
        # Log the sensor data
        print(f"Received sensor data from {sensor_data.device_id}")
        print(f"Water Level: {sensor_data.water_level}")
        print(f"Humidity: {sensor_data.humidity}")
        print(f"Temperature: {sensor_data.temperature}")
        print(f"Water Temperature: {sensor_data.waterTemp}")
        return {"message": "Sensor data received"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


# --- Start the server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)