# SenangPark YOLO – Smart Parking Detection System

An AI-powered smart parking detection system that uses YOLO (You Only Look Once) object detection to detect vehicles in parking areas and support parking-slot occupancy monitoring.

## About the Project

**SenangPark YOLO** is a computer vision project designed to make parking-space monitoring more efficient through automated vehicle detection.

The system uses a trained YOLO model to identify cars in parking-area images or video. Combined with parking-zone configuration, the detections can be used to determine whether designated parking spaces are occupied.

### Key Features

* **Vehicle Detection:** Detects cars using a trained YOLO object detection model.
* **Parking-Zone Configuration:** Supports configured parking areas through `parking_zones.json`, if used by the application.
* **Automated Monitoring:** Uses computer vision to assist with parking-space occupancy monitoring.
* **Custom Model Support:** Includes trained YOLO model weights for inference.
* **Python-Based Implementation:** Provides scripts for detection, application execution and model training.

## Technologies Used

* Python
* YOLO object detection
* Ultralytics, if used by the selected model
* OpenCV, if used for image or video processing
* JSON for parking-zone configuration

## Project Structure

```text
senangpark-yolo/
├── app.py
├── detector.py
├── train.py
├── parking_zones.json
├── requirements.txt
├── .gitignore
├── README.md
└── yolov8n.pt
```

The model filename shown above is an example. Replace it with the actual model file used by the application. Include any additional required files or folders.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/senangpark-yolo.git
cd senangpark-yolo
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure that `requirements.txt` contains all the packages required by the application.

### 4. Prepare the Model

Place the required trained YOLO weights in the location expected by the application.

Update the model path in the configuration or detection script if necessary.

### 5. Run the Application

```bash
python app.py
```

The command above assumes that `app.py` is the application entry point. Follow any additional configuration or camera-input requirements defined in the source code.

## Model Information

* **Task:** Vehicle object detection
* **Model family:** YOLO
* **Target object:** Cars
* **Input:** Images or video, depending on the implemented application
* **Output:** Detected vehicle locations and associated detection information

The model weights, training dataset, training parameters and evaluation results should be documented when available.

## Parking-Slot Occupancy

Vehicle detection and parking-slot occupancy are related but different tasks.

The YOLO model identifies vehicles in the input. Parking occupancy is determined by relating vehicle detections to the configured parking zones and applying the occupancy logic implemented in the application.

The reliability of occupancy detection depends on factors such as camera angle, lighting, vehicle overlap, parking-zone configuration and detection performance.

## Dataset and Training

The `train.py` script is intended to document or perform model training.

If the dataset is available for redistribution, provide its source, license, class labels, train/validation split and relevant training configuration here.

If the dataset cannot be redistributed, provide instructions for obtaining it from its original source.

## Limitations

* Detection performance may vary with lighting, camera position and image quality.
* Occluded vehicles and overlapping detections may affect occupancy estimates.
* Parking-zone coordinates may need to be configured for each camera view.
* Model performance should be evaluated on representative parking-area footage before deployment.

## Future Improvements

* Parking availability counting and visualization.
* Real-time camera monitoring.
* Improved parking-zone configuration tools.
* Detection performance evaluation on different parking environments.
* Integration with a dashboard or IoT-based parking management system.

## License

Add a license that matches your project's intended use and your rights to distribute the code, model weights and dataset. Check the original model and dataset licenses before redistribution.

## Acknowledgements

This project builds on YOLO-based object detection. Credit the relevant model framework, pretrained weights and dataset providers according to their licensing and attribution requirements.

---

**SenangPark YOLO**
AI-powered vehicle detection for smarter parking management.
