# Seat Detector

This directory contains resources for a YOLOv8-based model designed to detect seats in images or video streams.

## Directory Contents

-   `yolov8n.pt`: This is a pre-trained YOLOv8 nano model checkpoint, specifically fine-tuned or trained for seat detection.
-   `seat_detector_dataset_yolo/`: This directory contains the dataset used for training the seat detection model. It typically follows the YOLO format with `images` (for raw images) and `labels` (for annotation files) subdirectories, along with a `data.yaml` configuration file.
-   `runs/`: This directory stores the results of model training, validation, and detection operations. It usually contains subdirectories for `detect` (inference results) and `train` (training logs, metrics, and best models).

## Usage

To use the `yolov8n.pt` model for inference, you will typically use the `ultralytics` Python package.

### Installation

First, ensure you have the `ultralytics` package installed:

```bash
pip install ultralytics
```

### Detection

You can run detection on an image or video using the following command structure:

```bash
yolo detect predict model=yolov8n.pt source='path/to/your/image.jpg' # or 'path/to/your/video.mp4'
```

Replace `'path/to/your/image.jpg'` or `'path/to/your/video.mp4'` with the actual path to your input media.

### Training (Optional)

If you wish to re-train or fine-tune the model with additional data, you would typically use a command similar to this, assuming your dataset is properly formatted in the `seat_detector_dataset_yolo/` directory:

```bash
yolo detect train model=yolov8n.pt data=seat_detector_dataset_yolo/data.yaml epochs=100 imgsz=640
```

Adjust `epochs` and `imgsz` (image size) as needed.

## Dataset

The `seat_detector_dataset_yolo/` directory contains the dataset organized in a YOLO-compatible format.
- `images/`: Contains the actual image files.
- `labels/`: Contains the corresponding `.txt` annotation files for each image, specifying object classes and bounding box coordinates.
- `data.yaml`: A configuration file that defines the paths to the training, validation, and test images, as well as the class names.
