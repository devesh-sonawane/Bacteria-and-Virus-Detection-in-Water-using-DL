
```markdown
# Clean Water AI: Bacteria and Virus Detection in Water Using Deep Learning

## Overview

This project leverages deep learning for the detection of harmful microorganisms such as bacteria and viruses in water. Using object detection models like SSD MobileNet, the system is trained to recognize microscopic pathogens from image data, enabling fast and automated water quality assessment.

## Features

- Real-time object detection using webcam input
- Trained on labeled water sample images
- Converts XML annotations to CSV and TFRecord for TensorFlow training
- Inference-ready with saved frozen inference graph
- Lightweight and efficient model using SSD MobileNet v3

## Directory Structure

```
├── Object_detection_webcam.py     # Run detection using webcam
├── clean_water_ai.py              # Main script for detection
├── generate_tfrecord.py           # Converts CSV to TFRecord format
├── xml_to_csv.py                  # Converts XML annotations to CSV
├── training/                      # Contains training configs and checkpoints
├── inference_graph/               # Frozen inference graph for deployment
├── ssd_mobilenet_v3_large_coco/  # Pretrained model base
├── test.record                    # TFRecord for evaluation
├── images/                        # Annotated training and test images
└── stream/                        # Sample image stream data
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/devesh-sonawane/Bacteria-and-Virus-Detection-in-Water-using-DL.git
cd Bacteria-and-Virus-Detection-in-Water-using-DL
```

### 2. Install Dependencies

Ensure Python 3.7+ and TensorFlow 2.x are installed.

```bash
pip install -r requirements.txt  # if available
```

You may need to install manually:
```bash
pip install opencv-python pandas lxml tensorflow
```

### 3. Prepare Dataset

- Place your labeled image dataset in the `images/` directory.
- Ensure annotations are in Pascal VOC XML format.

Convert them to CSV:

```bash
python xml_to_csv.py
```

Then convert CSVs to TFRecord:

```bash
python generate_tfrecord.py
```

### 4. Train the Model

Use TensorFlow Object Detection API with the configuration in the `training/` folder. Make sure `pipeline.config` is updated with correct paths.

### 5. Run Inference

For webcam-based detection:

```bash
python Object_detection_webcam.py
```

Or use the main inference script:

```bash
python clean_water_ai.py
```

## Notes

- Uses SSD MobileNet V3 pretrained on COCO for feature extraction.
- The frozen inference graph is available under `inference_graph/` for quick testing..
```
