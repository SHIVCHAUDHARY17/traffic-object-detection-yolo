# Traffic Object Detection for Road-Scene Perception using YOLO

## Overview

This project builds a traffic-scene object detection pipeline using **Ultralytics YOLO11** on a **BDD100K-style road-scene dataset** exported from Roboflow in YOLO format.

The goal of the project was not only to train an object detector, but to build and understand the **full end-to-end workflow** of a practical computer vision detection project:

- setting up a reproducible local environment  
- preparing a YOLO-format dataset  
- validating class mappings and label structure  
- training baseline and improved models  
- evaluating performance using standard detection metrics  
- running inference on unseen validation images  
- reviewing qualitative strengths and failure cases  

This project was completed on a **Windows laptop with CPU-only training**, so the experiments were intentionally designed to balance runtime and practical project value.

The project also includes a simple Docker-based inference setup for reproducible execution.

---

## Project Objective

The main objective was to develop a **multi-class traffic object detection system** that can detect common road-scene objects such as:

- cars  
- buses  
- pedestrians  
- bicycles  
- trucks  
- motorcycles  
- trains  
- riders  
- traffic signs  
- traffic lights  

In addition to getting a model to run successfully, the project focused on understanding:

- how YOLO datasets are structured  
- how class IDs and class names are mapped  
- how pretrained object detectors are adapted to custom datasets  
- how to interpret detection metrics such as **Precision**, **Recall**, **mAP50**, and **mAP50-95**  
- how to analyze model strengths and weaknesses qualitatively  

---

## Why This Project Was Chosen

My prior profile was already stronger in **semantic segmentation**, especially through transformer-based segmentation work in autonomous driving conditions. This project was chosen to broaden my practical computer vision skill set by adding:

- **2D object detection**  
- practical road-scene model training  
- real validation and inference workflow  
- class-wise metric interpretation  
- qualitative failure analysis  

This makes the project highly relevant for junior-level roles in:

- computer vision  
- machine learning  
- automotive perception  
- robotics perception  

---

## Dataset

The dataset used in this project is a **BDD100K-style traffic-scene dataset** downloaded from **Roboflow Universe** and exported in **YOLO format**.

### Dataset Characteristics

- Total exported images: **9984**  
- Split used in project:  
  - Train images: **7987**  
  - Validation images: **1997**  

### Classes

The final class mapping used in the project was:

| ID | Class         |
|----|---------------|
| 0  | car           |
| 1  | bus           |
| 2  | pedestrian    |
| 3  | bicycle       |
| 4  | truck         |
| 5  | motorcycle    |
| 6  | train         |
| 7  | rider         |
| 8  | traffic-sign  |
| 9  | traffic-light |

### Important Dataset Issue and How It Was Solved

One practical problem encountered early in the project was that the exported `data.yaml` initially contained **numeric class names only**.  
This was an important learning point: a model can train successfully even when metadata is poorly defined, but a good project requires both correct training and correct interpretability.

---

## Tools and Libraries Used

### Core Stack

- Python 3.11  
- Ultralytics YOLO11  
- PyTorch  
- OpenCV  
- Matplotlib  
- PyYAML  

### Why These Were Used

**Ultralytics YOLO11**  
Used as the main object detection framework for:  
- loading pretrained weights  
- training a detection model  
- validating the trained detector  
- running inference on images  
- automatically saving plots and metrics  

**PyTorch**  
Used under the hood by Ultralytics for:  
- neural network execution  
- model optimization  
- loss computation  
- training on CPU  

**OpenCV**  
Used as part of the project environment for image handling and future inference scripting.

**Matplotlib**  
Used for visualizations and analysis outputs.

**PyYAML**  
Used for reading and writing the dataset configuration file (`data.yaml`).

---

## Project Structure

```
traffic-object-detection-yolo/
├── README.md
├── requirements.txt
├── requirements-docker.txt
├── data.yaml
├── .gitignore
├── .dockerignore
├── Dockerfile
├── data/
│   └── traffic_detection/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
├── scripts/
│   └── infer.py
├── results/
│   ├── experiment_notes.txt
│   ├── day1_summary.txt
│   ├── day2_checklist.txt
│   ├── qualitative_analysis.txt
│   ├── selected_good/
│   └── selected_failures/
└── runs/
    └── detect/
        └── results/
            ├── baseline_fast/
            ├── baseline_better/
            ├── infer_better/
            └── docker_infer/
```

---

## Environment Setup

The project was developed locally on Windows using a dedicated Python virtual environment.

### Why a Virtual Environment Was Used

A virtual environment keeps project dependencies isolated from the rest of the system. This is important because:

- package versions do not conflict with other projects  
- the setup is cleaner and more reproducible  
- project dependencies can be captured in `requirements.txt`  

### Setup Steps

1. Create project folder structure  
2. Create and activate `.venv`  
3. Install required packages  
4. Verify YOLO installation with `yolo checks`  

---

## Dataset Preparation Workflow

The dataset preparation process was an important part of the project and included:

### 1. Downloading and Exporting the Dataset

The dataset was downloaded from Roboflow in YOLO-compatible format.

### 2. Organizing the Dataset Locally

The dataset was copied into the project-specific folder structure:

```
data/traffic_detection/
├── images/train
├── images/val
├── labels/train
└── labels/val
```

### 3. Verifying Consistency

The number of image files and label files was checked carefully:

- train images = 7987, train labels = 7987  
- val images = 1997, val labels = 1997  

This ensured that every image had a corresponding label file.

### 4. Writing the Final `data.yaml`

The project-level dataset config file was written manually after validating the class mapping.

**Final `data.yaml`:**

```yaml
path: data/traffic_detection
train: images/train
val: images/val

names:
  0: car
  1: bus
  2: pedestrian
  3: bicycle
  4: truck
  5: motorcycle
  6: train
  7: rider
  8: traffic-sign
  9: traffic-light
```

---

## What YOLO Training Means in This Project

YOLO training in this project means:

- loading a pretrained detection model  
- replacing its original detection head with one that matches the custom dataset classes  
- learning to predict bounding boxes and class labels for the 10 target traffic-scene classes  

The pretrained model used was:  
`yolo11n.pt` (the nano version of YOLO11 – lightweight, fast, suitable for CPU-only experimentation).

### What the Main Training Parameters Mean

- **epochs** – number of full passes through the training data.  
  Example: `epochs=3` means the model sees the full selected dataset 3 times.

- **imgsz** – training image size.  
  Example: `imgsz=512` means images are resized to 512 for training and inference.  
  *Smaller image size = faster training, less computation, but may reduce small-object detection performance.*

- **batch** – number of images processed in one training step.  
  Example: `batch=8`

- **fraction** – fraction of the dataset used during training.  
  Example: `fraction=0.2` means use 20% of the training set.  
  *Very important for CPU-based experimentation because full-data training was too slow.*

- **project** and **name** – control where the run outputs are saved.  
  Example: `project=results name=baseline_fast`

---

## Training Strategy

The project was trained in two main completed runs.

### Why Two Runs Were Used

The original full-data attempt was too slow on CPU, so the practical strategy became:

1. Run a fast baseline to validate the pipeline end-to-end  
2. Run a stronger improved baseline with more training budget  

This reflects a realistic engineering workflow: first validate correctness, then improve quality.

### Training Commands

**Fast Baseline**

```bash
yolo detect train data=.\data.yaml model=yolo11n.pt epochs=3 imgsz=512 batch=8 fraction=0.2 project=results name=baseline_fast
```

**Improved Baseline**

```bash
yolo detect train data=.\data.yaml model=yolo11n.pt epochs=6 imgsz=512 batch=8 fraction=0.35 project=results name=baseline_better
```

### Inference Using the Final Model

```bash
yolo detect predict model=".\runs\detect\results\baseline_better\weights\best.pt" source=".\data\traffic_detection\images\val" conf=0.25 imgsz=512 save=True project=results name=infer_better
```

---

## Experimental Results

### Run 1: `baseline_fast`
- Model: `yolo11n.pt`  
- Epochs: 3  
- Image size: 512  
- Batch size: 8  
- Dataset fraction: 0.2  

| Metric      | Value   |
|-------------|---------|
| Precision   | 0.522   |
| Recall      | 0.152   |
| mAP50       | 0.134   |
| mAP50-95    | 0.0707  |

### Run 2: `baseline_better` (Final Model)
- Model: `yolo11n.pt`  
- Epochs: 6  
- Image size: 512  
- Batch size: 8  
- Dataset fraction: 0.35  

| Metric      | Value     |
|-------------|-----------|
| Precision   | 0.45467   |
| Recall      | 0.19987   |
| mAP50       | 0.19919   |
| mAP50-95    | 0.10809   |

---

## Understanding the Evaluation Metrics

- **Precision**: Out of all predicted objects, how many were correct? Higher precision means fewer false positives.  
- **Recall**: Out of all real ground-truth objects, how many were found by the model? Higher recall means fewer missed detections.  
- **mAP50**: Mean Average Precision at IoU 0.50 – detection quality with a moderate overlap requirement.  
- **mAP50-95**: A stricter and more comprehensive metric, averaged across multiple IoU thresholds. Harder to score well on and gives a better sense of true detector quality.

### Interpretation of This Project’s Results

The improved run increased recall, mAP50, and mAP50-95. This means the stronger run was better at finding objects overall and gave better detection quality, even though precision dropped slightly. That slight precision drop likely means the model found more objects but also made more false positives – a common trade-off in detection systems.

---

## Key Observations

- The improved run increased recall and both mAP metrics compared with the fast baseline.  
- Car detection was the strongest among all classes.  
- Traffic signs and traffic lights showed moderate detection performance.  
- Rare classes such as rider, bicycle, motorcycle, and train remained challenging.  
- Small and distant objects were harder to detect reliably.  

### Why Car Detection Was Strongest

Cars had:  
- many more training examples  
- more visual consistency  
- larger representation in urban scenes  

This made them easier for the model to learn than rare classes.

### Why Some Classes Remained Weak

Classes such as rider, motorcycle, bicycle, and train likely remained weak because of:  
- lower class frequency  
- smaller object sizes  
- more visual variability  
- limited training budget  

---

## Qualitative Evaluation

Selected prediction examples were reviewed manually, including 5 strong examples and 5 failure cases.

### Main Strengths

- Strong performance on cars and other common traffic participants.  
- Good detection quality in clear daylight scenes.  
- Reasonable detection of small traffic signs and traffic lights in some scenes.  
- Stable performance in several nighttime scenes, especially for reflective traffic signs.

### Main Failure Modes

- False positives on visually similar roadside structures.  
- Missed detections in some nighttime scenes.  
- Weak performance on rare classes such as rider and motorcycle.  
- Difficulty with parked vehicles and crowded static layouts.  
- Occasional confusion between rider and car.

### Examples From Qualitative Analysis

**Good Examples**  
- The model correctly detected all three cars and one pedestrian in a clear daylight scene, with well-aligned bounding boxes.  
- The model detected all visible cars and the truck correctly, and it also successfully detected a small traffic sign.  
- The model correctly detected a pedestrian, traffic sign, truck, and car in the same scene, showing good multi-class detection performance.  
- Most relevant objects were detected correctly, including small traffic lights and traffic signs. There were a few false positives where normal posters or sign-like objects were also detected as traffic signs.  
- The model showed good nighttime performance similar to daytime scenes. Traffic signs were detected especially well at night, likely because reflective surfaces made them more visually prominent.

**Failure Examples**  
- The model made a major false positive by detecting a side wall or roadside structure as a large bus.  
- In a nighttime scene, the model missed a car and also failed to detect a large truck.  
- Small and rare classes such as rider and motorcycle were missed, and traffic signs were also not detected even in a clear daylight scene.  
- In a parking-area scene, the model failed to detect parked cars, showing weak performance in static dense parking layouts.  
- A rider was misclassified as a car, and pedestrians in the scene were missed.

---

## Inference Performance

Inference was run on the full validation image set using the final model:

- Model: `baseline_better/weights/best.pt`  
- Image size: 512  
- Confidence threshold: 0.25  

Approximate per-image speed on CPU:

- Preprocess: 1.7 ms  
- Inference: 43.1 ms  
- Postprocess: 0.7 ms  

This shows that the model can produce predictions reasonably quickly even on CPU, although training remains much slower.

---

## What I Learned From This Project

This project was valuable not only because it produced a trained model, but because it improved my practical understanding of the complete object detection pipeline.

### Technical Understanding Gained

- how YOLO-format datasets are organized  
- how image and label pairs must align exactly  
- how class IDs and semantic class names are mapped  
- how pretrained detection models are adapted to new class counts  
- how to use dataset fractions and reduced image sizes for faster experimentation  
- how to interpret precision, recall, and mAP  
- how to compare runs meaningfully rather than relying on a single number  
- how to complement quantitative metrics with qualitative failure analysis  

### Practical Engineering Understanding Gained

- building a project in a structured and reproducible way matters  
- metadata quality is as important as raw data availability  
- fast baseline experiments are useful before expensive full runs  
- CPU-only training requires pragmatic compromises  
- a good project should explain not only results, but also assumptions, limitations, and next steps  

---

## Limitations

This project has several limitations:

- Training was done on CPU only, which restricted the overall training budget.  
- The final model was trained on a fraction of the dataset, not the full training set.  
- Rare classes remained under-learned.  
- Small object detection is still weak in difficult scenes.  
- The model is a practical baseline, not a highly optimized final detector.

These limitations are important because they explain why the project should be understood as a strong practical baseline rather than a production-level system.

---

## Future Improvements

Possible next steps include:

- train for more epochs using stronger hardware  
- use a larger dataset fraction or the full training set  
- compare YOLO11n with a larger model variant  
- perform class-wise balancing or targeted data filtering  
- benchmark performance under challenging conditions such as blur, low light, and clutter  
- integrate the detector into a ROS 2 perception node  
- combine detection with tracking for a multi-object tracking pipeline  

---

## Docker

This project also includes a simple Docker-based inference setup for reproducible execution.

### Why Docker Was Added

Docker was added to package the inference environment in a reproducible way. Instead of manually recreating the Python setup on another machine, the project can be run inside a container with the required dependencies already defined.

### Build the Image

```bash
docker build -t traffic-object-detection-yolo .
```

### Run Inference in the Container

```bash
docker run --rm -v "${PWD}:/app" traffic-object-detection-yolo
```

### What This Docker Setup Does

- builds a reproducible Python environment inside a container  
- installs the required dependencies  
- loads the trained YOLO model  
- runs inference on selected example images  
- saves outputs back into the local project folder through a mounted volume  

### Docker-Related Files

- `Dockerfile`  
- `.dockerignore`  
- `requirements-docker.txt`  

*This Docker setup was designed mainly for inference reproducibility, not for long CPU-based training.*

---

## Conclusion

This project established a complete traffic-object detection workflow, including:

- dataset preparation  
- class-mapping validation  
- YOLO training  
- quantitative evaluation  
- inference  
- qualitative failure analysis  

The improved baseline demonstrated measurable gains over the initial fast run and provides a solid foundation for future work using longer training schedules, larger dataset fractions, or stronger hardware.

Overall, the project successfully broadened my practical computer vision profile beyond semantic segmentation by adding hands-on experience in object detection, training workflow design, evaluation, failure analysis, and basic Docker-based reproducibility.

---

## Files of Interest

- **Final Model**: `runs/detect/results/baseline_better/weights/best.pt`  
- **Comparison Run**: `runs/detect/results/baseline_fast/`  
- **Inference Outputs**:  
  - `runs/detect/results/infer_better/`  
  - `runs/detect/results/docker_infer/`  
- **Docker Files**:  
  - `Dockerfile`  
  - `.dockerignore`  
  - `requirements-docker.txt`  
- **Notes**: `results/experiment_notes.txt`
