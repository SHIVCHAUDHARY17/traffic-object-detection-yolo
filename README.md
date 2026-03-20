## Experimental Results

### Run 1: baseline_fast
- Model: `yolo11n.pt`
- Epochs: 3
- Image size: 512
- Batch size: 8
- Dataset fraction: 0.2

**Metrics**
- Precision: 0.522
- Recall: 0.152
- mAP50: 0.134
- mAP50-95: 0.0707

### Run 2: baseline_better (final model)
- Model: `yolo11n.pt`
- Epochs: 6
- Image size: 512
- Batch size: 8
- Dataset fraction: 0.35

**Metrics**
- Precision: 0.45467
- Recall: 0.19987
- mAP50: 0.19919
- mAP50-95: 0.10809

### Key Observations
- The improved run increased recall and both mAP metrics compared with the fast baseline.
- Car detection was the strongest among all classes.
- Traffic signs and traffic lights showed moderate detection performance.
- Rare classes such as rider, bicycle, motorcycle, and train remained challenging.
- Small and distant objects were harder to detect reliably.

### Qualitative Evaluation
Selected prediction examples were reviewed manually, including 5 strong examples and 5 failure cases.

**Main strengths**
- Strong performance on cars and other common traffic participants.
- Good detection quality in clear daylight scenes.
- Reasonable detection of small traffic signs and traffic lights in some scenes.
- Stable performance in several nighttime scenes, especially for reflective traffic signs.

**Main failure modes**
- False positives on visually similar roadside structures.
- Missed detections in some nighttime scenes.
- Weak performance on rare classes such as rider and motorcycle.
- Difficulty with parked vehicles and crowded static layouts.
- Occasional confusion between rider and car.

### Conclusion
This project established a complete traffic-object detection workflow, including dataset preparation, class-mapping validation, YOLO training, quantitative evaluation, inference, and qualitative failure analysis. The improved baseline demonstrated measurable gains over the initial fast run and provides a solid foundation for future work using longer training schedules, larger dataset fractions, or stronger hardware.