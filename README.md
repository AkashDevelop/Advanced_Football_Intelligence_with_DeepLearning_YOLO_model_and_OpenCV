# ⚽ AI-Powered Football Tracking System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-red?style=flat-square)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-green?style=flat-square&logo=scikitlearn)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-orange?style=flat-square&logo=opencv)

An **advanced football tracking** system leveraging cutting-edge **computer vision and machine learning techniques**. This project provides actionable insights into player performance and match dynamics through automated detection, tracking, and analysis.


![Football Tracking](https://github.com/user-attachments/assets/8b3e06aa-6c3a-4386-a16c-40f005e39e34)


---
## ✨ Highlights

---

**Feature**                   | **Details**
------------------------------ | ------------------------------------------------------------------------
**Real-Time Detection**        | Achieves high-accuracy player and ball detection with YOLOv8.
**Custom Training**            | Sports-specific YOLOv8 model fine-tuned for precise performance.
**Performance Metrics**        | Automatically computes player speed and distance during gameplay.
**Perspective Transformation** | Converts pixel-level data into real-world measurements for analytics.
**Dynamic Analytics**          | Offers insights into match strategies and player efficiency.

---


## 🔧 Tools and Technologies

- ![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
- ![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-red?style=flat-square)
- ![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange?style=flat-square&logo=opencv)
- ![NumPy](https://img.shields.io/badge/NumPy-Array%20Processing-lightblue?style=flat-square&logo=numpy)
- ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black?style=flat-square&logo=pandas)
- ![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-green?style=flat-square&logo=scikitlearn)

---

## 🔧 Advanced Setup

### Configure Perspective Transformation
To ensure accurate real-world measurements:
1. Define four points on the playing field in the video frame.
2. Map these points to their real-world coordinates in meters.
3. Use OpenCV's `cv2.getPerspectiveTransform` and `cv2.warpPerspective` functions to apply the transformation.


Here’s a more comprehensive and uniquely styled README template for your project, with additional sections to make it stand out. It includes features like "Usage Scenarios," "Challenges Faced," "Learnings," and stylized code examples.

---

### 🔎 **Usage Scenarios**
1. **Coaching**: Analyze player performance and team strategy.
2. **Broadcasting**: Provide real-time player stats during live matches.
3. **Scouting**: Evaluate player potential based on performance metrics.

---

## 💡 Challenges Faced
- **Data Labeling**: Preparing and labeling the custom dataset was time-intensive.
- **Perspective Transformation**: Ensuring accurate real-world conversions required precise calibration.
- **Tracking Movements**: Handling occlusion and camera panning added complexity to tracking.

---

## 📘 Learnings
- Mastered YOLOv8 fine-tuning and implementation for object detection.
- Improved understanding of perspective transformation and real-world metric calculations.
- Enhanced Python scripting skills for end-to-end pipeline integration.

---


### **What’s New?**
1. **Usage Scenarios**: Highlighted practical applications of the project.
2. **Challenges and Learnings**: Demonstrated the technical journey and solutions.
3. **Advanced Setup**: Added perspective transformation example code for clarity.
4. **Sample Metrics**: Included tabular data for better representation.
5. **Screenshots Section**: Highlighted visual output to engage users. 

## 📊 Sample Code Snippet

### Object Detection with YOLOv8
```python
from ultralytics import YOLO
model = YOLO("models/yolov8-weights.pt")
results = model.predict(source="videos/match.mp4", conf=0.5)
for result in results:
    print(result.boxes.xyxy, result.names, result.confidence)






