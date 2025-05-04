# VisionGuard
**AI-powered system for analyzing surveillance footage, detecting object movements, and summarizing security events using object detection, tracking, and visual-language models.**

---

##  Overview

VisionGuard automates the analysis of surveillance footage to detect unusual activities and provide context-aware summaries of the events. It is designed to help security officers interpret and understand the behavior of objects in real time from surveillance footage. The system uses:

YOLOv8 for object detection and Centroid Tracker for tracking moving objects across video frames.

LLaVA and BLIP to generate context-aware video summaries that describe the movement, interactions, and behavior of objects in the footage.

A security-focused feature that highlights suspicious or unusual behavior based on object movements and interactions, providing alerts and actionable insights.

Key Features:
Frame-by-frame analysis of surveillance footage with detailed object tracking and movement analysis.

Intelligent summaries that detect and describe unusual activities such as sudden changes in position, interactions between objects (e.g., vehicles and pedestrians), and other potentially suspicious patterns.

Real-time alerts for unusual events based on movement patterns and behavior inconsistencies.


-  **YOLOv8n** for real-time object detection  
-  **Centroid Tracker** to track object movements  
-  **LLaVA/BLIP** via Ollama for vision-language-based summarization  

The pipeline detects objects, tracks them across frames, and summarizes their actions in plain English.

---

##  Tech Stack

| Component        | Tool Used               |
|------------------|--------------------------|
| Object Detection | YOLOv8n (Ultralytics)    |
| Tracking         | Centroid Tracker (custom)|
| Summarization    | LLaVA / BLIP via Ollama  |
| Video Handling   | OpenCV                   |
| Backend Logic    | Python                   |

---
##  Run Order

Run the following Python scripts **in this order**:

1. **Download the YOLOv8n Model (only once)**
   ```bash
   from ultralytics import YOLO
   YOLO('yolov8n.pt')
2. **Download the YouTube Video** [ For Sample only ]
   ```bash
   python youtube_video.py
3. **Detect and Track Objects**
   ```bash
   python classify.py
4. **Generate Batch Summaries using LLaVA/BLIP**
   ```bash
   python llm_init.py

##  Built By

**Aarush Saxena**  
Student at **IIT Madras**, passionate about AI agents, reinforcement learning, and automation.

---

## Contact

- 📧 **Email**: aarushsaxena1615@gmail.com 
- 💼 **LinkedIn**: linkedin.com/in/aarush-saxena 


