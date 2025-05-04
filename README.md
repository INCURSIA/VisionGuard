# VisionTrackSummarizer  
**AI-powered system to summarize YouTube videos using object detection, tracking, and visual-language models.**

---

##  Overview

This project automates the process of generating summaries from YouTube videos. It uses:

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
2. **Download the YouTube Video**
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


