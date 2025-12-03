# 🏋️‍♂️ AI Squat Velocity Tracker: From Video to Metrics

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-MediaPipe-orange)
![Data Analysis](https://img.shields.io/badge/Signal_Processing-1€_Filter-yellow)
![Status](https://img.shields.io/badge/Project_Type-Portfolio-success)

> **"Turning unstructured video data into actionable biomechanical insights using Computer Vision and Signal Processing."**

## 💡 About The Project

As a Data Analyst, I constantly look for ways to quantify the world around me. This project was born from a specific need: **measuring athletic performance (fatigue and velocity) without expensive hardware.**

While traditional velocity-based training (VBT) requires sensors costing hundreds of euros, this project proves that a standard webcam and Python are enough to build a precise measurement tool.

It is not just a squat counter; it is an **end-to-end data pipeline** that ingests raw video, cleans the noisy signal using advanced mathematics, and exports structured datasets for analysis.

---

## 🔍 The Data Pipeline (ETL)

The core value of this project is treating video as a data source that needs to be extracted, transformed, and loaded (ETL).

### 1. Extraction (Input)
The system captures video frames using **OpenCV** and extracts skeletal landmarks (specifically the hip joint) using **Google MediaPipe**. This provides raw X,Y coordinates in real-time.

### 2. Transformation (Signal Processing)
Raw data from AI models is inherently noisy ("jitter"). To solve this, I implemented the **1€ Filter (One Euro Filter)**, an adaptive algorithm that trades off latency and jitter:
* **Static phase:** High filtering to stabilize the point.
* **Dynamic phase:** Low filtering to track high-speed movements with zero lag.

This step is crucial to distinguish between a "shaky camera" and actual movement.

### 3. Loading & Analysis (Output)
The system applies a **State Machine Logic** to identify phases of the movement (Eccentric vs. Concentric).
* **Data Structure:** It records the precise timestamp of the "deepest point" and the "lockout point".
* **Calculation:** It computes the average velocity of the concentric phase (the rise).
* **Export:** Data is saved to an **Excel (.xlsx)** file for further BI analysis.

---

## 📂 Project Structure & Evolution

This repository contains the evolution of the software, demonstrating the iterative problem-solving process:

* **`y_value.py` (The Prototype):**
    * An initial proof-of-concept focused solely on stabilization.
    * Uses a simple *Exponential Moving Average*.
    * *Result:* Good stability, but too much "lag" for fast movements.

* **`final_analysis.py` (The Product):**
    * The complete analytical tool.
    * Implements the **1€ Filter** for professional-grade precision.
    * Includes logic for rep counting, velocity calculation, and data export.
    * Generates visualization charts automatically using **Matplotlib**.

---

## 📊 Visual Outcomes

The script transforms pixels into a fatigue analysis curve. The chart below, generated automatically by the pipeline, shows the correlation between repetition count and velocity loss (indicating fatigue).

<img width="1000" height="600" alt="line_chart" src="https://github.com/user-attachments/assets/72719b00-e532-4f71-a5d9-561a1f562343" />

---

## 🛠️ Technologies Used

* **Language:** Python
* **Computer Vision:** MediaPipe, OpenCV
* **Data Manipulation:** Pandas
* **Visualization:** Matplotlib
* **Math/Physics:** Custom implementation of Adaptive Low-Pass Filtering (1€ Filter)

---

## 👤 Author

**Samuel Caballero**
*Data Analyst & BI Specialist*

Connecting the dots between physical performance and data science.
* [LinkedIn Profile](www.linkedin.com/in/samuelcablop)
