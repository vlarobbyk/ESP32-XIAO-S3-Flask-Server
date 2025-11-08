# **🌐 ESP32-XIAO-S3 Flask Video Processing Server**

This repository contains a simple Python Flask server for streaming video from an ESP32-CAM (or ESP32-XIAO-S3) module, applying real-time noise simulation (Salt Noise), and performing image filtering using **PyTorch's 2D Convolution** (F.conv2d).

The project demonstrates the critical steps for successful interoperability between OpenCV (image capture/display) and PyTorch (tensor-based filtering), ensuring proper data types and value ranges are maintained.

## **✨ Features**

* **Real-time Video Streaming** via Flask (MJPEG format).  
* **Frame Processing Pipeline** combining OpenCV, NumPy, and PyTorch.  
* **Salt Noise Simulation** added to the grayscale video frames.  
* **Linear Filtering** applied using a **Normalized Mean Filter** (implemented with F.conv2d and a $3\\times3$ kernel).  
* Correct handling of float32 convolution output to prevent image saturation/artifacts.

## **🖼️ Application Screenshot**

The application displays three panels: **Original Grayscale**, **Noisy Image**, and **Filtered Output**.

![A screenshot of the web-server running. The left-side shows the original image, the center the noisy image (salt noise) and the right-side image shows the result of applying a kernel of 3x3 in PyTorch.](./static/Screenshot_20251108_171053.png)

## **🚀 Setup Guide**

### **Prerequisites**

1. **ESP32-CAM/XIAO-S3:** The module must be running firmware that serves an MJPEG stream (e.g., using the CameraWebServer example from the Arduino ESP32 library).  
2. **IP Address:** The ESP32's IP address must be known and updated in the app.py file.

### **Installation**

1. **Clone the Repository:**  
   git clone [https://github.com/vlarobbyk/ESP32-XIAO-S3-Flask-Server.git](https://github.com/vlarobbyk/ESP32-XIAO-S3-Flask-Server.git)

2. Install Dependencies:  
   This project requires Flask, OpenCV, NumPy, requests, and PyTorch. It is highly recommended to use a virtual environment.  
   pip install Flask opencv-python numpy requests torch

### **Configuration**

Open app.py and update the stream details:

\# app.py  
\_URL \= 'http://\[YOUR\_ESP32\_IP\_ADDRESS\]' \# e.g., '[http://192.168.1.100](http://192.168.1.100)'
\_PORT \= '81'                           \# Default, change if necessary

### **Running the Server**

Execute the Python script:

python app.py

The application will be accessible at http://127.0.0.1:5000/.

## **⚙️ Key Filtering Details (PyTorch & OpenCV Interop)**

The core logic for stable image filtering lies in these data handling steps:

1. **Kernel Normalization:** The $3\\times3$ kernel (torch.ones(3, 3)) is divided by $9.0$ before being used in F.conv2d. This prevents the convolution sum from exceeding the original pixel range (0-255).  
2. **Type Conversion:** The floating-point output from PyTorch's convolution is converted back to an 8-bit integer array suitable for display using the robust OpenCV function: cv2.convertScaleAbs(). This function correctly handles the necessary clipping and type casting.
