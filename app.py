
# Author: vlarobbyk
# Version: 1.0
# Date: 2024-10-20
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.
# Updated: 2025-08-11

# Forked by: Markeeto
# Date of Fork: 2025-08-12

from flask import Flask, render_template, Response, stream_with_context, Request
from io import BytesIO

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F

app = Flask(__name__)

# IP Address
_URL = 'http://10.0.0.11'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'

stream_url = ''.join([_URL,SEP,_PORT,_ST])

# Method to capture the video generated from the ESP32-XIAO-S3 mini cam
def video_capture():
    res = requests.get(stream_url, stream=True)

    median_mask = torch.ones(3, 3, dtype = torch.float32)/9.0

    kernel = median_mask.unsqueeze(0).unsqueeze(0)
    convolved_output = None
    img_output = None

    for chunk in res.iter_content(chunk_size=100000):

        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                N = 537
                height, width = gray.shape

                noise = np.full((height, width), 0, dtype=np.uint8)
                random_positions = (np.random.randint(0, height, N), np.random.randint(0, width, N))
                
                noise[random_positions[0], random_positions[1]] = 255

                noise_image = cv2.bitwise_or(gray, noise)

                img_tensor = torch.from_numpy(noise_image.astype(np.float32)).unsqueeze(0).unsqueeze(0)

                convolved_output = F.conv2d(
                    input = img_tensor,
                    weight = kernel,
                    padding = 1,
                    bias = None
                )

                img_output = convolved_output.cpu().numpy().squeeze()
                img_output = cv2.convertScaleAbs(img_output)

                total_image = np.zeros((height, width * 3), dtype=np.uint8)
                total_image[:, :width] = gray
                total_image[:, width:width*2] = noise_image
                total_image[:, width*2:] = img_output

                (flag, encodedImage) = cv2.imencode(".jpg", total_image)
                if not flag:
                    continue

                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')

            except Exception as e:
                print(e)
                continue

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_stream")
def video_stream():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False)

