from flask import Flask, Response, render_template, request
import requests
import cv2
import numpy as np
import time
import threading
import torch
import torch.nn.functional as F

app = Flask(__name__)

# variables globales
frame_lock = threading.Lock()
column_jpeg_buffers = {
    "col_A": None,
    "col_B": None
}
noise_params = {'mean': 0.0, 'std': 10.0, 'var': 0.01}

# inicializadores y funciones de ayuda
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def put_title_and_fps(img, text, fps):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    color = (0, 255, 255) 
    fps_color = (0, 255, 0)
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = w - text_width - 10
    y = 25
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(img, fps_text, (10, 25), font, font_scale, fps_color, thickness+1, cv2.LINE_AA)
    return img

def add_gaussian_noise(image, mean=0, std=10):
    gauss = np.random.normal(mean, std, image.shape).astype('float32')
    noisy = cv2.add(image.astype('float32'), gauss)
    noisy = np.clip(noisy, 0, 255).astype('uint8')
    return noisy

def add_speckle_noise(image, var=0.01):
    row, col, ch = image.shape
    sigma = var**0.5
    gauss = np.random.normal(0, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + image * gauss
    return np.clip(noisy, 0, 255).astype('uint8')

# funciones de filtrado con PyTorch
gaussian_kernel_3x3 = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
gaussian_kernel_3x3 = gaussian_kernel_3x3 / torch.sum(gaussian_kernel_3x3)
gaussian_kernel_3x3 = gaussian_kernel_3x3.view(1, 1, 3, 3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gaussian_kernel_3x3 = gaussian_kernel_3x3.to(device)
print(f"PyTorch usará: {device}")

def apply_pytorch_filter(image_np, kernel_tensor):
    img_tensor = torch.from_numpy(image_np.astype('float32')).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.to(device)
    img_tensor = img_tensor.unsqueeze(0)
    kernel = kernel_tensor.repeat(3, 1, 1, 1)
    filtered_tensor = F.conv2d(img_tensor, kernel, groups=3, padding='same')
    filtered_tensor = filtered_tensor.squeeze(0)
    filtered_np = filtered_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
    return np.clip(filtered_np, 0, 255).astype('uint8')

# hilo de captura y procesamiento
def capture_and_process_thread():
    global column_jpeg_buffers, noise_params, frame_lock
    
    url = 'http://192.168.10.188/video_stream'
    prev_time = time.time()
    buffer = b''
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # individuales
    target_size = (320, 240) 
    black = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    error_img_row = np.hstack([black.copy(), black.copy(), black.copy()])
    cv2.putText(error_img_row, "Error en esta Fila", (380, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def resize(im):
        if im is None or im.size == 0:
            return black.copy()
        resized_im = cv2.resize(im, target_size, interpolation=cv2.INTER_AREA)
        if len(resized_im.shape) == 2:
            resized_im = cv2.cvtColor(resized_im, cv2.COLOR_GRAY2BGR)
        return resized_im

    print("Iniciando hilo de captura...")
    
    try:
        with requests.get(url, stream=True) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    buffer += chunk
                    start = buffer.find(b'\xff\xd8')
                    end = buffer.find(b'\xff\xd9')
                    if start != -1 and end != -1 and end > start:
                        jpg = buffer[start:end+2]
                        buffer = buffer[end+2:]
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        
                        if img is None: continue

                        curr_time = time.time()
                        fps = 1.0 / (curr_time - prev_time)
                        prev_time = curr_time

                        with frame_lock:
                            current_mean = noise_params['mean']
                            current_std = noise_params['std']
                            current_var = noise_params['var']

                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # precallculos para evitar fallos
                        try: noisy_gauss = add_gaussian_noise(img, current_mean, current_std)
                        except: noisy_gauss = img.copy()
                        try: gamma_img = adjust_gamma(img, gamma=1.5)
                        except: gamma_img = img.copy()
                        try: noisy_speckle = add_speckle_noise(gamma_img, current_var)
                        except: noisy_speckle = img.copy()
                        try: restored_gauss_cv = cv2.GaussianBlur(noisy_gauss, (5, 5), 0)
                        except: restored_gauss_cv = noisy_gauss.copy()

                        # calculo de filas independientes
                        try:
                            hist_eq = cv2.equalizeHist(gray); clahe_img = clahe.apply(gray)
                            row1 = np.hstack([ put_title_and_fps(resize(img.copy()), "Original", fps), put_title_and_fps(resize(hist_eq), "Histograma", fps), put_title_and_fps(resize(clahe_img), "CLAHE", fps) ])
                        except Exception as e: print(f"Error Fila 1: {e}"); row1 = error_img_row

                        try:
                            fgMask = backSub.apply(img); fg_only = cv2.bitwise_and(img, img, mask=fgMask)
                            row2 = np.hstack([ put_title_and_fps(resize(gamma_img), "Gamma", fps), put_title_and_fps(resize(fgMask), "MOG2", fps), put_title_and_fps(resize(fg_only), "Primer plano", fps) ])
                        except Exception as e: print(f"Error Fila 2: {e}"); row2 = error_img_row

                        try:
                            fg_and = cv2.bitwise_and(img, img, mask=fgMask); fg_or = cv2.bitwise_or(img, img, mask=fgMask); fg_xor = cv2.bitwise_xor(img, img, mask=fgMask)
                            row3 = np.hstack([ put_title_and_fps(resize(fg_and), "AND", fps), put_title_and_fps(resize(fg_or), "OR", fps), put_title_and_fps(resize(fg_xor), "XOR", fps) ])
                        except Exception as e: print(f"Error Fila 3: {e}"); row3 = error_img_row
                        
                        try:
                            row4 = np.hstack([ put_title_and_fps(resize(noisy_gauss), "Ruido Gaussiano", fps), put_title_and_fps(resize(noisy_speckle), "Ruido Speckle", fps), put_title_and_fps(resize(black.copy()), "(espacio)", fps) ])
                        except Exception as e: print(f"Error Fila 4: {e}"); row4 = error_img_row

                        try:
                            restored_speckle_cv = cv2.medianBlur(noisy_speckle, 5); restored_blur_cv = cv2.blur(noisy_gauss, (5, 5))
                            row5 = np.hstack([ put_title_and_fps(resize(restored_gauss_cv), "Filtro Gaussiano", fps), put_title_and_fps(resize(restored_speckle_cv), "Filtro Mediana", fps), put_title_and_fps(resize(restored_blur_cv), "Filtro Blur", fps) ])
                        except Exception as e: print(f"Error Fila 5: {e}"); row5 = error_img_row

                        try:
                            restored_pytorch = apply_pytorch_filter(noisy_gauss, gaussian_kernel_3x3); restored_cv2_gauss_3x3 = cv2.GaussianBlur(noisy_gauss, (3, 3), 0)
                            row6 = np.hstack([ put_title_and_fps(resize(noisy_gauss), "Ruido Gaussiano", fps), put_title_and_fps(resize(restored_pytorch), "Filtro PyTorch 3x3", fps), put_title_and_fps(resize(restored_cv2_gauss_3x3), "Filtro OpenCV 3x3", fps) ])
                        except Exception as e: print(f"Error Fila 6: {e}"); row6 = error_img_row
                        
                        try:
                            canny_edges = cv2.Canny(gray, 100, 200)
                            noisy_gauss_gray = cv2.cvtColor(noisy_gauss, cv2.COLOR_BGR2GRAY)
                            sobelx_noisy = cv2.Sobel(noisy_gauss_gray, cv2.CV_64F, 1, 0, ksize=3)
                            sobely_noisy = cv2.Sobel(noisy_gauss_gray, cv2.CV_64F, 0, 1, ksize=3)
                            sobel_mag_noisy = cv2.magnitude(sobelx_noisy, sobely_noisy)
                            sobel_mag_noisy = cv2.convertScaleAbs(sobel_mag_noisy)
                            
                            restored_gauss_gray = cv2.cvtColor(restored_gauss_cv, cv2.COLOR_BGR2GRAY)
                            sobelx_clean = cv2.Sobel(restored_gauss_gray, cv2.CV_64F, 1, 0, ksize=3)
                            sobely_clean = cv2.Sobel(restored_gauss_gray, cv2.CV_64F, 0, 1, ksize=3)
                            sobel_mag_clean = cv2.magnitude(sobelx_clean, sobely_clean)
                            sobel_mag_clean = cv2.convertScaleAbs(sobel_mag_clean)
                            
                            row7 = np.hstack([ put_title_and_fps(resize(canny_edges), "Bordes Canny", fps), put_title_and_fps(resize(sobel_mag_noisy), "Sobel (Con Ruido)", fps), put_title_and_fps(resize(sobel_mag_clean), "Sobel (Sin Ruido)", fps) ])
                        except Exception as e:
                            print(f"ERROR EN FILA 7: {e}")
                            row7 = error_img_row
                        
                        # apilar las dos columnas
                        col_A_img = np.vstack((row1, row2, row3))
                        col_B_img = np.vstack((row4, row5, row6, row7))
                        
                        # guardar los dos buffers
                        with frame_lock:
                            column_jpeg_buffers["col_A"] = cv2.imencode('.jpg', col_A_img)[1].tobytes()
                            column_jpeg_buffers["col_B"] = cv2.imencode('.jpg', col_B_img)[1].tobytes()

                        time.sleep(0.01) 

    except Exception as e:
        print(f"Error CRITICO en el hilo de captura: {e}")

# generadores de frames para cada columna
def column_frame_generator(col_key):
    global column_jpeg_buffers, frame_lock
    while True:
        with frame_lock:
            jpeg_bytes = column_jpeg_buffers.get(col_key)
        
        if jpeg_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
        time.sleep(1/30)

# rutas
@app.route('/')
def index():
    global noise_params, frame_lock
    with frame_lock:
        noise_params['mean'] = float(request.args.get('mean', noise_params['mean']))
        noise_params['std'] = float(request.args.get('std', noise_params['std']))
        noise_params['var'] = float(request.args.get('var', noise_params['var']))
    
    return render_template('index.html', 
                           mean=noise_params['mean'], 
                           std=noise_params['std'], 
                           var=noise_params['var'])

@app.route('/video_col_A')
def video_col_A():
    return Response(column_frame_generator("col_A"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_col_B')
def video_col_B():
    return Response(column_frame_generator("col_B"), mimetype='multipart/x-mixed-replace; boundary=frame')

# iniciar servidor e hilo de captura
if __name__ == '__main__':
    capture_thread = threading.Thread(target=capture_and_process_thread, daemon=True)
    capture_thread.start()
    app.run(host='0.0.0.0', port=1818, debug=False, threaded=True)