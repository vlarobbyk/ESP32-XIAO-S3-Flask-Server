from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import time

app = Flask(__name__)

# ver que el directorio exista
if not os.path.exists("static/results"):
    os.makedirs("static/results")

def apply_morphology(image_path, kernel_size=37):
    """
    Carga una imagen, LA REDIMENSIONA, aplica 5 operaciones morfológicas,
    guarda los resultados y devuelve sus rutas.
    """
    # cargar en escala de grises
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        print(f"Error: No se pudo cargar la imagen de {image_path}")
        return []

    # redimensionar img
    target_width = 512
    scale = target_width / img_original.shape[1]
    target_height = int(img_original.shape[0] * scale)
    img = cv2.resize(img_original, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # crear kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # aplicar las operaciones morfologicas
    erosion = cv2.erode(img, kernel, iterations=1)
    dilation = cv2.dilate(img, kernel, iterations=1)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
    # 4. calcular (e)
    temp = cv2.subtract(tophat, blackhat)
    equation_e = cv2.add(img, temp)
    
    # 5. guardar y devolver rutas
    ts = int(time.time()) 
    
    results_data = [
        ("Original", img),
        ("Erosion", erosion),
        ("Dilation", dilation),
        ("Top Hat", tophat),
        ("Black Hat", blackhat),
        ("Equation (e)", equation_e)
    ]
    
    result_paths = []
    base_output_dir = "static/results"
    
    for (title, image) in results_data:
        filename_safe = title.lower().replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"results/{filename_safe}_{ts}.png"
        
        try:
            cv2.imwrite(f"static/{filename}", image)
            result_paths.append((title, filename)) 
        except Exception as e:
            print(f"Error al guardar {filename}: {e}")
            
    return result_paths

@app.route('/', methods=['GET', 'POST'])
def index():
    # files
    image_files = {
        "Neumonía (Bacteria) 1": "static/medical_images/person100_bacteria_480.jpeg",
        "Neumonía (Bacteria) 2": "static/medical_images/person103_bacteria_490.jpeg",
        "Neumonía (Virus)": "static/medical_images/person1015_virus_1702.jpeg",
        "Normal": "static/medical_images/IM-0010-0001.jpeg"
    }
    
    image_results = None

    if request.method == 'POST':
        selected_image_name = request.form.get('image_select')
        kernel_size = int(request.form.get('kernel_size', 37))
        
        # verificar kernel impar
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        image_path = image_files.get(selected_image_name)
        if image_path:
            image_results = apply_morphology(image_path, kernel_size)

    # renderizar la plantilla
    return render_template('index.html', 
                           image_names=image_files.keys(), 
                           results=image_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1818, debug=True)