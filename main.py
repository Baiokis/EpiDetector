import cv2
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ma4uwwnTIDfJ567zH2gL"
)

def cv2_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def desenhar_bounding_boxes(image, result):
    draw = ImageDraw.Draw(image)
    for prediction in result['predictions']:
        x = prediction['x']
        y = prediction['y']
        width = prediction['width']
        height = prediction['height']
        
        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2

        draw.rectangle([left, top, right, bottom], outline="red", width=3)
    return image

def mostrar_imagem(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.draw()
    plt.pause(0.001)

def tratar_resultado(result):
    print(f"Inference ID: {result['inference_id']}")
    print(f"Tempo de inferência: {result['time']:.3f} segundos")
    print(f"Tamanho da imagem: {result['image']['width']}x{result['image']['height']} pixels")
    
    print("\nPredições detectadas:")
    for i, prediction in enumerate(result['predictions']):
        print(f"  Predição {i+1}:")
        print(f"    Classe: {prediction['class']}")
        print(f"    Confiança: {prediction['confidence']:.2f}")
        print(f"    Posição: (x: {prediction['x']}, y: {prediction['y']})")
        print(f"    Tamanho: (Largura: {prediction['width']}, Altura: {prediction['height']})")
        print(f"    ID de detecção: {prediction['detection_id']}")
    print("-" * 50)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    pil_image = cv2_to_pil(frame)
    result = CLIENT.infer(pil_image, model_id="construction-safety-gsnvb/1")
    
    pil_image = desenhar_bounding_boxes(pil_image, result)
    frame_with_boxes = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    mostrar_imagem(frame_with_boxes)
    
    tratar_resultado(result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
plt.close()
