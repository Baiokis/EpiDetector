import os
from inference_sdk import InferenceHTTPClient

# Configurar o cliente da API
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ma4uwwnTIDfJ567zH2gL"
)

# Diretório onde as imagens estão
image_dir = "images"

# Listar todas as imagens no diretório
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

# Função para tratar e exibir resultados de forma organizada
def tratar_resultado(result, image_file):
    print(f"Resultado para {image_file}:")
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

# Fazer a inferência para cada imagem e tratar os resultados
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    result = CLIENT.infer(image_path, model_id="construction-safety-gsnvb/1")
    tratar_resultado(result, image_file)
