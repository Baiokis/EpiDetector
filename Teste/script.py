import os
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw

# Configurar o cliente da API
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ma4uwwnTIDfJ567zH2gL"
)

# Diretório onde as imagens estão e onde as imagens com bounding boxes serão salvas
image_dir = "teste/images"
output_dir = "images_com_boxes"

# Criar o diretório de saída se não existir
os.makedirs(output_dir, exist_ok=True)

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

# Função para desenhar as caixas delimitadoras na imagem
def desenhar_bounding_boxes(image_path, result, output_image_path):
    # Abrir a imagem original
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Desenhar cada predição na imagem
    for prediction in result['predictions']:
        x = prediction['x']
        y = prediction['y']
        width = prediction['width']
        height = prediction['height']
        
        # Calcular as coordenadas da caixa delimitadora
        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2

        # Desenhar o retângulo na imagem
        draw.rectangle([left, top, right, bottom], outline="red", width=3)

    # Salvar a nova imagem com as caixas
    image.save(output_image_path)

# Fazer a inferência para cada imagem e tratar os resultados
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    output_image_path = os.path.join(output_dir, image_file)
    
    # Fazer a inferência
    result = CLIENT.infer(image_path, model_id="construction-safety-gsnvb/1")
    
    # Tratar o resultado
    tratar_resultado(result, image_file)
    
    # Desenhar as caixas delimitadoras na imagem e salvar
    desenhar_bounding_boxes(image_path, result, output_image_path)

print(f"Processo concluído! As imagens com caixas delimitadoras estão em: {output_dir}")
