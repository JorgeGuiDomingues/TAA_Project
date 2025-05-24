import tensorflow as tf
import os

def testar_imagens_tensorflow(diretorio):
    total = 0
    erros = 0
    for root, _, files in os.walk(diretorio):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                caminho = os.path.join(root, file)
                total += 1
                try:
                    img = tf.io.read_file(caminho)
                    _ = tf.image.decode_image(img)
                except Exception as e:
                    print(f"[ERRO] {caminho} — {e}")
                    os.remove(caminho)
                    erros += 1
    print(f"Imagens testadas: {total}")
    print(f"Imagens inválidas removidas: {erros}")

testar_imagens_tensorflow("train_folder")
testar_imagens_tensorflow("test_folder")