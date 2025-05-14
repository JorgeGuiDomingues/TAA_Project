import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os


# Define paths for the datasets





def load_images(dataset, img_width, img_height, val_float):
    
    data_train_path = f'{dataset}/train'
    data_test_path = f'{dataset}/test'
    data_train = tf.keras.utils.image_dataset_from_directory(
        data_train_path,
        validation_split=val_float,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=32
    )

    # Get class names from the training data
    data_cat = data_train.class_names

    # Print the number of categories and category names
    print(len(data_cat))
    print(data_cat)

    # Load validation data
    data_val = tf.keras.utils.image_dataset_from_directory(
        data_train_path,
        validation_split=val_float,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=32
    )

    # Load test data
    data_test = tf.keras.utils.image_dataset_from_directory(
        data_test_path,
        image_size=(img_height, img_width),
        shuffle=False,
        batch_size=32,
        validation_split=False
    )
    return data_train, data_cat, data_val, data_test 
# Display a sample of the training images
# plt.figure(figsize=(10, 10))
# for image, labels in data_train.take(1):
#     for i in range(9):
#         plt.subplot(3, 3, i+1)
#         plt.imshow(image[i].numpy().astype('uint8'))
#         plt.title(data_cat[labels[i]])
#         plt.axis('off')

# Define the model
def modelCreate1(data_cat):
    model = Sequential([
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(128),
        layers.Dense(len(data_cat))
    ])
    return model,"model_1"


def modelCreate2(data_cat):
    model = Sequential([
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(128),
        layers.Dense(len(data_cat))
    ])
    return model,"model_2"

def modelCreate3(data_cat):
    model = Sequential([
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(128),
        layers.Dense(len(data_cat))
    ])

    return model,"model_3"


def modelCreate4(data_cat):
    model = Sequential([
        # Pré-processamento de imagens
        layers.Rescaling(1./255),

        # Camadas convolucionais com diferentes tamanhos de filtros
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),

        # Camada de flatten para achatar os dados antes de passá-los para as camadas densas
        layers.Flatten(),

        # Regularização com dropout
        layers.Dropout(0.3),

        # Camada densa
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),

        # Camada de saída
        layers.Dense(len(data_cat), activation='softmax')  # A função softmax é usada para classificação multiclasses
    ])
    
    # Compilar o modelo
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    return model,"model_4"

def modelCreate(data_cat, num):
    if num == 1:
        return modelCreate1(data_cat)
    elif num == 2:
        return modelCreate2(data_cat)
    elif num == 3:
        return modelCreate3(data_cat)
    elif num == 4:
        return modelCreate4(data_cat)
    else:
        raise ValueError("Número do modelo inválido! Use 1, 2, 3 ou 4.")


def create_model_summary(dataset, img_width, img_height, val_float, model_name):
    # Formatando os valores no estilo desejado
    return f"{img_width}x{img_height}_val{int(val_float*100)}_{model_name}_{dataset}"

def train_and_storage(model, data_train, data_cat, data_val, data_test, summar_name):
    # Compile the model
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    # Train the model

    epochs_size = 15
    history = model.fit(data_train, validation_data=data_val, epochs=epochs_size)

    # Plot accuracy and loss
    epochs_range = range(epochs_size)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], label='Training Loss')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')

    os.makedirs(f"result/{summar_name}", exist_ok=True)
    plt.savefig(f'result/{summar_name}/{summar_name}.png')
    
    print("Numero de imagens de treino:", len(data_train))
    print("Numero de imagens de validação:", len(data_val))
    print('train_accuracy', history.history['accuracy'])
    print('val_accuracy', history.history['val_accuracy'])
    print('train_loss', history.history['loss'])
    print('val_loss', history.history['val_loss'])
    test_loss, test_acc = model.evaluate(data_test)
    print(f"Perda no conjunto de teste: {test_loss}")
    print(f"Acuracia no conjunto de teste: {test_acc}")

    with open(f'result/{summar_name}/{summar_name}.txt', 'w') as f:
    # Escreve as informações no arquivo
        f.write(f"Numero de imagens de treino: {len(data_train)}\n")
        f.write(f"Numero de imagens de validacao: {len(data_val)}\n")
        f.write(f"Numero de imagens de test: {len(data_test)}\n\n")
        f.write(f'train_accuracy: {history.history["accuracy"]}\n')
        f.write(f'val_accuracy: {history.history["val_accuracy"]}\n')
        f.write(f'train_loss: {history.history["loss"]}\n')
        f.write(f'val_loss: {history.history["val_loss"]}\n')

        # Avaliação no conjunto de teste
        f.write(f"Perda no conjunto de teste: {test_loss}\n")
        f.write(f"Acuracia no conjunto de teste: {test_acc}\n")
    with open(f'result/{summar_name}/0,{int(test_acc*100)}', 'w') as f:
    # Escreve as informações no arquivo
        f.write("")
    # Mensagem de confirmação
    print(f"Resultados salvos em {summar_name}")
    model.save(f'result/{summar_name}/model_{summar_name}.keras')

def lpt(size, val_float, num_model, pasta):
    dataset="animais"+str(pasta)
    img_width = size
    img_height = size
    data_train, data_cat, data_val, data_test = load_images(dataset, img_width, img_height, val_float)
    model,model_name=modelCreate(data_cat,num_model)
    summar_name=create_model_summary(dataset, img_width, img_height, val_float, model_name)
    train_and_storage(model, data_train, data_cat, data_val, data_test, summar_name)

# Image dimensions



# Load training data
# size=40
# val_float = 0.5
# num_model=1
# pasta=4
# # [s for s in range(20,250,20)]
# for size in range(20,241,20):
#     lpt(size, val_float, num_model, pasta)

# size=40
# val_float = 0.2
# num_model=1
# pasta=4
# for size in [40,80,120]:
#     for val_float in [0.1,0.2,0.4]:
#         lpt(size, val_float, num_model, pasta)

# size=40
# val_float = 0.2
# num_model=1
# pasta=8
# for size in [40,80,120]:
#     for val_float in [0.1,0.2]:
#         lpt(size, val_float, num_model, pasta)

# size=40
# val_float = 0.2
# num_model=1
# pasta=4
# for size in [40,80,120]:
#         for num_model in [2,3,4]:
#             lpt(size, val_float, num_model, pasta)

# size=40
# val_float = 0.2
# num_model=1
# pasta=8
# for size in [40,80,120]:
#         for num_model in [2,3,4]:
#             lpt(size, val_float, num_model, pasta)

# size=40
# val_float = 0.05
# num_model=1
# pasta=8
# for pasta in [4,8]:
#     for size in [40,80,120]:
#         for num_model in [2,3,4]:
#             lpt(size, val_float, num_model, pasta)

size=80
val_float = 0.1
num_model=1
pasta=4
lpt(size, val_float, num_model, pasta)

# lpt(size, val_float, num_model, pasta)
# import os
# os.system("shutdown /s /f /t 0")
