import tensorflow as tf
import streamlit as st
import numpy as np

# Lista de categorias
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic',
    'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion',
    'orange', 'paprika', 'pear', 'peas', 'sweetcorn', 'sweetpotato', 'tomato',
    'turnip', 'watermelon'
]

# Carregar o modelo completo
model = tf.keras.models.load_model(r'C:\Users\cue2ca\Desktop\Imagem\Image_classify.keras')

# Recompilar o modelo com a função de perda correta
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Interface Streamlit
st.title('Classificação de Imagens')

# Carregar e preparar a imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type="jpg")
if uploaded_file is not None:
    # Caminho da imagem carregada
    image_path = uploaded_file
    
    img_height = 180
    img_width = 180
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    image_arr = tf.keras.utils.img_to_array(image_load)
    image_bat = tf.expand_dims(image_arr, 0)  # Expande dimensões para criar um batch

    # Fazer a previsão
    predict = model.predict(image_bat)

    # Aplicar softmax para obter probabilidades
    score = tf.nn.softmax(predict[0])

    # Mostrar a imagem e a previsão no Streamlit
    st.image(image_path, caption='Imagem de Entrada', use_column_width=True)
    st.write('This is a {} with the precision of {:0.2f}%'.format(data_cat[np.argmax(score)], np.max(score) * 100))