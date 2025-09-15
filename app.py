import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np 

st.header('deep-learning-project-vegetable-fruit-identifier-model')

# Load the trained model (make sure you saved it as .keras or .h5, not .ipynb)

transfer_model = load_model(r"image_classification.keras")


# Class categories
class_names = [
    'apple','banana','beetroot','bell pepper','cabbage','capsicum','carrot','cauliflower',
    'chilli pepper','corn','cucumber','eggplant','garlic','ginger','grapes','jalepeno',
    'kiwi','lemon','lettuce','mango','onion','orange','paprika','pear','peas','pineapple',
    'pomegranate','potato','raddish','soy beans','spinach','sweetcorn','sweetpotato',
    'tomato','turnip','watermelon'
]

# Define image size (same as training)
img_height, img_width = 180, 180
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    predict = transfer_model.predict(img_bat)
    score = tf.nn.softmax(predict[0])

    st.image(image_load, width=200)
    st.write(f"Veg/Fruit in image is **{class_names[np.argmax(score)]}**")
    st.write(f"With accuracy of **{np.max(score) * 100:.2f}%**")
