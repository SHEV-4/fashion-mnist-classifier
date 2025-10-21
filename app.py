import streamlit as st
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
from PIL import Image, ImageOps
import numpy as np

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_graphs(history,value):
    if value == "loss":
        loss_value = history["loss"]
        test_loss_values = history["val_loss"]
        epochs = range(1,len(loss_value)+1)
        plt.plot(epochs,loss_value,label='Training Loss')
        plt.plot(epochs,test_loss_values,label='Test Loss',color='red')
        plt.title('Training and Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()
        st.pyplot(plt)
    elif value == "accuracy":
        loss_value = history["accuracy"]
        test_loss_values = history["val_accuracy"]
        epochs = range(1,len(loss_value)+1)
        plt.plot(epochs,loss_value,label='Training Accuracy')
        plt.plot(epochs,test_loss_values,label='Test Accuracy',color='red')
        plt.title('Training and Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()
        st.pyplot(plt)
    
    
    
def image_parser(upload_file,type_model):
    if type_model == "cnn":
        image = Image.open(upload_file)
        image = ImageOps.grayscale(image.resize((28, 28)))   
        image = np.array(image)      
        
                                  
        image = image.astype("float32") / 255.0  
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)          
        return image
    elif type_model == "vgg16":
        image = Image.open(upload_file).convert("RGB")      
        image = image.resize((32, 32))                      
        image = np.array(image).astype("float32") / 255.0   
        image = np.expand_dims(image, axis=0)               
        return image
        
model_fashion = load_model("fashion_mnist_model.keras")
model_vgg16 = load_model("vgg16_fashion_mnist_model.keras")

history_model_fashion = pd.read_json("history_model_fashion.json")
history_model_vgg16 = pd.read_json("history_conv.json")


selected = option_menu(menu_title = None, options = ["VGG 16","Згорткова"],menu_icon="cast",
                       orientation="horizontal")

plot_type = st.sidebar.selectbox('Виберіть графік:', ['Функції втрат', 'Точність'])

uploaded_file = st.sidebar.file_uploader("Виберіть зображення...", type=["jpg", "jpeg", "png"])
button = st.sidebar.button("Протестувати")
if selected == "Згорткова":
    if button: 
        if uploaded_file is not None:
            image_arr = image_parser(uploaded_file,'cnn')
            prediction = model_fashion.predict(image_arr)
            prediction_class = np.argmax(prediction,axis=1)[0]
            st.image(uploaded_file)
            st.write(f"Це - {class_names[prediction_class]}")
            st.write("Ймовірності для всіх класів:")
            for index, prob in enumerate(prediction[0]):
                st.write(f"{class_names[index]}:{prob:.3f}")
    else:
        if plot_type == "Функції втрат":
            get_graphs(history_model_fashion,"loss")
        elif plot_type == "Точність":
            get_graphs(history_model_fashion,"accuracy")
elif selected == "VGG 16":
    if button:
        if uploaded_file is not None:
            image_arr = image_parser(uploaded_file, "vgg16")
            prediction = model_vgg16.predict(image_arr)
            prediction_class = np.argmax(prediction,axis=1)[0]
            st.image(uploaded_file)
            st.write(f"Передбачений клас - {class_names[prediction_class]}")
            st.write("Ймовірності для всіх класів:")
            for index, prob in enumerate(prediction[0]):
                st.write(f"{class_names[index]}:{prob:.3f}")
    else:
        if plot_type == "Функції втрат":
            get_graphs(history_model_vgg16,"loss")
        elif plot_type == "Точність":
            get_graphs(history_model_vgg16,"accuracy")
