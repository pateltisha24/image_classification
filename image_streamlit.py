import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import streamlit as st
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import preprocessing

# Streamlit UI
st.title('Image Classifier')

file_uploaded=st.file_uploader("Choose the file", type=['jpg','jpeg','png'])
if file_uploaded is not None:
    image = Image.open(file_uploaded)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    plt.axis("off")

model = tf.keras.models.load_model('C:/Users/patel/image_classifier.h5')

def classify_image(image):
    # Preprocess the image
    img = np.array(image)
    img = tf.image.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0)

    # Make predictions
    predictions = model.predict(img)
    return predictions
           
 # Classification button
if st.button('Classify'):
    # Perform classification
    predictions = classify_image(image)

    # Display predictions 
    pred_labels = np.argmax(predictions) 
    class_name=['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']

    st.write('Prediction:',class_name[pred_labels])
