import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Define paths
model_path = 'path/to/model_checkpoint.h5'

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Define class names
class_names = ['Healthy', 'Diseased']

def predict_image(image):
    image_array = img_to_array(load_img(image, target_size=(224, 224))) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_class = int(prediction[0][0] >= 0.5)
    class_name = class_names[predicted_class]

    return {"class": class_name, "confidence": float(prediction[0][0])}
