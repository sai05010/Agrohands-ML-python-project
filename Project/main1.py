import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os


def getPrediction(filename):
    mymodel=load_model("models/pest.h5")
    data_dir = 'pes/pest/train'
    img_size = (224, 224)
    batch_size = 32

    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )


    test_img_path = 'static/images/'+filename
    img = tf.keras.preprocessing.image.load_img(test_img_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0


    prediction = mymodel.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    pred_class = class_indices[predicted_class_idx]
    print("Weed IS",pred_class)
    return pred_class