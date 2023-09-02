
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Define the path to the directory containing the images
data_dir = 'pes/pest/train'
test_dir='pes/pest/test'

# Define the image size and batch size
img_size = (224, 224)
batch_size = 32

# Define the number of classes
num_classes = len(os.listdir(data_dir))

# Set up the data generators
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

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load the pre-trained ResNet152V2 model
base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a custom output layer for our classification task
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

# Build the model
model = Model(inputs=base_model.input, outputs=x)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator),
)

# Evaluate the model on the test set
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

model.save('pest.h5')

# Make predictions on new images
test_img_path = 't1.jpg'
img = tf.keras.preprocessing.image.load_img(test_img_path, target_size=img_size)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)
predicted_class_idx = np.argmax(prediction)
class_indices = {v: k for k, v in train_generator.class_indices.items()}
predicted_class = class_indices[predicted_class_idx]
print('Predicted class:', predicted_class)