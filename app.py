import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
import cv2
import numpy as np
from mtcnn import MTCNN


DATASET_PATH = 'C:\\Users\\pedro\\OneDrive\\Área de Trabalho\\face_Security\\dataset - Copia'


IMG_WIDTH, IMG_HEIGHT = 128, 128  # Dimensões originais


BATCH_SIZE = 32
EPOCHS = 200





detector = MTCNN()

def detect_and_crop_face(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_img)
    if len(faces) == 0:
        return img  
    
    for face in faces:
        x, y, w, h = face['box']
        cropped_face = img[y:y+h, x:x+w]
        return cv2.resize(cropped_face, (128, 128))  
    return img  










def preprocess_function(img):
  
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    img = detect_and_crop_face(img)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    preprocessing_function=preprocess_function 
)

valid_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    preprocessing_function=preprocess_function  


train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = valid_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)




@tf.keras.utils.register_keras_serializable()  
class CustomMaxPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=(1, 1), strides=(4, 4), padding='valid', **kwargs):
        super(CustomMaxPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.dropout = Dropout(0.5)
        self.regularizer = tf.keras.regularizers.l2(0.01)

    def build(self, input_shape):
        super(CustomMaxPooling, self).build(input_shape)
    

    def call(self, inputs):
       
        x = tf.nn.max_pool2d(inputs, ksize=self.pool_size, strides=self.strides, padding=self.padding.upper())
        return self.dropout(x)
        #return x
    
    
    def get_config(self):
        config = super(CustomMaxPooling, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config

model2 = Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    CustomMaxPooling(pool_size=(1, 1), strides=(4, 4), padding='valid'),
    BatchNormalization(),
    
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    CustomMaxPooling(pool_size=(1, 1), strides=(4, 4), padding='valid'),
    
    BatchNormalization(),
    
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    CustomMaxPooling(pool_size=(1, 1), strides=(4, 4), padding='valid'),
    BatchNormalization(),
    
    Flatten(),
    Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model2.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

model2.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

history = model2.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

model2.save('model2_keras2.keras')


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda ao longo das épocas')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()


plt.show()

