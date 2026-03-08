import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout
from keras.saving import register_keras_serializable
from mtcnn import MTCNN
from tensorflow.keras.regularizers import l2

detector = MTCNN()

def detect_and_crop_face(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_img)
=    if len(faces) == 0:
        return img  
    for face in faces:
        x, y, w, h = face['box']
        cropped_face = img[y:y+h, x:x+w]
        return cv2.resize(cropped_face, (128, 128)) 
    return img 


@tf.keras.utils.register_keras_serializable()  
class CustomMaxPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='valid', **kwargs):
        super(CustomMaxPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.dropout = Dropout(0.5)
        self.regularizer = l2(0.01)

    def build(self, input_shape):
        super(CustomMaxPooling, self).build(input_shape)

    def call(self, inputs):
        
        x = tf.nn.max_pool2d(inputs, ksize=self.pool_size, strides=self.strides, padding=self.padding.upper())
        return self.dropout(x)

    def get_config(self):
        config = super(CustomMaxPooling, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config


model = load_model('model2_keras2.keras', custom_objects={'CustomMaxPooling': CustomMaxPooling})

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'Erro ao carregar a imagem: {image_path}')
    img = detect_and_crop_face(img)
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img


test_image_path = 'C:\\Users\\pedro\\Downloads\\teste-2.jpg'

preprocessed_image = preprocess_image(test_image_path)


print("Forma da imagem após redimensionamento:", preprocessed_image.shape)

prediction = model.predict(preprocessed_image)[0][0]

print(f'Previsão bruta: {prediction:.6f}')
print(f'Probabilidade de não ser você: {prediction * 100:.2f}%')

if prediction <= 0.01:
    print('É você!')
else:
    print('Não é você!')



