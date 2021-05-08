import tensorflow as tf
import cv2
import numpy as np


class FaceDetector:
    def __init__(self):
        self.model = self.load_model()

    @staticmethod
    def load_model():
        return tf.keras.models.load_model("face_detection/RFB")

    def predict(self, loaded_image):
        img_resize = cv2.resize(loaded_image, (320, 240))
        img_resize = img_resize - 127.0
        img_resize = img_resize / 128.0
        return self.model.predict(np.expand_dims(img_resize, axis=0))

    def get_detection_count(self, loaded_image, threshold):
        result = self.predict(loaded_image)
        result = len([i for i in result if i[1] >= threshold])
        return result
