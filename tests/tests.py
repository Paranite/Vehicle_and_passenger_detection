from unittest import TestCase

from yolov3.utils import load_yolo_model, detect_image, write_csv
from face_detection.FaceDetector import FaceDetector
import numpy as np
import cv2
import csv
import os


class Tests(TestCase):
    def test_yolov3_results(self):
        model = load_yolo_model()
        temp_res = detect_image(model, "../model_data/car2.jpg", '')
        temp_res2 = detect_image(model, "../model_data/car2.jpg", '')
        self.assertTrue(np.alltrue(temp_res == temp_res2), "Should be equal")

    def test_RFBnet_results(self):
        model = FaceDetector()
        image = cv2.imread("../model_data/abc123.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        temp_res = model.predict(image)
        temp_res2 = model.predict(image)
        self.assertTrue(np.alltrue(temp_res == temp_res2), "Should be equal")

    def test_write_csv(self):
        temp = [["255", "210", "type", "type", "0"]]
        write_csv(temp, 'test.csv')
        temp2 = None
        with open("test.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                temp2 = row
        os.remove("test.csv")
        self.assertTrue(np.alltrue(temp == [temp2]), "Should be equal")

