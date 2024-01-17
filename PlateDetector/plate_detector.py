import copy

import cv2

from ObjectDetector.YOLOv7.YOLO_Plate_Detector import YOLO_Plate_Detector
from PlateDetector.PlateRecognizer import PlateRecognizer
import torch
import numpy as np

class PlateDtector:
    def __init__(self):
        self.plate_detector = YOLO_Plate_Detector()
        self.plate_recognizer = PlateRecognizer()


    def xyxy2xywh(self,x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def detect_plate(self,image):

        plates_dets = self.plate_detector.detect(copy.deepcopy(image))
        if(len(plates_dets) == 0):
            return "no plate detected"

        plate_numbers = self.plate_recognizer.recognize(plates_dets,image)
        return plate_numbers



