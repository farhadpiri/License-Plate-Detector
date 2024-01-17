import numpy as np
import cv2
import os
from PlateDetector.plate_detector import PlateDtector
ROOT_DIR = os.getcwd()

if(__name__=="__main__"):
    Address = "Cars183.png"
    img = cv2.imread(Address)
    plate_detector = PlateDtector()
    text = plate_detector.detect_plate(img)
    print(text)