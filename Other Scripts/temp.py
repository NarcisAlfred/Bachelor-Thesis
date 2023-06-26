import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

# 1102,669
# directory of the masks

mask_dir = r"C:\Users\Alfred\Desktop\Importante\Licenta\TestSet\Test\Labeled"
for path in os.scandir(mask_dir):
    if path.is_file():
        img = cv2.imread(path.path,cv2.IMREAD_GRAYSCALE)
        print(path.name)
        img = cv2.resize(img,(1102,669))

        _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area = cv2.contourArea(contours[-1])
        print("Area of the LV: ", area)
        print("The volume is: ",area/72*2.54)
        print("\n")