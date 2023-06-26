import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# directory of the masks
mask_dir = r"C:\Users\Alfred\Desktop\Importante\Licenta\Visual Studio\CNN\TestSet\Test\Labeled"

# directory of the predictions
predict_dir = r"C:\Users\Alfred\Desktop\Importante\Licenta\Visual Studio\CNN\output\Predictions"

# output directory
output = r"C:\Users\Alfred\Desktop\Importante\Licenta\Visual Studio\CNN\output"

# Initialize empty lists
actual_edv = []
predicted_edv = []

actual_esv = []
predicted_esv = []

actual_ef = []
predicted_ef = []

count = 0
# Calculate the volume and ejection fraction for masks
for path in os.scandir(mask_dir):
    if path.name.find(r"ED") != -1:
        img = cv2.imread(path.path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(1102,669))
        _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour_index = -1
        largest_contour_area = 0
        for i, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            if contour_area > largest_contour_area:
                largest_contour_area = contour_area
                largest_contour_index = i

        actual_edv.append(round(cv2.contourArea(contours[largest_contour_index])*0.0027225,2))
        count+=1
    else:
        img = cv2.imread(path.path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(1102,669))
        _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour_index = -1
        largest_contour_area = 0
        for i, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            if contour_area > largest_contour_area:
                largest_contour_area = contour_area
                largest_contour_index = i

        actual_esv.append(round(cv2.contourArea(contours[largest_contour_index])*0.0027225,2))
        count+=1
    if count == 2:
        actual_ef.append(round(100*(actual_edv[-1]-actual_esv[-1])/actual_edv[-1],2))
        count = 0

count = 0
# Calculate the volume and ejection fraction for predictions
for path in os.scandir(predict_dir):
    if path.name.find(r"ED") != -1:
        img = cv2.imread(path.path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(1102,669))
        _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour_index = -1
        largest_contour_area = 0
        for i, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            if contour_area > largest_contour_area:
                largest_contour_area = contour_area
                largest_contour_index = i

        predicted_edv.append(round(cv2.contourArea(contours[largest_contour_index])*0.0027225,2))
        count += 1
    else:
        img = cv2.imread(path.path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(1102,669))
        _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour_index = -1
        largest_contour_area = 0
        for i, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            if contour_area > largest_contour_area:
                largest_contour_area = contour_area
                largest_contour_index = i

        predicted_esv.append(round(cv2.contourArea(contours[largest_contour_index])*0.0027225,2))
        count += 1
    if count == 2:
        predicted_ef.append(round(100*(predicted_edv[-1]-predicted_esv[-1])/predicted_edv[-1],2))
        count = 0

# Save plots
# EDV
plt.scatter(actual_edv,predicted_edv)
p1 = max(max(predicted_edv),max(actual_edv))
p2 = min(min(predicted_edv),min(actual_edv))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('Actual EDV (mL)')
plt.ylabel('Predicted EDV (mL)')
plt.title('End dyastolic volume')
plt.grid(True)
plt.savefig(output+"\EDV_plot.png")
plt.close()

# ESV
plt.scatter(actual_esv,predicted_esv)
p1 = max(max(predicted_esv),max(actual_esv))
p2 = min(min(predicted_esv),min(actual_esv))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('Actual ESV (mL)')
plt.ylabel('Predicted ESV (mL)')
plt.title('End sistolic volume')
plt.grid(True)
plt.savefig(output+"\ESV_plot.png")
plt.close()

# EDV
plt.scatter(actual_ef,predicted_ef)
p1 = max(max(predicted_ef),max(actual_ef))
p2 = min(min(predicted_ef),min(actual_ef))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('Actual EF (%)')
plt.ylabel('Predicted EF (%)')
plt.title('Ejection fraction volume')
plt.grid(True)
plt.savefig(output+"\EF_plot.png")
plt.close()