import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd

# directory of the masks
mask_dir = r"C:\Users\Alfred\Desktop\Importante\Licenta\TestSet\Test\Labeled"

# directory of the predictions
predict_dir = r"C:\Users\Alfred\Desktop\Importante\Licenta\Visual Studio\CNN\output\Predictions"

# directory of the excel file
excel = r"C:\Users\Alfred\Desktop\Importante\Licenta\database\testing\Data.xlsx"

# read the excel
df = pd.read_excel(excel)

# iterate through each row
for index, row in df.iterrows():

    # get file name
    filename = row["File Name"]

    # get actual EDV and ESV
    actual_EDV = row["EDV"]
    actual_ESV = row["ESV"]

    # read the masks using PIL
    mask_ED = Image.open(mask_dir+"\%s"%filename+"_ED_gt.jpg")
    mask_ES = Image.open(mask_dir+"\%s"%filename+"_ES_gt.jpg")

    # resize the original masks to the size of the predictions
    mask_ED = mask_ED.resize((256,256))
    mask_ES = mask_ES.resize((256,256))

    # get the number of white pixels in the masks
    mask_pixels_ED = mask_ED.getcolors(256)[-1][0]
    mask_pixels_ES = mask_ED.getcolors(256)[-1][0]

    # read the corresponding predictions
    prediction_ED = Image.open(predict_dir+"\%s"%filename+"_ED.jpg")
    prediction_ES = Image.open(predict_dir+"\%s"%filename+"_ES.jpg")

    # get the number of white pixels in the predictions
    prediction_pixels_ED = prediction_ED.getcolors(256)[-1][0]
    prediction_pixels_ES = prediction_ES.getcolors(256)[-1][0]

    # get the ml / pixel value
    volume_EDV = actual_EDV/mask_pixels_ED
    volume_ESV = actual_ESV/mask_pixels_ES

    # write the predicted values to excel
    df.loc[index,"Predicted EDV"] = prediction_pixels_ED*volume_EDV
    df.loc[index,"Predicted ESV"]  = prediction_pixels_ES*volume_ESV
    
df.to_excel(excel,index=False)
