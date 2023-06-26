# Echocardiography Segmentation
import PySimpleGUI as gui
import os.path
import cv2
import torch
import numpy as np
import tensorflow as tf
import torch.nn.functional as F

# check if the temporary folder exists
if not os.path.exists(os.getcwd()+"\Temp"):
    os.mkdir(os.getcwd()+"\Temp")

temp_path = os.getcwd()+"\Temp"

# set the device for processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define variables
model = None
raw = None
mask = None 
predict = None
pEDV = None
pESV = None
pEF = None
aEDV = None
aESV = None
aEF = None

# Define the title
title = r"Echocardiography Segmentation"

# Columns that contain the file lists
file_list_column = [
    [   # Raw Images folder selection
        gui.Text("Image Folder"),
        gui.In(size=(15, 1), enable_events=True, key="folderRaw"),
        gui.FolderBrowse(),

        # Masked Images folder selection
        gui.Text("Mask Folder"),
        gui.In(size=(15, 1), enable_events=True, key="folderMask"),
        gui.FolderBrowse(),
    ],
    # AI and predict buttons
    [gui.HSeparator()],
    # Button to choose the AI Model
    [gui.Button('Choose AI Model', key="bttnModel"),
     # Button to start prediction
     gui.Button('Predict Mask', key="bttnPredict"), 
     # Model Name
     gui.Text(f'Current model is: {model}', key="labelModel"),
     # Image name
     gui.Text(f'Current image is: {raw}', key="labelRaw")],
    # Raw and Mask Images lists
    [gui.Listbox(values=[], enable_events=True, size=(40, 20), expand_y = True, key="listRaw"),gui.Listbox(values=[], enable_events=True, size=(40, 20), expand_y = True, key="listMask")]
]

# Column that contains the name of the image and the images
image_viewer_column = [
    [gui.Button('Get Predicted EF',expand_x = True, key="bttnPredictEF"),gui.Button('Get Actual EF', expand_x = True, key="bttnActualEF")],
    [gui.Text(f'Predicted EDV:{pEDV}', key="pEDV"),gui.Text(f'Predicted ESV:{pESV}', key="pESV"),gui.Text(f'Predicted EF:{pEF}',key="pEF")],
    [gui.Text(f'Actual EDV:{aEDV}', key="aEDV"),gui.Text(f'Actual ESV:{aESV}',key="aESV"),gui.Text(f'Actual EF:{aEF}',key="aEF")],
    [gui.HSeparator()],
    [gui.Text('Raw Image',key='textRaw', visible = False), gui.Text('                 Ground Truth',key='textMask',visible = False), gui.Text('                 Predicted Mask',key='textPred',visible = False)],
    [gui.Image(key="imageRaw"),
    gui.Image(key="imageMask"),
    gui.Image(key="imagePredict")]
]

# Main Layout
layout = [
    [gui.Column(file_list_column),
     gui.VSeparator(),
     gui.Column(image_viewer_column),
    ]
]

# Define the main window
window = gui.Window(title, layout)

# Run the App
while True:

    event, values = window.read()
    match event:
        # Select the AI Model
        case "bttnModel":
            model = gui.popup_get_file('Select your model',no_window=True) # Browse and select the model
            window["labelModel"].Update(value = f'Current model used is: {model.split("/")[-1]}') # Display the name of the model
        
        # Add raw images to list
        case "folderRaw":
            raw_folder = values["folderRaw"]
            try:
                # Get list of files in folder
                file_list = os.listdir(raw_folder)

            except:
                file_list = []
            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(raw_folder, f))
                and f.lower().endswith((".jpg"))
            ]
            window["listRaw"].update(fnames)

        # Add masked images to list
        case "folderMask":
            mask_folder = values["folderMask"]
            try:
                # Get list of files in folder
                file_list = os.listdir(mask_folder)

            except:
                file_list = []
            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(mask_folder, f))
                and f.lower().endswith((".jpg"))
            ]
            window["listMask"].update(fnames)

        # Select raw image from the list
        case "listRaw":  
            try:
                image_raw = values["folderRaw"]+"/%s"%values["listRaw"][0]
                image_cv2 = cv2.imread(image_raw) # read the image
                image_cv2 = cv2.resize(image_cv2, (128,128)) # resize image
                cv2.imwrite(temp_path+"\%s"%values["listRaw"][0][0:-4]+".png", image_cv2) # convert image to .png
                raw = temp_path+"\%s"%values["listRaw"][0][0:-4]+".png" # get the new image path

                window["labelRaw"].update(values["listRaw"][0][0:11])
                window["imageRaw"].update(filename=raw)
                window["textRaw"].update(visible=True)
            except:
                pass

        # Select masked image from the list
        case "listMask":  
            try:
                image_mask = values["folderMask"]+"/%s"%values["listMask"][0]
                image_cv2 = cv2.imread(image_mask) # read the image
                image_cv2 = cv2.resize(image_cv2, (128,128)) # resize image
                cv2.imwrite(temp_path+"\%s"%values["listMask"][0][0:-4]+".png", image_cv2) # convert image to .png
                mask = temp_path+"\%s"%values["listMask"][0][0:-4]+".png" # get the new image path

                window["imageMask"].update(filename=mask)
                window["textMask"].update(visible=True)
            except:
                pass
        
        # Predict the mask
        case "bttnPredict":
            if model is None:
                gui.popup("Please select a model!") # Add a pop-up that the model has to be selected
            elif raw is None:
                gui.popup("Please select an image!") # Add a pop-up that the image has to be selected
            else:
                unet = torch.load(model).to(DEVICE)
                # set model to evaluation mode
                try:
                    unet.eval()
                except:
                    gui.popup("The model selected is either incompatible or corrupt")
                    break
                # turn off gradient tracking
                with torch.no_grad():
                    # load the image from disk, swap its color channels, cast it to float data type, and scale its pixel values
                    image = cv2.imread(image_raw)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = image.astype("float32") / 255.0
                    
                    # make the channel axis to be the leading one, add a batch dimension, create a PyTorch tensor, and flash it to the current device
                    image = cv2.resize(image,(256,256))
                    image = np.transpose(image, (2, 0, 1))
                    image = np.expand_dims(image, 0)
                    image = torch.from_numpy(image).to(DEVICE)

		            # make the prediction, pass the results through the sigmoid function, and convert the result to a NumPy array
                    predMask = unet(image).squeeze()
                    predMask = torch.sigmoid(predMask)
                    predMask = predMask.cpu().numpy()

		            # filter out the weak predictions and convert them to integers
                    predMask = (predMask > 0.7) * 255
                    predMask = predMask.astype(np.uint8)

                    # smothen the mask
                    predMask = cv2.medianBlur(predMask,5)
                    
                    image_cv2 = cv2.resize(predMask, (128,128)) # resize the mask back
                    # Save the prediction temporary
                    cv2.imwrite(temp_path+"\%s"%values["listRaw"][0][0:-4]+"_predict.png", image_cv2) # convert image to .png
                    predict = temp_path+"\%s"%values["listRaw"][0][0:-4]+"_predict.png"

                # Update the predicted image
                window["imagePredict"].Update(filename=predict)
                window["textPred"].update(visible=True)
                del unet

        case "bttnPredictEF":
            if raw is not None:
                try:
                    image_ED = cv2.imread(temp_path+"\%s"%values["listRaw"][0][0:11]+"_2CH_ED_predict.png",cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(image_ED,(1102,669))
                    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    pEDV = round(cv2.contourArea(contours[-1])*0.0027225,2)
                except:
                    gui.popup("End diastolic moment was not segmented or could not be found!")
                try:
                    image_ES = cv2.imread(temp_path+"\%s"%values["listRaw"][0][0:11]+"_2CH_ES_predict.png",cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(image_ES,(1102,669))
                    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    pESV = round(cv2.contourArea(contours[-1])*0.0027225,2)

                    pEF = round(100*(pEDV-pESV)/pEDV,2)

                    window["pEDV"].Update(value = f'Predicted EDV: {pEDV}mL') # Display the EDV
                    window["pESV"].Update(value = f'Predicted ESV: {pESV}mL') # Display the ESV
                    window["pEF"].Update(value = f'Predicted EF: {pEF}%') # Display the EF
                except:
                    gui.popup("End systolic moment was not segmented or could not be found")
            else:
                gui.popup("No image was selected for prediction!")

        case "bttnActualEF":
            if mask is not None:
                try:
                    image_ED = cv2.imread(temp_path+"\%s"%values["listRaw"][0][0:11]+"_2CH_ED_gt.png",cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(image_ED,(1102,669))
                    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    aEDV = round(cv2.contourArea(contours[-1])*0.0027225,2)
                except:
                    gui.popup("End diastolic moment was not segmented or could not be found!")
                try:
                    image_ES = cv2.imread(temp_path+"\%s"%values["listRaw"][0][0:11]+"_2CH_ES_gt.png",cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(image_ES,(1102,669))
                    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    aESV = round(cv2.contourArea(contours[-1])*0.0027225,2)

                    aEF = round(100*(aEDV-aESV)/aEDV,2)

                    window["aEDV"].Update(value = f'Actual EDV: {aEDV}mL') # Display the EDV
                    window["aESV"].Update(value = f'Actual ESV: {aESV}mL') # Display the ESV
                    window["aEF"].Update(value = f'Actual EF: {aEF}%') # Display the EF
                except:
                    gui.popup("End systolic moment was not segmented or could not be found")
            else:
                gui.popup("No mask was selected!")

        # Close the window
        case gui.WIN_CLOSED:

            # Remove any files or subfolders within the folder
            for root, dirs, files in os.walk(temp_path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)

            # Delete the empty folder
            os.rmdir(temp_path)
            break


window.close()