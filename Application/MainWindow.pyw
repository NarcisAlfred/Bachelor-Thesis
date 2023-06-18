# Echocardiography Segmentation
import PySimpleGUI as gui
import os.path
import cv2
import torch
import numpy as np
import tensorflow as tf
import torch.nn.functional as F

# check if the temporary folder exists
if not os.path.exists(os.getcwd()+"\Application\Temp"):
    os.mkdir(os.getcwd()+"\Application\Temp")

temp_path = os.getcwd()+"\Application\Temp"

# set the device for processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define the model variable
model = None
raw = None

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
    [gui.Image(key="imageRaw")],
    [gui.Image(key="imageMask")],
    [gui.Image(key="imagePredict")]
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
            unet = torch.load(model).to(DEVICE)
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
                print(image_raw)
                image_cv2 = cv2.imread(image_raw) # read the image
                image_cv2 = cv2.resize(image_cv2, (128,128)) # resize image
                cv2.imwrite(temp_path+"\%s"%values["listRaw"][0]+".png", image_cv2) # convert image to .png
                raw = temp_path+"\%s"%values["listRaw"][0]+".png" # get the new image path

                window["labelRaw"].update(values["listRaw"][0])
                window["imageRaw"].update(filename=raw)
            except:
                pass

        # Select masked image from the list
        case "listMask":  
            try:
                image_mask = values["folderMask"]+"/%s"%values["listMask"][0]
                image_cv2 = cv2.imread(image_mask) # read the image
                image_cv2 = cv2.resize(image_cv2, (128,128)) # resize image
                cv2.imwrite(temp_path+"\%s"%values["listMask"][0]+".png", image_cv2) # convert image to .png
                mask = temp_path+"\%s"%values["listMask"][0]+".png" # get the new image path

                window["imageMask"].update(filename=mask)
            except:
                pass
        
        # Predict the mask
        case "bttnPredict":
            if model is None:
                gui.popup("Please select a model!") # Add a pop-up that the model has to be selected
            elif raw is None:
                gui.popup("Please select an image!") # Add a pop-up that the image has to be selected
            else
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
                    image = np.transpose(image, (2, 0, 1))
                    image = np.expand_dims(image, 0)
                    image = torch.from_numpy(image).to(DEVICE)

		            # make the prediction, pass the results through the sigmoid function, and convert the result to a NumPy array
                    predMask = unet(image).squeeze()
                    predMask = torch.sigmoid(predMask)
                    predMask = predMask.cpu().numpy()

		            # filter out the weak predictions and convert them to integers
                    predMask = (predMask > 0.5) * 255
                    predMask = predMask.astype(np.uint8)
                    
                image_cv2 = cv2.resize(predMask, (128,128)) # resize image
                # Save the prediction temporary
                cv2.imwrite(temp_path+"\%s"%values["listRaw"][0]+"_predict.png", image_cv2) # convert image to .png
                predict = temp_path+"\%s"%values["listRaw"][0]+"_predict.png"

                # Update the predicted image
                window["imagePredict"].Update(filename=predict)
                del unet

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