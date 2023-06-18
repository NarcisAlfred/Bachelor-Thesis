import PySimpleGUI as gui

# Define the model variable
model = None
image = None
# Define the title
title = r"Echocardiography Segmentation"


# Define the main layout
layout = [
    # Text
    [gui.Text("Welcome to Echocardiography Segmentation App.\nYou can begin the segmentation by first selecting the model that you will be using and then selecting the image folder.", key="txtMain", size=(50, 5), # Main text of app
       auto_size_text = True, expand_x = True, expand_y = True)],
    [gui.Text(f'Current model used is: {model}', key="textModel", visible = False)], # Text to display the current model
    [gui.Text(f'Current file: {image}', key="textFile", visible = False)], # Text to display the current image processed
    # Buttons
    [gui.Button('Choose AI model', key="bttnModel", visible = True)], # Button to load the model into memory
    [gui.Button('Choose Images', key="bttnImages", visible = True)], # Button to select the image for segmentation
    [gui.Button('Display Images', key="bttnPlot", visible = False)], # Button to display the plots
    [gui.Button('Calculate EDV', key="bttnEDV", visible = False)], # Button to calculate EDV
    [gui.Button('Calculate EDS', key="bttnEDS", visible = False)], # Button to calculate EDS
    [gui.Button('Go Back', key="bttnBack1", visible = False)] # Button to go back to the main menu
]

# Create the window
window = gui.Window(title, layout, margins=(300,300))

# Create the event loop
while True:
    event, values = window.read()
    
    match event:
        
        # Select the AI Model
        case "bttnModel":
            model = gui.popup_get_file('Select your model',no_window=True) # Browse and select the model

        # Select the images folder
        case "bttnImages":
            # If the model was not selected yet
            if model is None:
                gui.popup("Please select your model first") # Add a pop-up that the model has to be selected first

            # If the model was already selected
            else:
                #Hide
                window["txtMain"].Update(visible = False) # Hide the main text
                window["bttnModel"].Update(visible = False) # Hide "Choose AI model" button
                window["bttnImages"].Update(visible = False) # Hide "Choose Images" button

                #Display
                window["textModel"].Update(value = f'Current model used is: {model.split("/")[-1]}', visible = True) # Display the name of the model
                window["bttnBack1"].Update(visible = True) # Display the "Go Back" button
                
                image = gui.popup_get_file('Select your image') # Browse and select the image
                if image is not None: 
                    if image != '':
                        option_mask = gui.popup_yes_no(f'You selected the {image.split("/")[-1]} image. Does it have a mask?',  title = "Mask Selection")
                        if option_mask == "Yes":
                            mask = gui.popup_get_file('Select the corresponding mask') # Browse and select the mask
                        window["bttnPlot"].Update(visible = True)
                        window["textFile"].Update(value = f'Current file: {image.split("/")[-1]}', visible = True) # Update the image name

        case "bttnBack1":
            # Display
            window["txtMain"].Update(visible = True) # Display the main text
            window["bttnModel"].Update(visible = True) # Display "Choose AI model" button
            window["bttnImages"].Update(visible = True) # Display "Choose Images" button

            # Hide
            window["textModel"].Update(visible = False) # Hide the name of the model
            window["bttnBack1"].Update(visible = False) # Hide the go back button
            window["bttnPlot"].Update(visible = False) # Hide the go back button
            window["textFile"].Update(visible = False) # Hide the image name

            # Clear variables from memory
            model = None
            image = None
            mask = None
            
        # End program if user closes the window
        case gui.WIN_CLOSED:
            break
        
window.close()

