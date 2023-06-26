# USAGE
# python predict.py
# import the necessary packages
from configuration import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import tensorflow as tf
import sys
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import ImageFilter
from PIL import Image


def save_images(origImage,origMask,predMask,path,name):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	# set the titles of the subplots
	ax[0].set_title("Imagine")
	ax[1].set_title("Masca originala")
	ax[2].set_title("Masca prezisa")
	# set the layout of the figure and display it
	figure.tight_layout()
	plt.savefig(path)
	plt.close()
	

def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	# set the layout of the figure and display it
	figure.tight_layout()
	figure.show()

def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it to float data type, and scale its pixel values
		image = cv2.imread(imagePath.path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0

		# resize the image and make a copy of it for visualization
		image = cv2.resize(image,(config.INPUT_IMAGE_WIDTH,config.INPUT_IMAGE_HEIGHT))
		orig = image.copy()
		pixel_total = config.INPUT_IMAGE_WIDTH*config.INPUT_IMAGE_HEIGHT

		# find the filename and generate the path to ground truth mask
		groundTruthPath = imagePaths[:-3] + r"Labeled" + "\%s"%path.name[:-4] + "_gt.jpg"

		# load the ground-truth segmentation mask in grayscale mode and resize it
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_HEIGHT))

		# make the channel axis to be the leading one, add a batch dimension, create a PyTorch tensor, and flash it to the current device
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)

		# make the prediction, pass the results through the sigmoid function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()

		# filter out the weak predictions and convert them to integers
		predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)

		predMask = cv2.medianBlur(predMask,5)


		# save masks to a folder
		cv2.imwrite(r"C:\Users\Alfred\Desktop\Importante\Licenta\Visual Studio\CNN\output\Predictions"+"\%s"%path.name[0:22],predMask)
		# Calculate IoU
		intersection = np.logical_and(gtMask, predMask)
		union = np.logical_or(gtMask, predMask)
		iou = np.sum(intersection) / np.sum(union)

		# Calculate accuracy
		imPred = torch.tensor(predMask)
		imMask = torch.tensor(gtMask)
		accuracy = 100 * np.array(torch.sum(imPred == imMask))/(pixel_total)

		print("Accuracy for {} is: {:.4f}%, IoU is: {:.2f}%, predicted {:.0f} pixels out of {:.0f}".format(imagePath.name,accuracy,100*iou,accuracy*pixel_total/100,pixel_total))

		
		# check if the program is running in debug mode ( for plotting purposes)
		gettrace = getattr(sys, 'gettrace', None)

		# prepare a plot for visualization
		#if gettrace():
			#prepare_plot(orig, gtMask, predMask)

		save_images(orig,gtMask,predMask,r"C:\Users\Alfred\Desktop\Importante\Licenta\Visual Studio\CNN\output\Images"+"\%s"%path.name[0:22],path.name[0:18])

		return accuracy, iou
		
		


# load the image path
print("[INFO] Loading up test image paths...")
imagePaths = r"C:\Users\Alfred\Desktop\Importante\Licenta\Visual Studio\CNN\TestSet\Test\Raw"

# load our model from disk and flash it to the current device
print("[INFO] Loading up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

# Initialize the mean accuracy and the number of iterations
count = 0
mean_acc = 0
mean_iou = 0
# iterate over the randomly selected test image paths
for path in os.scandir(imagePaths):
	# make predictions and visualize the results
	acc, iou = make_predictions(unet, path)
	mean_acc += acc
	mean_iou += iou
	count += 1
	
# print the mean accuracy
print("The mean accuracy of the dataset is: {:.4f} and the mean IoU is: {:.2f}".format(mean_acc/count,mean_iou/count))

