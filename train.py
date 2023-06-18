# USAGE
# python train.py
# import the necessary packages
from configuration.dataset import SegmentationDataset
from configuration.model import UNet
from configuration import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchsummary import summary
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np


if __name__ == '__main__': 
	# load the image and mask filepaths in a sorted manner
	imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
	maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

	# partition the data into training and testing splits using 85% of the data for training and the remaining 15% for testing
	split = train_test_split(imagePaths, maskPaths,
		test_size=config.TEST_SPLIT, random_state=42)

	# unpack the data split
	(trainImages, testImages) = split[:2]
	(trainMasks, testMasks) = split[2:]

	# define transformations
	transforms = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((config.INPUT_IMAGE_HEIGHT,config.INPUT_IMAGE_WIDTH)), # Resize the image
		transforms.ToTensor() # Convert image to tensor
		]) 

	# create the train and test datasets
	trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
	valDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms) 
	print(f"[INFO] found {len(trainDS)} examples in the training set...")
	print(f"[INFO] found {len(valDS)} examples in the validation set...")

	# create the training and test data loaders
	trainLoader = DataLoader(trainDS, shuffle=True,
		batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
		num_workers=os.cpu_count())
	testLoader = DataLoader(valDS, shuffle=False,
		batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
		num_workers=os.cpu_count())

	# initialize our UNet model
	unet = UNet().to(config.DEVICE)

	# initialize loss function and optimizer
	lossFunc = BCEWithLogitsLoss()
	opt = Adam(unet.parameters(), lr=config.INIT_LR)

	# calculate steps per epoch for training and test set
	trainSteps = len(trainDS) // config.BATCH_SIZE
	testSteps = len(valDS) // config.BATCH_SIZE

	# initialize a dictionary to store training history
	H = {"train_loss": [], "val_loss": []}

	# loop over epochs
	print("[INFO] training the network...")
	startTime = time.time()
	torch.backends.cudnn.benchmark = True
	for e in tqdm(range(config.NUM_EPOCHS)):
		# set the model in training mode
		unet.train()

		# initialize the total training and test loss
		totalTrainLoss = 0
		totalValLoss = 0

		# loop over the training set
		for (i, (x, y)) in enumerate(trainLoader):
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

			# perform a forward pass and calculate the training loss
			pred = unet(x)
			loss = lossFunc(pred, y)

			# first, zero out any previously accumulated gradients, then
			# perform backpropagation, and then update model parameters
			opt.zero_grad()
			loss.backward()
			opt.step()

			# add the loss to the total training loss so far
			totalTrainLoss += loss

		# switch off autograd
		with torch.no_grad():
			# set the model in evaluation mode
			unet.eval()
			# loop over the test set
			for (x, y) in testLoader:
				# send the input to the device
				(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

				# make the predictions and calculate the test loss
				pred = unet(x)
				totalValLoss += lossFunc(pred, y)


		# calculate the average training and test loss
		avgTrainLoss = totalTrainLoss / trainSteps
		avgTestLoss = totalValLoss / testSteps

		# update our training history
		H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
		H["val_loss"].append(avgTestLoss.cpu().detach().numpy())

		# print the model training and test information
		print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
		print("Train loss: {:.4f}%, validation loss: {:.4f}%".format(100*avgTrainLoss, 100*avgTestLoss))

	# display the total time needed to perform the training and total accuracy
	endTime = time.time()
	print("[INFO] total time taken to train the model: {:.2f}s".format(
		endTime - startTime))

	# plot the training loss
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H["train_loss"], label="train_loss")
	plt.plot(H["val_loss"], label="validation_loss")
	plt.title("Training Loss on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	plt.savefig(config.PLOT_PATH)

	# serialize the model to disk
	torch.save(unet, config.MODEL_PATH)