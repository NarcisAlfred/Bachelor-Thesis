import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

directory = r"C:\Users\Alfred\Desktop\Importante\Licenta\Dataset\Training\Raw"
image_directory = r"C:\Users\Alfred\Desktop\Importante\Licenta\Imagini"
plotting = input('Do you want to plot the images? (0 = no, 1 = yes)\n')

for path in os.scandir(directory):
    if path.is_file():
        print("Processing image " + path.name + "\n")
        img = cv2.imread(path.path)
        assert img is not None, "file could not be read"

        # crop image
        #img  = img[20:75,45:80]


        #fgmask = mog.apply(img)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        #fgmask = cv2.erode(fgmask, kernel, iterations=1)
        #fgmask = cv2.dilate(fgmask, kernel, iterations=1)

        # Otsu's thresholding after Gaussian filtering
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        ret,th = cv2.threshold(blur,0,87,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        if(plotting == '1'):
            # plot all the images and their histograms
            images = [blur, 0, th]
            titles = ['Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

            # plotting images
            plt.subplot(1,3,1),plt.imshow(images[0],'gray')
            plt.title(titles[0]), plt.xticks([]), plt.yticks([])
            plt.subplot(1,3,2),plt.hist(images[0].ravel(),256)
            plt.title(titles[1]), plt.xticks([]), plt.yticks([])
            plt.subplot(1,3,3),plt.imshow(images[2],'gray')
            plt.title(titles[2]), plt.xticks([]), plt.yticks([])
            plt.show()

        # find contours
        #contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_L1)

        # draw all contours
        #for i in range(0,len(contours)-1):
            #img = cv2.drawContours(img,contours, i, (0,255,0), 1)
            #cv2.imwrite(image_directory + "\%s"%path.name[0:-4] + "_contour_%d"%i + ".jpg", img)
        cv2.imwrite(image_directory + "\%s"%path.name[0:-4] + "_contour.jpg", th)
            
                


