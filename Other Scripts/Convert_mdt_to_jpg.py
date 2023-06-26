import SimpleITK as sitk
import os
from PIL import Image
import sys
import cv2
import numpy as np
import matplotlib as plt

directory = r"C:\Users\Alfred\Desktop\Importante\Licenta\database\testing"
image_directory = r"C:\Users\Alfred\Desktop\Importante\Licenta\Dataset\Test"
for subdir, dirs, files in os.walk(directory):
    count = 0
    for file in files:
        if(file[-3:] == "mhd"):
            print(os.path.join(subdir, file))
            raw_image = sitk.GetArrayFromImage(sitk.ReadImage(subdir +"\%s"%file, sitk.sitkFloat32))
            count +=1
            img = (Image.fromarray(raw_image[0])).resize((512,512)) #intai era 112,112, am schimbat la 512,512
            if (count%2 == 0):
                img = img.convert("RGBA")
                pixdata = img.load()

                # change pixel color
                for y in range(img.size[1]):
                    for x in range(img.size[0]):
                        # change main part to white
                        if pixdata[x, y] == (1,1,1,255):
                            pixdata[x, y] = (255,255,255,255)

                # add red contour to the main part
                for y in range(img.size[1]):
                    for x in range(img.size[0]):

                        # adds contour by checking if the pixel color is black (shades of black) and if there are at least 3 white pixels in any of the direction
                        # check right
                        if x<509 and (pixdata[x, y] == (0,0,0,255) or pixdata[x, y] == (2,2,2,255))\
                           and pixdata[x+1, y] == (255,255,255,255) and pixdata[x+2, y] == (255,255,255,255) and pixdata[x+3, y] == (255,255,255,255):
                            pixdata[x, y] = (255,0,0,255)
                        # check below
                        elif y<509 and (pixdata[x, y] == (0,0,0,255) or pixdata[x, y] == (2,2,2,255))\
                           and pixdata[x, y+1] == (255,255,255,255) and pixdata[x, y+2] == (255,255,255,255) and pixdata[x, y+3] == (255,255,255,255):
                            pixdata[x, y] = (255,0,0,255)
                        # check above
                        elif y>2 and (pixdata[x, y] == (0,0,0,255) or pixdata[x, y] == (2,2,2,255))\
                           and pixdata[x, y-1] == (255,255,255,255) and pixdata[x, y-2] == (255,255,255,255) and pixdata[x, y-3] == (255,255,255,255):
                            pixdata[x, y] = (255,0,0,255)
                        # check left
                        elif x>2 and (pixdata[x, y] == (0,0,0,255) or pixdata[x, y] == (2,2,2,255))\
                           and pixdata[x-1, y] == (255,255,255,255) and pixdata[x-2, y] == (255,255,255,255) and pixdata[x-3, y] == (255,255,255,255):
                            pixdata[x, y] = (255,0,0,255)
                
                # change the main part to another color           
                for y in range(img.size[1]):
                    for x in range(img.size[0]):

                        # if the current pixel is white, the pixel from the left is either red, blue or white, the pixel from below is white, red or black (shades of black) 
                            # and above it is either blue or red 
                        if (x>1 and y<511) and pixdata[x, y] == (255,255,255,255)\
                            and (pixdata[x-1, y] == (255,0,0,255) or pixdata[x-1, y] == (0, 0, 255, 255) or pixdata[x-1, y] == (255,255,255,255))\
                                and (pixdata[x, y+1] == (255,255,255,255) or pixdata[x, y+1] == (255,0,0,255)\
                                    or (pixdata[x, y+1] == (0,0,0,255) or pixdata[x, y+1] == (2,2,2,255)))\
                                        and (pixdata[x,y-1] == (255,0,0,255) or pixdata[x, y-1] == (0, 0, 255, 255)):
                            pixdata[x, y] = (0, 0, 255, 255)

                # delete everything and change main to white          
                for y in range(img.size[1]):
                    for x in range(img.size[0]):
                        # if the pixel is not blue, turn it to black
                        if pixdata[x, y] != (0, 0, 255, 255):
                            pixdata[x, y] = (0, 0, 0, 255)
                        # if the current pixel is blue, the pixels from left and right are either red, blue or white, we turn the pixel to white
                        if (x>0 and x<511) and pixdata[x, y] == (0, 0, 255, 255)\
                           and (pixdata[x-1, y] == (255,0,0,255) or pixdata[x-1, y] == (0, 0, 255, 255)\
                                or pixdata[x-1, y] == (255, 255, 255, 255)) and (pixdata[x+1, y] == (255,0,0,255)\
                                    or pixdata[x+1, y] == (0, 0, 255, 255) or pixdata[x-1, y] == (255, 255, 255, 255)):
                            pixdata[x, y] = (255, 255, 255, 255)
                        # else, turn the pixel to black

                # remove the blue residuals
                for y in range(img.size[1]):
                    for x in range(img.size[0]):
                        if pixdata[x, y] == (0, 0, 255, 255):
                            pixdata[x, y] = (0, 0, 0, 255)

                # convert image to grayscale
                img = img.convert("L")
                img.save(image_directory + "\Labeled\%s.jpg"%file[0:-4])
            else:
                img = img.convert("L")
                img.save(image_directory + "\Raw\%s.jpg"%file[0:-4])
        if(count == 4): #era 4 inainte
           break
