import os
from tkinter import END
import cv2

#get all files inside the videos folder

#directory containing the videos
directory = input('What is the directory that contains the videos? (Full path)\n')
#directory that will contain the frames
image_directory = input('What is the directory where you want to paste the frame? (Full path)\n')
for path in os.scandir(directory):
    if path.is_file():
        print(path.name)
        video_capture = cv2.VideoCapture(path.path)
        succes,image = video_capture.read()
        count = 0
        count_label_dis = 0
        count_label_sis = 0
        while succes:
            #save frame as jpeg
            if (count % 25 == 0 and count <= 75):
                if(count % 2 == 0):
                    label = "dias"
                    count_label_dis += 1
                    cv2.imwrite(image_directory + "\%s"%path.name[0:-4] + "_%s_%d.jpg" %(label,count_label_dis), image)
                    print('Reading a new diastolic frame', succes)
                else:
                    label = "sis"
                    count_label_sis += 1
                    cv2.imwrite(image_directory + "\%s"%path.name[0:-4] + "_%s_%d.jpg" %(label,count_label_sis), image) 
                    print('Reading a new sistolic frame', succes)
            succes,image = video_capture.read()
            
            count+=1