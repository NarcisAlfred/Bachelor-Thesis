from distutils.file_util import move_file
import openpyxl
import shutil
import os

#main directory of the files that have to be moved
main_dir = r"C:\Users\Alfred\Desktop\Importante\Licenta\Test"

for path in os.scandir(main_dir):
    if path.is_file():
        print(path.path)
        if(path.path[-6:] == "gt.jpg"):
            move_file(path.path,main_dir+"\Labeled")
        else:
            move_file(path.path,main_dir+"\Raw")
