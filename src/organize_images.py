import h5py   
import numpy as np  
import cv2
from PIL import Image
import tarfile
import os    


def tar_to_jpeg():    
    ###this converts tar files to usable images, commented out because I have the images already
    for entry in os.scandir('.\\xray_images'): #this is where tar files are saved
        if entry.path.endswith(".tar.gz"):
            print("file input: %s" % (entry.path,))
            my_tar = tarfile.open(entry.path)
            my_tar.extractall('.\\image_files') 
            my_tar.close()       

def images_to_folders(x_train, y_train, x_test, y_test, x_val, y_val):
    keepGoing = True
    for entry in os.scandir('.\\image_files\\thatOneTestFile'):#normally this would be .\\image_files\\images, this is for testing
        filePath = os.fsdecode(entry)
        filename = filePath.rsplit('\\', 1)[-1]
        keepGoing = True
        #zijn dit de goede folders? "if filename in x_train" werkt niet
        for i in x_train:
            if filename == i :
                os.rename(entry.path, ".\\sortedImages\\x_train\\%s" % (filename,))
                keepGoing = False
                break
        if keepGoing == False:
            continue
        for i in y_train:
            if filename == i:
                os.rename(entry.path, ".\\sortedImages\\y_train\\%s" % (filename,))
                break
        if keepGoing == False:
            continue    
        for i in x_test:
            if filename == i:
                os.rename(entry.path, ".\\sortedImages\\x_test\\%s" % (filename,))
                break   
        if keepGoing == False:
            continue    
        for i in y_test:
            if filename == i:
                os.rename(entry.path, ".\\sortedImages\\y_test\\%s" % (filename,))
                break
        if keepGoing == False:
            continue    
        for i in x_val:
            if filename == i:
                os.rename(entry.path, ".\\sortedImages\\x_val\\%s" % (filename,))
                break
        if keepGoing == False:
            continue    
        for i in y_val:
            if filename == i:
                os.rename(entry.path, ".\\sortedImages\\y_val\\%s" % (filename,))
                break  
