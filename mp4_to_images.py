import sys
import argparse
import os
from os.path import isfile, join
import numpy as np

import cv2
print(cv2.__version__)

#Generates Images from .mp4 video files in a directory
def extractImages(pathIn, pathOut):
    """
    
    Arguments: 
    pathIn  -- Path of the directory containing mp4 files
    pathOut -- Path of target directory for saving images
    
    """
    #list = [pathIn, f]
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn,f))]
    files.sort(key = lambda x: int(x[5:-4]))
    for i in range(len(files)):
        count = 0
        vidcap = cv2.VideoCapture(pathIn + files[i])
        success,image = vidcap.read()
        #success = True
        fi = int(files[i][-8:-4])
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
            success,image = vidcap.read()
            print ('Read a new frame: ', success)
        #pi = int(pathIn[-8:-4])
            if success:         
                cv2.imwrite(pathOut + "frame%d_%d.jpg" % (fi,count), image)     # save frame as JPEG file
                count = count + 1
            
if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path to images")
    args = a.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut)