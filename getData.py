import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import HandPositionDetector as hpd
import HandCropper as hc


# For each video, to get crop coordinates, obtaining cropped videos and adding every cropped frame to data folder.

videoFolder = os.path.join('TrainVideo')
croppedVideosFolder = os.path.join('CroppedVideos/')
dataFolder = os.path.join('data')

def getData():
    for (dirpath,dirnames,filenames) in os.walk(videoFolder):
        for dirname in dirnames:
            videosPath = os.path.join(videoFolder,dirname,'*.mp4')
            alphabetVideoFilesList = glob.glob(videosPath)
            alphabetVideoFilesList.sort()
            count = 0
            for letterSignVideo in alphabetVideoFilesList:
                fileName = os.path.basename(letterSignVideo)
                outvideoFolder = os.path.join(croppedVideosFolder, dirname)
                if not os.path.exists(outvideoFolder):
                    os.makedirs(outvideoFolder)
                outDataFolder = os.path.join(dataFolder, dirname)
                if not os.path.exists(outDataFolder):
                    os.makedirs(outDataFolder)
                initialFrameCount = count + 1
                letterFolderName = dirname
                cap = cv2.VideoCapture(letterSignVideo)
                detector = hpd.handDetector()
                xMinList = []
                xMaxList = []
                yMinList = []
                yMaxList = []
                success, img = cap.read()
                while success:
                    success, img = cap.read()
                    if success:
                        img = detector.findHands(img)
                        # Getting hand coordinates
                        xMin, xMax, yMin, yMax = detector.findPosition(img)
                        xMinList.append(xMin)
                        xMaxList.append(xMax)
                        yMinList.append(yMin)
                        yMaxList.append(yMax)
                        cv2.imshow("Image", img)
                        cv2.waitKey(1)
                cropper = hc.HandCropper(int(min(xMinList)), int(max(xMaxList)), int(min(yMinList)), int(max(yMaxList)), letterSignVideo, os.path.join(outvideoFolder, fileName), os.path.join(outDataFolder), initialFrameCount)
                count = cropper.cropHands()

# getData()