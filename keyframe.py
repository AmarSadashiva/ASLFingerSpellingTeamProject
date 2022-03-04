import cv2 
import os
import tensorflow as tf
import numpy as np
import HandPositionDetector as hpd
import glob, re, csv
import HandCropper as hc
# from keras.models import load_model
keras = tf.keras
load_model = keras.models.load_model

  
def getWordLength(name):
    found = ''
    m = re.search('(.+?).mp4', name)
    if m:
        found = m.group(1)
    return len(found)


def detectNCrop(path):
    imgList = glob.glob(os.path.join(path, '*.jpg'))
    imgList.sort()
    for f in imgList:
        detector = hpd.handDetector()
        img = cv2.imread(f)
        img = detector.findHands(img)
        xMin, xMax, yMin, yMax = detector.findPosition(img)
        img = cv2.imread(f)
        crop_img = img[max(yMin-30,0):min(yMax+30,1920), max(xMin-40,0):min(xMax+30,1080), :]
        cropped_resized_img = cv2.resize(crop_img, (200, 200))
        cv2.imshow("Image", crop_img)
        # cv2.waitKey(0)
        cv2.imwrite(f, cropped_resized_img)


def predictWords(word):
    wordFileList = glob.glob(os.path.join(word, '*.jpg'))
    wordFileList.sort(key=lambda f: int(re.sub('\D', '', f))) #sorted in order
    print(wordFileList)
    predictedWord = []
    for i in wordFileList:   
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        cv2.imshow("image", img)
        img = np.array(img) / 255.0
        img = img.reshape(1, 200, 200, 1)
        pred = np.argmax(model.predict(img))
        predictedWord.append(chr(65+pred))
    return predictedWord

def detectWords():
    wordPathList = []
    for (dirpath,dirnames,filenames) in os.walk('words'):
        for filename in filenames:
            newFolder = os.path.join(dirpath, filename[:len(filename)-4])
            print(newFolder)
            if not os.path.exists(newFolder):
                os.mkdir(newFolder)
            wordPathList.append(newFolder)
            wordVideo = os.path.join(dirpath, filename)
            print(wordVideo)
            cap = cv2.VideoCapture(wordVideo)
            #divide video into segments equal to number of letters in the word
            wordLength = getWordLength(filename)
            totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(totalFrames)
            size = totalFrames / wordLength
            frames = []
            ctr = 0
            # Extracting the middle frame of each segment
            while ctr != wordLength:
                frames.append(int((0.5+ctr)*size))
                ctr += 1
            print("Selecting frames: ", frames)
            for i in frames:
                cap.set(1,i)
                ret, frame = cap.read() # Read the frame
                cv2.imwrite(newFolder+"/"+str(i)+'.jpg', frame)
            # run mediapipe on extracted frame to get the hand co-ordinates
            detectNCrop(newFolder)


    #start prediction of words
    model = load_model('cnn_new_model.h5')
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    for word in wordPathList:
        a = predictWords(word)
        actualWord = word.split('/', 2)[2]
        wordJoined = ''
        for i in a:
            wordJoined = wordJoined + i
        with open('predicted.csv', mode='a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([actualWord, wordJoined])
        print(actualWord, wordJoined)

# detectwords()
