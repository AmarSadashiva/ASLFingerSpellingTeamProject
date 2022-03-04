import cv2
import mediapipe as mp
import numpy as np
import HandPositionDetector as hpd
import HandCropper as hc
import createDataset as cd
import getData as gd
import tensorflow as tf
import cnnTrain as cnn
import keyframe as kf
import os

keras = tf.keras
load_model = keras.models.load_model
Model = keras.models.Model
BASE = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE, 'cnn_new_model.h5'))
model.summary()

# Order of execution
# 1. getData.py - To get hand co-ordinates and to crop the videos.
# 2. createDataset.py - To create dataset (dataset.npy) out of the cropped frames.
# 3. cnnTrain - To retrain the model and to save the new model (cnn_new_model.h5).
# 4. keyframe.py - To predict words and to generate predicted.csv file.

# get data (Executes HandPositionDetector, HandCropper modules)
gd.getData()

# creating dataset - dataset.npy
cd.create_train_data()

# To retrain CNN model and to report accuracy and F1 score
cnn.trainCNN()

# To predict words
kf.detectWords()






# # List to map the predicted sign
# #           0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26
# out_label=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','SPACE']


# img = cv2.imread(os.path.join('data/B/130.jpg'), cv2.IMREAD_GRAYSCALE)
# img = np.array(img) / 255.0
# cv2.imshow("image", img)
# cv2.waitKey(8000)
# img = img.reshape(1, 200, 200, 1)
# print("Actual letter: B")
# pred = np.argmax(model.predict(img))
# print("predicted letter: ",out_label[pred])





