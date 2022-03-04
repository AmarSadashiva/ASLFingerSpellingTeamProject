import cv2                
import numpy as np         
import os

from numpy.lib.function_base import average     
import tensorflow as tf             
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# Importing pre trained CNN model
# Loadind data from dataset.npy
# Preprocessing data
# Retraining model with new data and saving the new model.
# Reporting accuracy and F1 score

def trainCNN():
    keras = tf.keras
    load_model = keras.models.load_model
    Model = keras.models.Model
    BASE = os.path.dirname(os.path.abspath(__file__))
    nClasses = 27

    model = load_model(os.path.join(BASE, 'cnn_model.h5'))
    model.summary()

    # Loading data to re train the model
    data = np.load('dataset.npy',allow_pickle=True)

    # Splitting data as train set and test set
    trainData = data[:int(0.8*len(data))]
    testData = data[int(0.8*len(data)):]
    X_train = np.array([i[0] for i in trainData]).reshape(-1,200,200,1)
    y_train = [i[1] for i in trainData]
    X_test = np.array([i[0] for i in testData]).reshape(-1,200,200,1)
    y_test = [i[1] for i in testData]


    yTrain = tf.keras.utils.to_categorical(y_train, nClasses)
    yTest = tf.keras.utils.to_categorical(y_test, nClasses)
    yTrain = yTrain.reshape(-1,27)
    yTest = yTest.reshape(-1,27)

    # Compiling, training and saving the model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X_train, yTrain, epochs=2, validation_data=(X_test,  yTest), validation_steps=500)
    model.save('cnn_new_model.h5')

    testEval = model.evaluate(X_test, yTest)
    print('Test accuarcy: %0.4f%%' % (testEval[1] * 100))

    yhat_classes = np.argmax(model.predict(X_test), axis = -1)
    yTest_classes = np.argmax(yTest, axis=-1)
    f1 = f1_score(yTest_classes, yhat_classes, average='weighted')
    print("F1 score:",f1)

# trainCNN()
