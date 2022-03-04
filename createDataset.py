import cv2                 
import numpy as np         
import os                  
from random import shuffle 

# Creates dataset.npy file
# Associates each frame with its label
# Shuffles data and add it to dataset

path='data'
def create_train_data():
    dataset = []
    for (dirpath,dirnames,filenames) in os.walk(path):
        for dirname in dirnames:
            for(direcpath,direcnames,files) in os.walk(path+"/"+dirname):
                for file in files:
                    path1 =path+"/"+dirname+'/'+file
                    img=cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
                    img = np.array(img)/255.0
                    img = img.reshape(1, 200, 200, 1)
                    # Adding label to each frame base on the folder in which it is present
                    if dirname=='Space':
                        dataset.append([np.array(img, dtype=np.float32),26])
                    else:
                        dataset.append([np.array(img, dtype=np.float32),ord(dirname)-65])
    shuffle(dataset)
    np.save('dataset.npy', dataset)
    return dataset

#  create_train_data()