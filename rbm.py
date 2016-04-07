
# coding: utf-8

# In[9]:

import os
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from pandas import DataFrame
from sklearn.neural_network import BernoulliRBM
# from sklearn.neural_network import MLPClassifier
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from scipy.ndimage import convolve
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import linear_model
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import cv2
import numpy as np
first = True
ocr_map = {}


# In[10]:

def read_files(path): 
    for root, dirs, files in os.walk(path):
        for file_names in files:
            file_path = os.path.join(root, file_names)
            #print file_path
            #f = open(file_path, 'r')
            raw_image=cv2.imread(file_path)
            raw_image=cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
            
            #resize the image into 5 cols(width) and 10 rows(height)
            raw_image=cv2.resize(raw_image,(40,40), interpolation=cv2.INTER_AREA)
            #Do a hard thresholding.
            _,th2=cv2.threshold(raw_image, 70, 255, cv2.THRESH_BINARY)
            
            #generate features
            th2 = hog(raw_image)
            sample=th2.flatten()
            #concatenate these features together
            # feature=np.concatenate([horz_hist, vert_hist, sample])
            yield sample


# In[11]:

def get_data(part_number):
    rows = []
    if(part_number<10):
        path_name = "/home/pramod/sem6/mlt/project/English/Fnt/Sample00" + str(part_number)
    else:
        path_name = "/home/pramod/sem6/mlt/project/English/Fnt/Sample0" + str(part_number)
    for row in read_files(path_name):
        rows.append({'text':row,'label':ocr_map[part_number]})
    return rows


# In[12]:

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


# In[15]:

def train_model():
    global ocr_map
    count = 1
    a=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    for char in a:
        ocr_map[count] = char
        count += 1
    data_frames = []
    X_train = []
    Y_train = []
    X_test = [] 
    Y_test = []
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0,verbose=True)
    # classifier = Pipeline(steps=[ ('logistic', LinearSVC())])
    classifier = Pipeline(steps=[('rbm',rbm),('logistic',logistic)])
    for i in range(1,count):
        l = get_data(i)
        print len(l)
        for data in range(0, 900):
            X_train.append(l[data]['text'])
            Y_train.append(l[data]['label'])
        for data in range(900, len(l)):
            X_test.append(l[data]['text'])
            Y_test.append(l[data]['label'])
    

        
    # X_train, Y_train = nudge_dataset(X_train, Y_train)
    # X_test, Y_test = nudge_dataset(X_test, Y_test)
    X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)  # 0-1 scaling
    X_test = (X_test - np.min(X_test, 0)) / (np.max(X_test, 0) + 0.0001)  # 0-1 scaling
    
    print X_train.shape,X_test.shape
    # skf = StratifiedKFold(Y, n_folds=2)
    # joblib.dump(X_train, 'X_train.pkl',compress=3)
    # joblib.dump(Y_train, 'Y_train.pkl',compress=3)
    # joblib.dump(X_test, 'X_test.pkl',compress=3)
    # joblib.dump(Y_test, 'Y_test.pkl',compress=3)
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    rbm.n_components = 100
    # logistic.C = 6000.0
    classifier.fit(X_train, Y_train)
    f = open("ocr_results.txt",'w')
    answers = classifier.predict(X_test)
    print confusion_matrix(Y_test, answers)
    score_data = accuracy_score(Y_test, answers)
    print score_data
    f.write(str(score_data))
    f.close()
    # joblib.dump(classifier, 'rfmodel.pkl',compress=3)
train_model()
        


# In[ ]:



