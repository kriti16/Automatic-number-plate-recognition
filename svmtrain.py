from sklearn import svm
from array import array as pyarray
import os, struct
from numpy import *
import cv2
import re
from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import AdaBoostClassifier as ab
from sklearn.tree import DecisionTreeClassifier as dt
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l

N = 440
fulldata = zeros((N, 63,260), dtype=uint16)
# hog_array = zeros((N,63*260),dtype=uint16)
labels = loadtxt('labels.txt')
ppc = (8,8)
cpb = (3,3)
dummy = hog(fulldata[0],pixels_per_cell=ppc,cells_per_block=cpb)
length = len(dummy)
hog_array = zeros((N,length),dtype=float64)
i = 0
for root,dir,files in os.walk("svmtrain/"):
    for image_path in sort_nicely(files):
        loc = 'svmtrain/'+str(image_path)
        # print loc
        fulldata[i] = cv2.imread(loc,0)
        # hog_array[i] = fulldata[i].reshape(1,63*260)
        hog_array[i] = hog(fulldata[i],pixels_per_cell=ppc,cells_per_block=cpb)
        i+=1
print hog_array.shape
# X_train = hog_array[0:400]
# Y_train = labels[0:400]
# X_val = hog_array[400:440]
# Y_val = labels[400:440]
# param_grid = [{C:[1]}
clf = rf(n_estimators=500)
clf.fit(hog_array,labels)
joblib.dump(clf, 'model/model.pkl')
# kf = StratifiedKFold(labels,n_folds=5,shuffle=True)
# mean = 0
# for train,test in kf:
# 	clf.fit(hog_array[train],labels[train])
# 	score = clf.score(hog_array[test],labels[test])
# 	print score
# 	mean = mean + score/5
# print "mean = %f" %(mean)
# plt.imshow(temp, interpolation='nearest')#,cmap=plt.get_cmap('gray'))
# plt.show()

