
# coding: utf-8

# ## Load MNIST on Python 3.x

# In[ ]:


import pickle
import gzip

from sklearn.metrics import confusion_matrix


# In[ ]:


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()


# In[ ]:


training_data


# ## Load USPS on Python 3.x

# In[ ]:


from PIL import Image
import os
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.sparse


# In[ ]:


USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)


# ### Loading data

# In[ ]:


train_feat = training_data[0]
train_tar = training_data[1]
val_feat = validation_data[0]
val_tar = validation_data[1]
test_feat = test_data[0]
test_tar = test_data[1]


# In[ ]:


print(len(training_data[0][1,:]))#each image is of 28*28 --> so 784 features for each image(i.e., for each sample). 1 here represents the first image
print(len(training_data[1])) #training_data[1] is the ACTUAL target value for each image and this is also 50,000


# # LOGISTIC REGRESSION

# In[ ]:


# Performing one-hot encoding on the target values (i.e., 10 classes)
def onehot(tar):
    n = tar.shape[0]
    oh = scipy.sparse.csr_matrix((np.ones(n), (tar, np.array(range(n)))))
    oh = np.array(oh.todense()).T
    return oh

def softmax(x):
    x -= np.max(x)
    softmax = (np.exp(x).T / np.sum(np.exp(x), axis = 1)).T
    return softmax

#finding probabilities and predictions given a set of input data
def probsandpreds(features):
    probabilities = softmax(np.dot(features, w)) #a vector of 10 probabilities corresponding to each class
    predictions = np.argmax(probabilities, axis = 1) #maximum probability as the class
    return probabilities, predictions

def Loss(w, feat, tar, la):
    n = feat.shape[0]
    tar_oh = onehot(tar)
    Y = np.dot(feat, w) #predicting target using linear regression 
    probs = softmax(Y)
    loss = (-1/n)  * np.sum(tar_oh * np.log(probs)) + ((la/2) * np.sum(w*w))
    gradient = (-1/n) * np.dot(feat.T, (tar_oh - probs)) + la*w
    return loss, gradient

#Accuracy
def GetAccuracy(x, y):
    probs, preds = probsandpreds(x)
    accuracy = sum(preds == y)/(float(len(y)))
    return accuracy


# In[ ]:


w = np.zeros([train_feat.shape[1], len(np.unique(train_tar))]) #w will be a matrix from each feature to all the output classes, so 784x10
la = 0.0001
learningRate = 0.1
losses = []
x1 = 0
xn = 256

for i in range(x1, xn):
    loss, gradient = Loss(w, train_feat, train_tar, la)
    losses.append(loss)
    w = w - (learningRate * gradient)


# In[ ]:


print ('---------- Logistic Regression using Stochastic Gradient Descent --------------------')
print("Lambda = " + str(la/np.subtract(xn, x1))) # lambda is La/no. of samples
print("eta = " + str(learningRate))
print("Validation Accuracy = " + str(GetAccuracy(val_feat, val_tar)*100))
print("Testing Accuracy = " + str(GetAccuracy(test_feat, test_tar)*100))
print("USPS Accuracy = " + str(GetAccuracy(USPSMat, USPSTar)*100))


# In[ ]:


print('########################### CONFUSION MATRICES FOR LOGISTIC REGRESSION ###########################')
probs_val, preds_val = probsandpreds(val_feat)
print("\nConfusion Matrix of Validation Data: \n\n" + str(confusion_matrix(val_tar, preds_val)))
probs_t, preds_t = probsandpreds(test_feat)
print("\nConfusion Matrix of Testing Data: \n\n" + str(confusion_matrix(test_tar, preds_t)))
probs_usps, preds_usps = probsandpreds(USPSMat)
print("\n Confusion Matrix of USPS Data: \n\n" + str(confusion_matrix(USPSTar, preds_usps)))


# # SUPPORT VECTOR MACHINE (SVM)

# In[ ]:


from sklearn import svm

C = 0.1
clf = svm.SVC(kernel='linear', C=C)
clf.fit(train_feat, train_tar)


# In[ ]:


from sklearn.metrics import accuracy_score

# Getting Validation dataset accuracy
val_pred_svm = clf.predict(val_feat)
acc_val_svm = accuracy_score(val_tar, val_pred_svm)

# Getting Testing dataset accuracy
test_pred_svm = clf.predict(test_feat)
acc_test_svm = accuracy_score(test_tar, test_pred_svm)

# Getting USPS dataset Accuracy
usps_pred_svm = clf.predict(USPSMat)
usps_acc_svm = accuracy_score(USPSTar, usps_pred_svm)


# In[38]:


print ('---------- Support Vector Machine (SVM) --------------------')
print("Regularization Parameter/Penalty(C) = " + str(C))
print("SVM Validation Accuracy: ", acc_val_svm*100)
print("SVM Test Accuracy: ", acc_test_svm*100)
print("SVM USPS Accuracy: ", usps_acc_svm*100)


# In[41]:


print('########################### CONFUSION MATRICES FOR SVM ###########################')
print("\nConfusion Matrix of Validation Data: \n\n" + str(confusion_matrix(val_tar, val_pred_svm)))
print("\nConfusion Matrix of Testing Data: \n\n" + str(confusion_matrix(test_tar, test_pred_svm)))
print("\n Confusion Matrix of USPS Data: \n\n" + str(confusion_matrix(USPSTar, usps_pred_svm)))


# # RANDOM FOREST

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

estimators = 300
clf_rf = RandomForestClassifier(n_estimators = estimators, criterion = 'entropy', random_state = 123) # using ENTROPY to measure the split
clf_rf.fit(train_feat, train_tar) #training the model


# In[ ]:


from sklearn.metrics import accuracy_score

# Getting Validation dataset accuracy
val_pred_rf = clf_rf.predict(val_feat)
acc_val_rf = accuracy_score(val_tar, val_pred_rf)

# Getting Testing dataset accuracy
test_pred_rf = clf_rf.predict(test_feat)
acc_test_rf = accuracy_score(test_tar, test_pred_rf)

# Getting USPS dataset Accuracy
usps_pred_rf = clf_rf.predict(USPSMat)
usps_acc = accuracy_score(USPSTar, usps_pred_rf)

print ('------------- Random Forest --------------------')
print("Number of Trees in the forest = " + str(estimators))
print("Random Forrest Validation Accuracy: ", acc_val_rf*100)
print("Random Forrest Test Accuracy: ", acc_test_rf*100)
print("Random Forrest USPS Accuracy: ", usps_acc*100)


# In[ ]:


print('########################### CONFUSION MATRICES FOR RANDOM FOREST CLASSIFIER ###########################')
print("\nConfusion Matrix of Validation Data: \n\n" + str(confusion_matrix(val_tar, val_pred_rf)))
print("\nConfusion Matrix of Testing Data: \n\n" + str(confusion_matrix(test_tar, test_pred_rf)))
print("\n Confusion Matrix of USPS Data: \n\n" + str(confusion_matrix(USPSTar, usps_pred_rf)))


# # NEURAL NETWORK

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop
import keras.backend as K

# Converting Target Variables
train_tar_cat = keras.utils.to_categorical(train_tar, 10)
val_tar_cat = keras.utils.to_categorical(val_tar, 10)
test_tar_cat = keras.utils.to_categorical(test_tar, 10)
usps_tar_cat = keras.utils.to_categorical(USPSTar, 10)

# Converting a list of USPS arrays into single array
usps_feat = np.vstack(USPSMat)

K.clear_session()
model = Sequential()
model.add(Dense(units = 100, input_dim = 784, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 100, input_dim = 100, activation='relu')) #USE 32 FOR IDEAL
model.add(Dropout(0.1))
model.add(Dense(10, activation = 'softmax'))

model.compile('adam', 'categorical_crossentropy', metrics = ['accuracy'])

h = model.fit(train_feat, train_tar_cat,
             batch_size = 32,
             epochs = 100,
             verbose = 0)


# In[ ]:


val_pred_nn = model.predict_classes(val_feat)
test_pred_nn = model.predict_classes(test_feat)
usps_pred_nn = model.predict_classes(usps_feat)

loss_val, accuracy_val = model.evaluate(val_feat, val_tar_cat, verbose = False)
print("\nValidation CrossEntropy: ", loss_val)
print("\nValidation Accuracy: ", accuracy_val*100)

loss_test, accuracy_test = model.evaluate(test_feat, test_tar_cat, verbose = False)
print("\nTest CrossEntropy: ", loss_test)
print("\nTest Accuracy: ", (round(accuracy_test*100, 4)))

loss_usps, accuracy_usps = model.evaluate(usps_feat, usps_tar_cat, verbose = False)
print("\nUSPS CrossEntropy: ", loss_usps)
print("\nUSPS Accuracy: ", round(accuracy_usps*100, 4))


# In[ ]:


print('########################### CONFUSION MATRICES FOR NEURAL NETWORKS ###########################')
print("\nConfusion Matrix of Validation Data: \n\n" + str(confusion_matrix(val_tar, val_pred_nn)))
print("\nConfusion Matrix of Testing Data: \n\n" + str(confusion_matrix(test_tar, test_pred_nn)))
print("\n Confusion Matrix of USPS Data: \n\n" + str(confusion_matrix(USPSTar, usps_pred_nn)))


# 
# # ENSEMBLE CLASSIFIER

# In[39]:


from statistics import mode

val_final_pred = np.array([])
test_final_pred = np.array([])
usps_final_pred = np.array([])

for i in range(0, len(val_feat)):
    try:
        val_final_pred = np.append(val_final_pred, mode([preds_val[i], val_pred_svm[i], val_pred_rf[i], val_pred_nn[i]]))
    except:
        val_final_pred = np.append(val_final_pred, max([preds_t[i], val_pred_svm[i], val_pred_rf[i], val_pred_nn[i]]))

for i in range(0, len(test_feat)):
    try:
        test_final_pred = np.append(test_final_pred, mode([preds_t[i], test_pred_svm[i], test_pred_rf[i], test_pred_nn[i]]))
    except:
        test_final_pred = np.append(test_final_pred, max([preds_t[i], test_pred_svm[i], test_pred_rf[i], test_pred_nn[i]]))
        
for i in range(0, len(USPSTar)):
    try:
        usps_final_pred = np.append(usps_final_pred, mode([preds_usps[i], usps_pred_svm[i], usps_pred_rf[i], usps_pred_nn[i]]))
    except:
        usps_final_pred = np.append(usps_final_pred, max([preds_usps[i], usps_pred_svm[i], usps_pred_rf[i], usps_pred_nn[i]]))
        
print ('------------- Ensemble Classifier --------------------')
print("Ensemble Classifier Validation Accuracy: ", accuracy_score(val_tar, val_final_pred)*100)
print("Ensemble Classifier Test Accuracy: ", accuracy_score(test_tar, test_final_pred)*100)
print("Ensemble Classifier USPS Accuracy: ", accuracy_score(USPSTar, usps_final_pred)*100)


# In[40]:


print('########################### CONFUSION MATRICES FOR ENSEMBLE CLASSIFIER ###########################')
print("\nConfusion Matrix of Validation Data: \n\n" + str(confusion_matrix(val_tar, val_final_pred)))
print("\nConfusion Matrix of Testing Data: \n\n" + str(confusion_matrix(test_tar, test_final_pred)))
print("\n Confusion Matrix of USPS Data: \n\n" + str(confusion_matrix(USPSTar, usps_final_pred)))

