import sys
import json
import socket
reload(sys)
sys.setdefaultencoding('gb18030') 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from robust_pca import RobustPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def concatMat(Mat, x, y):
    originalX = list(Mat.shape)[0]
    originalY = list(Mat.shape)[1]
    a = np.zeros((originalX, y - originalY))
    b = np.zeros((x - originalX, y))
    Mat = np.hstack((Mat, a))
    Mat = np.vstack((Mat, b))
    return Mat

def split(dataX, dataY, num, total):
    dataX_all = []
    dataY_all = []
    X_train_all = []
    y_train_all = []
    avg = total / num
    acc_rf = []
    acc_svm = []
    acc_logic = []
    
    for i in range(num):
        X_train_ = dataX[i * avg: (i + 1) * avg]
        y_train_ = dataY[i * avg: (i + 1) * avg]
        dataX_all.append(X_train_)
        dataY_all.append(y_train_)
        
        X_train_all.append(X_train_)
        y_train_all.append(y_train_)
        print X_train_.shape

        clf = RandomForestClassifier(n_jobs=2)
        clf.fit(X_train_, y_train_)
        y_preds = clf.predict(X_train_)

        counts = 0

        for i in range(len(y_preds)):
            if y_preds[i] == y_train_.tolist()[i]:
                counts += 1

        acc_ = counts * 1.0 / len(y_preds)
        acc_rf.append(acc_)
        print acc_

    average_accuracy_rf = sum(acc_rf) / num
    print "Distributed Random Forest Accuracy: ", average_accuracy_rf

    for i in range(num):
        X_train_ = dataX[i * avg: (i + 1) * avg]
        y_train_ = dataY[i * avg: (i + 1) * avg]

        clf = LinearSVC()
        clf.fit(X_train_, y_train_)
        y_preds = clf.predict(X_train_)

        counts = 0

        for i in range(len(y_preds)):
            if y_preds[i] == y_train_.tolist()[i]:
                counts += 1

        acc_ = counts * 1.0 / len(y_preds)
        acc_svm.append(acc_)
        print acc_

    average_accuracy_svm = sum(acc_svm) / num
    print "Distributed SVM Accuracy: ", average_accuracy_svm

    for i in range(num):
        X_train_ = dataX[i * avg: (i + 1) * avg]
        y_train_ = dataY[i * avg: (i + 1) * avg]

        lr_model = LogisticRegression()
        lr_model.fit(X_train_, y_train_)

        acc_ = lr_model.score(X_train_, y_train_)
        acc_logic.append(acc_)
        print acc_

    average_accuracy_logic = sum(acc_logic) / num
    print "Distributed LogicRegression Accuracy: ", average_accuracy_logic

total = 120000

data = pd.read_csv("../dataset/NSL_KDD-master/KDDTrain+.csv", iterator = True)
data = data.get_chunk(total)
dataX1 = data.copy().drop(['label'],axis=1)
dataX2 = dataX1.copy().drop(['difficulty'], axis = 1)
dataX = pd.get_dummies(dataX2)
dataY = data['label'].copy()

for i in range(0, len(dataY)):
    if dataY[i] == 'normal':
        dataY[i] = 0
    else:
        dataY[i] = 1
dataY = pd.to_numeric(dataY)

featuresToScale = dataX.columns
sX = StandardScaler(copy=True)
dataX.loc[:,featuresToScale] = sX.fit_transform(dataX[featuresToScale])

X_train, X_test, y_train, y_test = \
    train_test_split(dataX, dataY, test_size=0.33, \
                    random_state=2018, stratify=dataY)
print X_train.shape

split(X_train, y_train, 4, X_train.shape[0])


clf = RandomForestClassifier(n_jobs=2)
clf.fit(X_train, y_train)
y_preds = clf.predict(X_train)

counts = 0

for i in range(len(y_preds)):
    if y_preds[i] == y_train.tolist()[i]:
        counts += 1

acc = counts * 1.0 / len(y_preds)
print "Centralized Random Forest Accuracy: ", acc

clf = LinearSVC()
clf.fit(X_train, y_train)
y_preds = clf.predict(X_train)

counts = 0
for i in range(len(y_preds)):
    if y_preds[i] == y_train.tolist()[i]:
        counts += 1

acc = counts * 1.0 / len(y_preds)
print "Centralized SVM Accuracy: ", acc

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

print "Centralized LogicRegression Accuracy: ", lr_model.score(X_train, y_train)