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

total = 24000

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

whiten = False
random_state = 2018
rpca = RobustPCA(lam = 0.0155).fit(X_train)
X_train_PCA = rpca.transform(X_train)
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train.index)

X_train_PCA_inverse = rpca.inverse_transform(X_train_PCA)
X_train_PCA_inverse = pd.DataFrame(data=X_train_PCA_inverse, \
                                   index=X_train.index)

clf = RandomForestClassifier(n_jobs=2)
clf.fit(X_train_PCA, y_train)
y_preds = clf.predict(X_train_PCA)

print y_preds
counts = 0

for i in range(len(y_preds)):
    if y_preds[i] == y_train.tolist()[i]:
        counts += 1

acc = counts * 1.0 / len(y_preds)
print "Centralized Random Forest Accuracy: ", acc

clf = LinearSVC()
clf.fit(X_train_PCA, y_train)
y_preds = clf.predict(X_train_PCA)

counts = 0
for i in range(len(y_preds)):
    if y_preds[i] == y_train.tolist()[i]:
        counts += 1

acc = counts * 1.0 / len(y_preds)
print "Centralized SVM Accuracy: ", acc

lr_model = LogisticRegression()
lr_model.fit(X_train_PCA, y_train)

print "Centralized LogicRegression Accuracy: ", lr_model.score(X_train_PCA, y_train)
