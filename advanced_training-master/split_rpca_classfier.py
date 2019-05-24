import sys
import json
import socket
import time
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
from sklearn import svm
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

def concatMat(Mat, x, y):
    originalX = list(Mat.shape)[0]
    originalY = list(Mat.shape)[1]
    a = np.zeros((originalX, y - originalY))
    b = np.zeros((x - originalX, y))
    Mat = np.hstack((Mat, a))
    Mat = np.vstack((Mat, b))
    return Mat

def split(dataX, dataY, num, total, lam_user):
    dataX_all = []
    dataY_all = []
    X_train_all = []
    y_train_all = []
    avg = total / num
    components_all = []
    rpca_all = []
    
    t1 = time.time()
    for i in range(num):
        X_train_ = dataX[i * avg: (i + 1) * avg]
        y_train_ = dataY[i * avg: (i + 1) * avg]
        dataX_all.append(X_train_)
        dataY_all.append(y_train_)
        
        X_train_all.append(X_train_)
        y_train_all.append(y_train_)
        print X_train_.shape

        rpca = RobustPCA(lam = lam_user).fit(X_train_)
        rpca_all.append(rpca)
        components_ = np.array(rpca.components_)
        components_all.append(components_)

    shape_x = max([component.shape[0] for component in components_all])
    shape_y = max([component.shape[1] for component in components_all])

    for component in components_all:
        component = concatMat(component, shape_x, shape_y)
    
    components_all_stack = components_all[0]
    for i in range(1, len(components_all)):
        components_all_stack = np.vstack((components_all_stack, components_all[i]))
    
    components = RobustPCA(lam_user).fit(components_all_stack).components_
    t2 = time.time()
    t_rpca = t2 - t1

    acc_rf = []
    acc_svm = []
    acc_logic = []
    time_rf = []
    time_svm = []
    time_logic = []
    time_rpca = []
    for i in range(num):
        trpca_1 = time.time()
        X_train_PCA = rpca_all[i].transform(X_train_all[i], components)
        print X_train_PCA.shape
        X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train_all[i].index)

        X_train_PCA_inverse = rpca_all[i].inverse_transform(X_train_PCA, components)
        X_train_PCA_inverse = pd.DataFrame(data=X_train_PCA_inverse, \
                                            index=X_train_all[i].index)
        trpca_2 = time.time()
        time_rpca.append(trpca_2 - trpca_1)

        time_rf_1 = time.time()
        clf = RandomForestClassifier(n_jobs=2)
        clf.fit(X_train_PCA, y_train_all[i])
        y_preds = clf.predict(X_train_PCA)

        counts = 0

        for j in range(len(y_preds)):
            if y_preds[j] == y_train_all[i].tolist()[j]:
                counts += 1

        acc_rf_ = counts * 1.0 / len(y_preds)
        acc_rf.append(acc_rf_)
        time_rf_2 = time.time()
        time_rf.append(time_rf_2 - time_rf_1)
        
        time_svm_1 = time.time()
        clf = SVC()
        clf.fit(X_train_PCA, y_train_all[i])
        y_preds = clf.predict(X_train_PCA)

        counts = 0
        for j in range(len(y_preds)):
            if y_preds[j] == y_train_all[i].tolist()[j]:
                counts += 1

        acc_svm_ = counts * 1.0 / len(y_preds)
        acc_svm.append(acc_svm_)
        time_svm_2 = time.time()
        time_svm.append(time_svm_2 - time_svm_1)

        time_lr_1 = time.time()
        lr_model = LogisticRegressionCV(multi_class="ovr",fit_intercept=True,Cs=np.logspace(-2,2,20),cv=2,penalty="l2",solver="lbfgs",tol=0.01)
        re = lr_model.fit(X_train_PCA, y_train_all[i])
        acc_logic_ = re.score(X_train_PCA, y_train_all[i])
        acc_logic.append(acc_logic_)
        time_lr_2 = time.time()
        time_logic.append(time_lr_2 - time_lr_1)
    
    average_time_rf = (t_rpca + sum(time_rpca) + sum(time_rf)) / num
    average_time_svm = (t_rpca + sum(time_rpca) + sum(time_svm)) / num
    average_time_lr = (t_rpca + sum(time_rpca) + sum(time_logic)) / num


    average_accuracy_rf = sum(acc_rf) / num
    print "Distributed RPCA+Random Forest Accuracy: ", average_accuracy_rf
    average_accuracy_svm = sum(acc_svm) / num
    print "Distributed RPCA+SVM Accuracy: ", average_accuracy_svm
    average_accuracy_logic = sum(acc_logic) / num
    print "Distributed RPCA+LogicRegression Accuracy: ", average_accuracy_logic
    return average_accuracy_rf, average_accuracy_svm, average_accuracy_logic, average_time_rf, average_time_svm, average_time_lr


total = 120000
node_num = int(sys.argv[1])

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

disrpca_rf_acc = []
disrpca_rf_time = []
central_rf_acc = []
central_rf_time = []

disrpca_svm_acc = []
disrpca_svm_time = []
central_svm_acc = []
central_svm_time = []

disrpca_lr_acc = []
disrpca_lr_time = []
central_lr_acc = []
central_lr_time = []

for random_num in range(2010,2012):
    X_train, X_test, y_train, y_test = \
        train_test_split(dataX, dataY, test_size=0.33, \
                        random_state=random_num, stratify=dataY)
    print X_train.shape
    dis_rf_acc, dis_svm_acc, dis_lr_acc, dis_rf_time, dis_svm_time, dis_lr_time = split(X_train, y_train, node_num, X_train.shape[0], None)
    disrpca_rf_acc.append(dis_rf_acc)
    disrpca_rf_time.append(dis_rf_time)
    disrpca_svm_acc.append(dis_svm_acc)
    disrpca_svm_time.append(dis_svm_time)
    disrpca_lr_acc.append(dis_lr_acc)
    disrpca_lr_time.append(dis_lr_time)

print "Nodes: ",node_num
dis_rf_acc_average = sum(disrpca_rf_acc) / len(disrpca_rf_acc)
print "Average DisRPCA+RandomForest Accuracy: ", dis_rf_acc_average
dis_rf_time_average = sum(disrpca_rf_time) / len(disrpca_rf_time)
print "Average DisRPCA+RandomForest Time: ", dis_rf_time_average

dis_svm_acc_average = sum(disrpca_svm_acc) / len(disrpca_svm_acc)
print "Average DisRPCA+SVM Accuracy: ", dis_svm_acc_average
dis_svm_time_average = sum(disrpca_svm_time) / len(disrpca_svm_time)
print "Average DisRPCA+SVM Time: ", dis_svm_time_average

dis_lr_acc_average = sum(disrpca_lr_acc) / len(disrpca_lr_acc)
print "Average DisRPCA+LogicRegression Accuracy: ", dis_lr_acc_average
dis_lr_time_average = sum(disrpca_lr_time) / len(disrpca_lr_time)
print "Average DisRPCA+LogicRegression Time: ", dis_lr_time_average

f = open(str(node_num)+"nodes_RPCA+Classifier.txt", 'a')
f.write("Average DisRPCA+RandomForest Accuracy: "+str(dis_rf_acc_average)+"\n")
f.write("Average DisRPCA+RandomForest Time: "+str(dis_rf_time_average)+"\n")
f.write("Average DisRPCA+SVM Accuracy: "+str(dis_svm_acc_average)+"\n")
f.write("Average DisRPCA+SVM Time: "+str(dis_svm_time_average)+"\n")
f.write("Average DisRPCA+LogicRegression Accuracy: "+str(dis_lr_acc_average)+"\n")
f.write("Average DisRPCA+LogicRegression Time: "+str(dis_lr_time_average)+"\n")
f.close()