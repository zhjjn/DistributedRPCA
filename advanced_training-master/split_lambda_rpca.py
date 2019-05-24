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

def decision(y_score, alpha):
    y_pred = []
    for score in y_score:
        if score <= alpha:
            y_pred.append(0)
        else:
            y_pred.append(1)
    
    return y_pred

def anomalyScores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF)-np.array(reducedDF))**2, axis=1)
    loss = pd.Series(data=loss,index=originalDF.index)
    loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
    return loss

def plotResults(trueLabels, anomalyScores, returnPreds = False):
    preds = pd.concat([trueLabels, anomalyScores], axis=1)
    preds.columns = ['trueLabel', 'anomalyScore']
    precision, recall, thresholds = \
        precision_recall_curve(preds['trueLabel'],preds['anomalyScore'])
    average_precision = \
        average_precision_score(preds['trueLabel'],preds['anomalyScore'])

    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.title('Precision-Recall curve: Average Precision = \
    {0:0.2f}'.format(average_precision))

    fpr, tpr, thresholds = roc_curve(preds['trueLabel'], \
                                     preds['anomalyScore'])
    

    areaUnderROC = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: \
    Area under the curve = {0:0.2f}'.format(areaUnderROC))
    plt.legend(loc="lower right")

    threshold = [0.0016875]
    accuracy = []
    for alpha in threshold:
        count = 0
        y_pred = decision(preds['anomalyScore'], alpha)
        for i in range(0, len(preds['trueLabel'])):
            if y_pred[i] == preds['trueLabel'].tolist()[i]:
                count += 1
        accuracy.append(count)

    Accuracy = [i * 1.0 / len(trueLabels) for i in accuracy]
    print Accuracy
    '''
    plt.figure()
    plt.plot(threshold, Accuracy, color = 'r', label = 'Accuracy curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    '''
    plt.show()

    if returnPreds==True:
        return preds

def Acc(trueLabels, anomalyScores):
    preds = pd.concat([trueLabels, anomalyScores], axis=1)
    preds.columns = ['trueLabel', 'anomalyScore']
    threshold = [0.0016875]
    accuracy = []
    for alpha in threshold:
        count = 0
        y_pred = decision(preds['anomalyScore'], alpha)
        for i in range(0, len(preds['trueLabel'])):
            if y_pred[i] == preds['trueLabel'].tolist()[i]:
                count += 1
        accuracy.append(count)

    Accuracy = [i * 1.0 / len(trueLabels) for i in accuracy]
    print Accuracy
    return Accuracy[0]

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

    acc = []
    for i in range(num):
        X_train_PCA = rpca_all[i].transform(X_train_all[i], components)
        print X_train_PCA.shape
        X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train_all[i].index)

        X_train_PCA_inverse = rpca_all[i].inverse_transform(X_train_PCA, components)
        X_train_PCA_inverse = pd.DataFrame(data=X_train_PCA_inverse, \
                                            index=X_train_all[i].index)

        anomalyScoresPCA = anomalyScores(X_train_all[i], X_train_PCA_inverse)
        #preds = plotResults(y_train_all[i], anomalyScoresPCA, True)
        Accuracy = Acc(y_train_all[i], anomalyScoresPCA)
        acc.append(Accuracy)

    average_accuracy = sum(acc) / num
    print "Distributed RPCA Accuracy: ", average_accuracy
    return average_accuracy

total = 120000
Lambda = float(sys.argv[1])

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

disrpca_acc_4 = []
disrpca_time_4 = []
disrpca_acc_8 = []
disrpca_time_8 = []
disrpca_acc_12 = []
disrpca_time_12 = []
central_acc = []
central_time = []

for random_num in range(2018, 2020):

    X_train, X_test, y_train, y_test = \
        train_test_split(dataX, dataY, test_size=0.33, \
                        random_state=random_num, stratify=dataY)
    print X_test.shape

    t1 = time.time()
    dis_accuracy = split(X_test, y_test, 4, X_test.shape[0], Lambda)
    disrpca_acc_4.append(dis_accuracy)
    t2 = time.time()

    dis_time = (t2 - t1) / 4
    disrpca_time_4.append(dis_time)

    t1 = time.time()
    dis_accuracy = split(X_test, y_test, 8, X_test.shape[0], Lambda)
    disrpca_acc_8.append(dis_accuracy)
    t2 = time.time()

    dis_time = (t2 - t1) / 8
    disrpca_time_8.append(dis_time)

    t1 = time.time()
    dis_accuracy = split(X_test, y_test, 12, X_test.shape[0], Lambda)
    disrpca_acc_12.append(dis_accuracy)
    t2 = time.time()

    dis_time = (t2 - t1) / 12
    disrpca_time_12.append(dis_time)

    whiten = False
    random_state = random_num
    t3 = time.time()
    rpca = RobustPCA(lam = Lambda).fit(X_test)
    X_test_PCA = rpca.transform(X_test)
    X_test_PCA = pd.DataFrame(data=X_test_PCA, index=X_test.index)

    X_test_PCA_inverse = rpca.inverse_transform(X_test_PCA)
    X_test_PCA_inverse = pd.DataFrame(data=X_test_PCA_inverse, \
                                    index=X_test.index)

    anomalyScoresPCA = anomalyScores(X_test, X_test_PCA_inverse)
    #preds = plotResults(y_train, anomalyScoresPCA, True)
    Accuracy = Acc(y_test, anomalyScoresPCA)
    central_acc.append(Accuracy)
    t4 = time.time()
    central_time.append(t4 - t3)

print "Lambda: ", Lambda
dis_acc_average_4 = sum(disrpca_acc_4) / len(disrpca_acc_4)
print "Average DisRPCA Accuracy, 4 Nodes: ", dis_acc_average_4
dis_time_average_4 = sum(disrpca_time_4) / len(disrpca_time_4)
print "Average DisRPCA Time, 4 Nodes: ", dis_time_average_4

dis_acc_average_8 = sum(disrpca_acc_8) / len(disrpca_acc_8)
print "Average DisRPCA Accuracy, 8 Nodes: ", dis_acc_average_8
dis_time_average_8 = sum(disrpca_time_8) / len(disrpca_time_8)
print "Average DisRPCA Time, 8 Nodes: ", dis_time_average_8

dis_acc_average_12 = sum(disrpca_acc_12) / len(disrpca_acc_12)
print "Average DisRPCA Accuracy, 12 Nodes: ", dis_acc_average_12
dis_time_average_12 = sum(disrpca_time_12) / len(disrpca_time_12)
print "Average DisRPCA Time, 12 Nodes: ", dis_time_average_12

central_acc_average = sum(central_acc) / len(central_acc)
print "Average RPCA Accuracy: ", central_acc_average
central_time_average = sum(central_time) / len(central_time)
print "Average RPCA Time: ", central_time_average

f = open("Lambda_"+str(Lambda)+"_RPCA.txt", 'a')
f.write("Average RPCA Accuracy: "+str(central_acc_average)+"\n")
f.write("Average RPCA Time: "+str(central_time_average)+"\n")
f.write("Average DisRPCA Accuracy, 4 Nodes: "+str(dis_acc_average_4)+"\n")
f.write("Average DisRPCA Time: 4 Nodes"+str(dis_time_average_4)+"\n")
f.write("Average DisRPCA Accuracy, 8 Nodes: "+str(dis_acc_average_8)+"\n")
f.write("Average DisRPCA Time: 8 Nodes"+str(dis_time_average_8)+"\n")
f.write("Average DisRPCA Accuracy, 12 Nodes: "+str(dis_acc_average_12)+"\n")
f.write("Average DisRPCA Time: 12 Nodes"+str(dis_time_average_12)+"\n")
f.close()
