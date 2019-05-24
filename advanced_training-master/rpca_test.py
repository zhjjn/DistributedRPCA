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

    threshold = [0.15]
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

data = pd.read_csv("../dataset/NSL_KDD-master/20110413_log_1_all.csv", iterator = True)
data_all = data.get_chunk(1000)
dataX1 = data_all.copy()
dataX2 = dataX1 #.copy().drop(['difficulty'], axis = 1)
dataX = dataX2 #pd.get_dummies(dataX2)
dataY = data_all['Operation_Teardown'].copy()

for i in range(0, len(dataY)):
    if dataY[i] == 0:
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
rpca = RobustPCA(lam = 0.175).fit(X_train)

X_train_PCA = rpca.transform(X_train)
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train.index)

X_train_PCA_inverse = rpca.inverse_transform(X_train_PCA)
X_train_PCA_inverse = pd.DataFrame(data=X_train_PCA_inverse, \
                                   index=X_train.index)

anomalyScoresPCA = anomalyScores(X_train, X_train_PCA_inverse)
preds = plotResults(y_train, anomalyScoresPCA, True)

'''
print "Parameters:"
print rpca.components_.tolist()
components = json.dumps(rpca.components_.tolist())

HOST = "192.168.1.122"
PORT = 6666

sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock.connect((HOST, PORT))
sock.sendall(components)
data = sock.recv(1024)

print("Received", repr(data))
'''
