from time import time
import pandas as pd
import numpy as np
from scipy.sparse import data
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score
def main():
    # Reading dataset into Pandas DataFrame
    dataset = pd.read_csv("./data/spambase.data", header=None)

    # Declaring an instance class for stratified 10-fold cross validation 
    stratKFold = StratifiedKFold(n_splits=10)

    # Splitting data into X (features) and Y (spam/ham label) values
    dataX = dataset[dataset.columns[0:57]]
    dataY = dataset[57]

    #scaler = StandardScaler()
    #dataX = scaler.fit_transform(dataX)
    models = []
    models.append(svm.SVC())
    models.append(GaussianNB())
    models.append(RandomForestClassifier())

    curFold = 1

    # Performing stratified 10-fold cross validation, training and evaluation
    for trainIndex, testIndex in stratKFold.split(dataX, dataY):
        trainFold = dataset.iloc[trainIndex, :]
        testFold = dataset.iloc[testIndex, :]
        print("-------- Fold: " + str(curFold) + " --------")
        for model in range(len(models)):
            trainModel(models[model], trainFold, testFold, curFold)
        
        curFold += 1


def trainModel(model, trainFold, testFold, foldNum):
    trainX = trainFold[trainFold.columns[0:57]]
    trainY = trainFold[57]

    testX = testFold[testFold.columns[0:57]]
    testY = testFold[57]

    trainingTime = time()
    model.fit(trainX, trainY)
    trainingTime = time() - trainingTime
    predY = model.predict(testX)
    accuracyScore = accuracy_score(testY, predY)
    f1Score = f1_score(testY, predY)
    print("-------- " + type(model).__name__ + " --------")
    print("Accuracy score: " + str(accuracyScore))
    print("F1 score: " + str(f1Score))
    print("Training time: " + str(trainingTime))
    print("--------------------------------------------")

if __name__ == "__main__":
    main()