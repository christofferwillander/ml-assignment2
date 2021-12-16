from columnar import columnar
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

    modelAccuracies = []
    modelF1Scores = []
    modelTrainingTimes = []

    curFold = 1

    # Performing stratified 10-fold cross validation, training and evaluation
    for trainIndex, testIndex in stratKFold.split(dataX, dataY):
        trainFold = dataset.iloc[trainIndex, :]
        testFold = dataset.iloc[testIndex, :]
        # print("-------- Fold: " + str(curFold) + " --------")
        curModelAccuracies = []
        curModelF1Scores = []
        curModelTrainingTimes = []
        for model in range(len(models)):
            accuracy, f1Score, trainingTime = trainModel(models[model], trainFold, testFold, curFold)
            curModelAccuracies.append(accuracy)
            curModelF1Scores.append(f1Score)
            curModelTrainingTimes.append(trainingTime)
        
        modelAccuracies.append(np.array(curModelAccuracies))
        modelF1Scores.append(np.array(curModelF1Scores))
        modelTrainingTimes.append(np.array(curModelTrainingTimes))
        curFold += 1
    
    printResults(modelAccuracies, models, "ACCURACY", False)
    printResults(modelF1Scores, models, "F1 SCORE", False)
    printResults(modelTrainingTimes, models, "TRAINING TIME", True)


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
    return accuracyScore, f1Score, trainingTime

def printResults(data, models, statType, ascOrder):
    headers = ["Fold"]
    dataRows = []

    for model in range(len(models)):
        headers.append(type(models[model]).__name__)

    for dataPoints in range (len(data)):
        dataRow = [str(dataPoints + 1)]
        #dataRanks = np.argsort(data[dataPoints])
        dataRanks = pd.DataFrame(data[dataPoints])
        dataRanks = dataRanks.rank(ascending=ascOrder)
        dataRanks = dataRanks.to_numpy()
        #print(ranks.rank(ascending=ascOrder))

        # if (ascOrder):
        #     dataRanks = dataRanks[::-1][:len(dataRanks)]

        for dataPoint in range(len(data[dataPoints])):
            curValue = format(round(data[dataPoints][dataPoint], 4), '.4f')
            tempRow = str(curValue) + " (" + str(int(dataRanks[dataPoint])) + ")"
            dataRow.append(tempRow)
        dataRows.append(dataRow)

    tempRow = ["MEAN: "]
    for dataPoints in range(len(data[0])):
        tempValues = []
        for dataPoint in range(len(data)):
            tempValues.append(data[dataPoint][dataPoints])
        meanVal = sum(tempValues) / len(tempValues)
        tempRow.append(format(round(meanVal, 4), '.4f'))
    
    dataRows.append(tempRow)



    table = columnar(dataRows, headers)
    print("Stratified ten-fold cross validation (" + statType + ")")
    print(table)

    


if __name__ == "__main__":
    main()