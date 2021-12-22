from columnar import columnar
from time import time
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score

def main():
    # Reading dataset into Pandas DataFrame
    dataset = pd.read_csv("./data/spambase.data", sep=",", header=None)

    # Declaring an instance class for stratified 10-fold cross validation 
    stratKFold = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)

    # Splitting data into X (features) and Y (spam/ham label) values
    dataX = dataset[dataset.columns[0:57]]
    dataY = dataset[57]

    # Initializing array for classification models
    models = []

    # Creating SVC, Gaussian Naive Bayes, Random Forest classification models
    models.append(svm.SVC())
    models.append(GaussianNB())
    models.append(RandomForestClassifier())

    # Initializing binning discretizer (ordinal, equal-width binning with 100 bins)
    discretizer = KBinsDiscretizer(n_bins=100, encode="ordinal", strategy="uniform")

    # Initializing arrays for storing model metrics
    modelAccuracies = []
    modelF1Scores = []
    modelTrainingTimes = []

    # Performing stratified 10-fold cross validation, training and evaluation
    for trainIndex, testIndex in stratKFold.split(dataX, dataY):
        # Creating folds for training and testing based on generated indices
        trainFold = dataset.iloc[trainIndex, :]
        testFold = dataset.iloc[testIndex, :]

        # Initializing structures for holding model metrics
        curModelAccuracies = []
        curModelF1Scores = []
        curModelTrainingTimes = []

        for model in range(len(models)):
            # Training models and calculating performance metrics
            accuracy, f1Score, trainingTime = trainModel(models[model], discretizer, trainFold, testFold)
            curModelAccuracies.append(accuracy)
            curModelF1Scores.append(f1Score)
            curModelTrainingTimes.append(trainingTime)
        
        # Appending model metrics for each fold iteration
        modelAccuracies.append(np.array(curModelAccuracies))
        modelF1Scores.append(np.array(curModelF1Scores))
        modelTrainingTimes.append(np.array(curModelTrainingTimes))
    
    # Printing different metrics
    printResults(modelAccuracies, models, "ACCURACY", False)
    printResults(modelF1Scores, models, "F1 SCORE", False)
    printResults(modelTrainingTimes, models, "TRAINING TIME", True)


def trainModel(model, transformer, trainFold, testFold):
    # Splitting training data into X, Y
    trainX = trainFold[trainFold.columns[0:57]]
    trainY = trainFold[57]

    # Splitting test data into X, Y
    testX = testFold[testFold.columns[0:57]]
    testY = testFold[57]

    # Discretizing training and test data
    trainX_disc = transformer.fit_transform(trainX)
    testX_disc = transformer.transform(testX)

    # Exctracting timer before starting model fitting
    trainingTime = time()

    # Fitting model to training data 
    model.fit(trainX_disc, trainY)

    # Calculating training time for model
    trainingTime = time() - trainingTime

    # Performing a prediction based on test data
    predY = model.predict(testX_disc)

    # Calculating accuracy score based on prediction and correct labels
    accuracyScore = accuracy_score(testY, predY)

    # Calculating F1 score based on prediction and correct labels
    f1Score = f1_score(testY, predY)

    return accuracyScore, f1Score, trainingTime

def printResults(data, models, statType, ascOrder):
    headers = ["Fold"]
    dataRows = []
    meanRank = [0, 0, 0]
    allRanks = []

    for model in range(len(models)):
        headers.append(type(models[model]).__name__)

    # Calculating rank for each data point for every model
    for dataPoints in range (len(data)):
        dataRow = [str(dataPoints + 1)]
        dataRanks = pd.DataFrame(data[dataPoints])
        dataRanks = dataRanks.rank(ascending=ascOrder)
        dataRanks = dataRanks.to_numpy()
        allRanks.append(dataRanks)

        for dataPoint in range(len(data[dataPoints])):
            curValue = format(round(data[dataPoints][dataPoint], 4), '.4f')
            tempRow = str(curValue) + " (" + str(int(dataRanks[dataPoint])) + ")"
            dataRow.append(tempRow)
            meanRank[dataPoint] += int(dataRanks[dataPoint])
        dataRows.append(dataRow)
    
    # Calculating average of metric for each model
    tempRow = ["AVG: "]
    for dataPoints in range(len(data[0])):
        tempValues = []
        for dataPoint in range(len(data)):
            tempValues.append(data[dataPoint][dataPoints])
        meanVal = sum(tempValues) / len(tempValues)
        tempRow.append(format(round(meanVal, 4), '.4f'))
    
    dataRows.append(tempRow)

    # Calculating standard deviation of metrics for each model
    tempRow = ["STDEV: "]
    for dataPoints in range(len(data[0])):
        tempValues = []
        for dataPoint in range(len(data)):
            tempValues.append(data[dataPoint][dataPoints])
        tempValues = np.array(tempValues)
        sdVal = np.std(tempValues)
        tempRow.append(format(round(sdVal, 4), '.4f'))
    
    dataRows.append(tempRow)

    # Calculating average rank for each model
    tempRow = ["AVG RANK: "]
    for model in range(len(meanRank)):
        meanRank[model] = meanRank[model]/len(data)
        tempRow.append(format(round(meanRank[model], 1), '.1f'))

    dataRows.append(tempRow)


    # Printing data as table
    table = columnar(dataRows, headers)
    print("\nStratified ten-fold cross validation (" + statType + ")")
    print(table)
    friedmanTest(models, data, allRanks, meanRank)

def friedmanTest(models, data, allRanks, meanRanks):
    criticalValue = 6.2
    avgRank = (len(models) + 1) / 2
    sqDifMeans = 0
    sqDifRanks = 0
    
    # Calculating the square difference between average rank per model and overall average rank
    for model in range(len(models)):
        sqDifMeans += pow((meanRanks[model] - avgRank), 2)
    sqDifMeans = sqDifMeans * len(data)

    # Calculating square differences for the rank in each fold for every model (compared to overall average rank)
    for model in range(len(models)):
        for rank in range(len(allRanks)):
            sqDifRanks += pow((allRanks[rank][model] - avgRank), 2)
    
    sqDifRanks = sqDifRanks * (1 / (len(data)*(len(models) - 1)))

    # Calculating final Friedman score
    friedmanScore = float(sqDifMeans * sqDifRanks)

    # Printing results and checking if to move on to the Nemenyi post-hoc test
    print("------------------------------------------------------------------------------------------------------------------------")
    if (friedmanScore > criticalValue):
        print("Friedman score of " + str(format(round(friedmanScore, 3), '.3f')) + " is greater than critical value " + str(format(criticalValue, '.3f')) + ", null hypothesis rejected.")
        print("------------------------------------------------------------------------------------------------------------------------")
        nemenyiTest(models, data, meanRanks)
    else:
        print("Friedman score of " + str(format(round(friedmanScore, 3), '.3f')) + " is less than critical value " + str(format(criticalValue, '.3f')) + ", null hypothesis could not be rejected (i.e. algorithms performed equally well).")
        print("------------------------------------------------------------------------------------------------------------------------")
        
def nemenyiTest(models, data, meanRanks):
    # Q-alpha value for alpha at significance level 0.05, k = 3
    qAlpha = 2.343

    # Calculating critical difference threshold value
    criticalDifference = qAlpha * math.sqrt((len(models)*(len(models)+1))/(6*len(data)))
    print("Performing Nemenyi post-hoc test at significance level 0.05, critical difference threshold: " + str(format(round(criticalDifference, 4), '.3f')) + ".")
    print("------------------------------------------------------------------------------------------------------------------------")

    differences = []
    modelNames = []

    # Calculating absolute differences between model metrics (pair-wise)
    for model in range(len(models) - 1):
        for comparison in range(model + 1, len(models)):
            curComparison = []
            curDif = []
            curDif.append(abs(meanRanks[model] - meanRanks[comparison]))
            curDif.append(meanRanks[model])
            curDif.append(meanRanks[comparison])
            differences.append(curDif)
            curComparison.append(type(models[model]).__name__)
            curComparison.append(type(models[comparison]).__name__)
            modelNames.append(curComparison)
    
    # Printing results from Nemenyi test
    for result in range(len(differences)):
        if differences[result][0] >= criticalDifference:
            print("There is a statistically significant difference in performance between models " + modelNames[result][0] + ", " + modelNames[result][1] + " (" + str(format(round(differences[result][0], 1), '.1f')) + ").")
            if differences[result][1] > differences[result][2]:
                print("The " + modelNames[result][1] + " model performed better." )
            else:
                print("The " + modelNames[result][0] + " model performed better." )
        else:
            print("There is not a statistically significant difference in performance between models " + modelNames[result][0] + ", " + modelNames[result][1] + " (" + str(format(round(differences[result][0], 1), '.1f')) + ").")
        print("------------------------------------------------------------------------------------------------------------------------")
        
        
        


    


    


if __name__ == "__main__":
    main()