import os
import configparser
import pandas as pd
currentDirPath = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, os.path.dirname(currentDirPath)+"/Utils")
import dataReading
import dataCleaning
import calculateFeatures
import modelSelection


## Step 1: Reading data:
config = configparser.ConfigParser()
config.read(os.path.dirname(currentDirPath)+'/config.ini')
inputFilesData=dataReading.readTextFiles(folderPath=config['DEFAULT']['trainingFilesPath'], lower=True, parseIDAuthorName=True)
print("finished step 1: Loading the data")



## Step 2: Pre-cleaning feature calculation: some feautures have to be calculated before cleaning (ex: number of special characters, named entities, ...):
inputFilesData=calculateFeatures.fullFeatureCalculation (pdDataFrame=inputFilesData,
                                                         textColumnName="text",
                                                         applyTextLength=True,
                                                         applyPunctiationMeasures=True,
                                                         applyCountByNamedEntityType=True,
                                                         applyNumberOfWords=True,
                                                         applyNumberOfStopWords=True,
                                                         applyAvgWordLength=True,
                                                         applyNumberOfNumerics=True,
                                                         applySentiment=True,
                                                         applyTfIdf=False)

print("finished step 2: Calculating pre-cleaning features")



## Step 3: Cleaning: now that pre-cleaning features are calculated, cleaning can be applied:
inputFilesData=dataCleaning.fullDataCleaning(pdDataFrame=inputFilesData,
                                                         textColumnName="text",
                                                         corrSpelling=False, # sadly enough too time consuming for my computer..
                                                         repContractions=True,
                                                         remPunctuation=True,
                                                         lemmatize=True,
                                                         delStopWords=True)

print("finished step 3: Cleaning the text data")



## step 4: apply feature engeneering that need to be applied on cleaned data (IDF) and save IDF model to apply later to new documents:
inputFilesData["TfIdf"]=calculateFeatures.TfIdf(pandasColumn=inputFilesData["text"], modelSavePath=os.path.dirname(currentDirPath)+"/Models/TFIDF Model/tfidfmodel.pkl")

# All tfidf values are contained in one column (column where each cell is a list of values). We will transform that to multople columns:
splittedTfIdf=pd.DataFrame(inputFilesData["TfIdf"].values.tolist())
inputFilesData=pd.concat([inputFilesData, splittedTfIdf], axis=1)
del inputFilesData["TfIdf"]

print("finished step 4: Calculating after-cleaning measures")




## Step 5: training the model: Testing two configurations: all features and TFIDF only

# deleting unnecessary cols:
del inputFilesData["text"]
del inputFilesData["path"]
del inputFilesData["fileName"]
del inputFilesData["id"]


# Some models (ex: Naive Bayes) don't accept negative values but sentiment is between [-1.0, 1.0]. The solution is to shift range to [0.0, 2.0]:
if "sentiment" in inputFilesData.columns:
    inputFilesData["sentiment"]=inputFilesData["sentiment"]+1

# split data into target and predictors:
Y, X = inputFilesData.iloc[:,0], inputFilesData.iloc[:,1:]

modelResultsList=[] # initialize a list to collect results of different model configurations

bestModelAllFeatures, bestAccuracyAllFeatures = modelSelection.selectModel(X=X,
                                                     Y=Y,
                                                     testSize=0.2,
                                                     applyXgBoost=True,
                                                     applyRandomForest=True,
                                                     applyGbm=True,
                                                     applyNaiveBayes=True,
                                                     applyH2oAutoMl=True,
                                                     H2o_max_runtime_secs=3600,
                                                     savePlots=True,
                                                     featureSelecetion=True,
                                                     random_state=2,
                                                     plotSavePath=os.path.dirname(currentDirPath) + '/Plots/Model Plots/Full Model')

modelResultsList.append((bestModelAllFeatures, bestAccuracyAllFeatures))


bestModelTFIDFOnly, bestAccuracyTFIDFOnly = modelSelection.selectModel(X=splittedTfIdf,
                                                     Y=Y,
                                                     testSize=0.2,
                                                     applyXgBoost=True,
                                                     applyRandomForest=True,
                                                     applyGbm=True,
                                                     applyNaiveBayes=True,
                                                     applyH2oAutoMl=True,
                                                     H2o_max_runtime_secs=3600,
                                                     savePlots=True,
                                                     featureSelecetion=True,
                                                     random_state=2,
                                                     plotSavePath=os.path.dirname(currentDirPath) + '/Plots/Model Plots/TFIDF Only Model')

modelResultsList.append((bestModelTFIDFOnly, bestAccuracyTFIDFOnly))


bestModelTuple=max(modelResultsList,key=lambda item:item[1])




## Step 6: Save best model:
modelSelection.saveModel(model=bestModelTuple[0], path=os.path.dirname(currentDirPath) + "/Models/bestPerformingModel")
