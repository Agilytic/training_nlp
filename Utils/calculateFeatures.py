from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import pandas as pd
from textblob import TextBlob
import os
import json
import spacy

currentDirPath = os.path.dirname(os.path.realpath(__file__))


def textLength(pandasColumn):
    return pandasColumn.str.len()

def numberOfSpecialCharacters(pandasColumn):
    return pandasColumn.str.count('[^\w\s]')

def numberOfQuestionMarks(pandasColumn):
    return pandasColumn.str.count('[?]')

def numberOfExclamationMarks(pandasColumn):
    return pandasColumn.str.count('[!]')

def numberOfDots(pandasColumn):
    return pandasColumn.str.count('[.]')

def numberOfQuotes(pandasColumn):
    return (pandasColumn.str.count('["]') + pandasColumn.str.count("[']"))

def numberOfWords(pandasColumn):
    return pandasColumn.apply(lambda x: len(str(x).split(" ")))

def numberOfStopWords(pandasColumn):
    stopWordsPath = os.path.dirname(currentDirPath) + "/Data/stopWords/stopWords_EN.txt"
    with open(stopWordsPath) as file:
        stopWords = file.read()
        stopWords = json.loads(stopWords)["topwords"]
        stopWords = list(map(str.strip, stopWords))
    return pandasColumn.apply(lambda x: len([x for x in x.split() if x.lower() in stopWords]))

def avgWordLength(pandasColumn):
    def avgWord(sentence):
        words = sentence.split()
        return (sum(len(word) for word in words) / len(words))
    return pandasColumn.apply(lambda x: avgWord(x))

def numberOfNumerics(pandasColumn):
    return pandasColumn.apply(lambda x: len([x for x in x.split() if x.isdigit()]))

def sentiment(pandasColumn):
    return pandasColumn.apply(lambda x: TextBlob(x).sentiment.polarity)

def TfIdf(pandasColumn, useIdf=True, modelSavePath=None, modelLoadPath=None, maxFeatures=500):

    if (modelSavePath) and (modelLoadPath):
        raise ValueError('modelSavePath and modelLoadPath can not be both filled. Put one of the two to None!')

    corpus = np.array(pandasColumn)

    if not modelLoadPath:
        vectorizer = TfidfVectorizer(use_idf=useIdf, max_features=maxFeatures)
        tfidf_result=vectorizer.fit_transform(corpus).toarray()
        if modelSavePath:
            # Save vectorizer.vocabulary_ to apply the same terms to new docs:
            pickle.dump(vectorizer, open(modelSavePath, "wb"))

    else:
        vectorizer = pickle.load(open(modelLoadPath, "rb"))
        tfidf_result=vectorizer.transform(corpus).toarray()


    return pd.Series(tfidf_result.tolist())



def countByNamedEntityType(pdDataFrame, textColumnName):
    nlp = spacy.load(os.path.dirname(currentDirPath)+"/Data/preTrainedModels/en_core_web_sm/en_core_web_sm-2.2.5")
    results = []
    for sentence in pdDataFrame[textColumnName]:
        labels = []
        doc = nlp(sentence)
        for ent in doc.ents:
            try:
                labels.append([ent.text, ent.label_])
            except Exception as e:
                pass
        if len(labels) > 0:
            result = pd.DataFrame(labels).groupby(1).count().T
        else:
            result = pd.DataFrame([0], columns=["GPE"])

        results.append(result)

    finalResult = pd.concat(results, sort=True).reset_index(drop=True).fillna(0)

    for entityName in nlp.entity.labels:
        if entityName not in finalResult.columns:
            finalResult[entityName] = 0
    return pd.concat([pdDataFrame, finalResult], axis=1)


def fullFeatureCalculation(pdDataFrame, textColumnName, applyTextLength=True, applyPunctiationMeasures=True, applyCountByNamedEntityType=True, applyNumberOfWords=True, applyNumberOfStopWords=True, applyAvgWordLength=True, applyNumberOfNumerics=True, applySentiment=True, applyTfIdf=True, TfIdfModelSavePath=None, TfIdfmMdelLoadPath=None):
    if applyTextLength:
        pdDataFrame["textLength"] = textLength(pandasColumn=pdDataFrame[textColumnName])
    if applyPunctiationMeasures:
        pdDataFrame["numberOfSpecialCharacters"]=numberOfSpecialCharacters(pdDataFrame[textColumnName])
        pdDataFrame["numberOfQuestionMarks"]=numberOfQuestionMarks(pdDataFrame[textColumnName])
        pdDataFrame["numberOfExclamationMarks"]=numberOfExclamationMarks(pdDataFrame[textColumnName])
        pdDataFrame["numberOfQuotes"]=numberOfQuotes(pdDataFrame[textColumnName])
        pdDataFrame["numberOfDots"] = numberOfDots(pdDataFrame[textColumnName])
    if applyCountByNamedEntityType:
        pdDataFrame=countByNamedEntityType(pdDataFrame=pdDataFrame, textColumnName=textColumnName)
    if applyNumberOfWords:
        pdDataFrame["numberOfWords"] = numberOfWords(pandasColumn=pdDataFrame[textColumnName])
    if applyNumberOfStopWords:
        pdDataFrame["numberOfStopWords"] = numberOfStopWords(pandasColumn=pdDataFrame[textColumnName])
    if applyAvgWordLength:
        pdDataFrame["avgWordLength"] = avgWordLength(pandasColumn=pdDataFrame[textColumnName])
    if applyNumberOfNumerics:
        pdDataFrame["numberOfNumerics"] = numberOfNumerics(pandasColumn=pdDataFrame[textColumnName])
    if applySentiment:
        pdDataFrame["sentiment"] = sentiment(pandasColumn=pdDataFrame[textColumnName])
    if applyTfIdf:
        pdDataFrame["TfIdf"] = TfIdf(pandasColumn=pdDataFrame[textColumnName], modelSavePath=TfIdfModelSavePath, modelLoadPath=TfIdfmMdelLoadPath)
    return pdDataFrame




if __name__ == '__main__':
    trainData = pd.DataFrame(["Idris was well content with this resolve of mine.",
                             "I was faint, even fainter than the hateful modernity of that accursed city had made me."],
                            columns=["text"])

    testDoc = pd.DataFrame(["Idris said: I want to test if it works Diego"],
                            columns=["text"])

    # TEST 1: testing TFIDF:
    trainData["TfIdfValues"]=TfIdf(trainData["text"], modelSavePath="tfidfmodel.pkl", useIdf=False)
    print(trainData)
    print("\n")

    testDoc["TfIdfValues"] = TfIdf(testDoc["text"], modelLoadPath="tfidfmodel.pkl", useIdf=False)
    print(testDoc)
    print("\n")

    print("Don't forget to delete tfidfmodel.pkl!!")

    # TEST 2: testing sentiment:
    trainData["sentiment"]=sentiment(pandasColumn=trainData["text"])
    print(trainData)

    # TEST 3: testing named entities:
    trainData = pd.DataFrame(["I love brussels and new-york and trump and facebook ",
                              "Jonathan only likes Philippines like Diego. He said it to Trump"],
                            columns=["text"])

    print("Named entities count:")
    print(countByNamedEntityType(pdDataFrame=trainData, textColumnName="text"))


    # TEST 4: testing combined feature creation:
    trainData = pd.DataFrame(["Idris was well content with this resolve of mine.",
                             "I was faint, even fainter than the hateful modernity of that accursed city had made me."],
                            columns=["text"])

    fullTrainFeatures=fullFeatureCalculation(pdDataFrame=trainData, textColumnName="text")


    print(fullTrainFeatures)




