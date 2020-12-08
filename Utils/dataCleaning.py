import os
import re
import json
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.data import find as dataFindNLTK
from nltk import download as downloadNLTK, pos_tag
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

currentDirPath = os.path.dirname(os.path.realpath(__file__))

# Check if nltk resources are present. If not, download those resources:
try:
    dataFindNLTK('tokenizers/punkt')
except LookupError:
     downloadNLTK('punkt')

try:
    dataFindNLTK('corpora/wordnet')
except LookupError:
     downloadNLTK('wordnet')

try:
    dataFindNLTK('taggers/averaged_perceptron_tagger')
except LookupError:
    downloadNLTK('averaged_perceptron_tagger')



## Function to replace contraction words:
contractionsPath = os.path.dirname(currentDirPath) + "/Data/contractions/contractionsMap_EN.txt" # loading contraction words outside function so that it is not loaded for each new sentence of a dataframe!!!
with open(contractionsPath) as file:
        contractionsMap = file.read()
        contractionsMap = json.loads(contractionsMap)

def replaceContractions(text):
    for word in text.split():
        if word.lower() in contractionsMap:
            text = text.replace(word, contractionsMap[word.lower()])

    return text

## Function to correct spelling:
def correctSpellingInSentence (sentence, returnCorrectedDummy=True):
    correctedSentence=TextBlob(sentence).correct().string
    if returnCorrectedDummy:
        if correctedSentence==sentence:
            containsSpellingMistake=0
        else:
            containsSpellingMistake=1
        return correctedSentence, containsSpellingMistake
    else:
        return correctedSentence



## Function to delete stopwords:
stopWordsPath=os.path.dirname(currentDirPath)+"/Data/stopWords/stopWords_EN.txt" # loading stopwords outside function so that it is not loaded for each new sentence of a dataframe!!!
with open(stopWordsPath) as file:
        stopWords = file.read()
        stopWords = json.loads(stopWords)["topwords"]
        stopWords = list(map(str.strip, stopWords))

def deleteStopwords(text):
    words=word_tokenize(text)
    filteredWords=[w for w in words if not w.lower() in stopWords]
    filteredSentence=' '.join(filteredWords)

    return filteredSentence



## Function to stem words from sentence using Porter's stemming technique:
porter = PorterStemmer()
def stemming(text):
    return " ".join([porter.stem(i.lower()) for i in text.split()])


## Function to lemmatize words from sentence. This is not default lemmatize because I changed lemmetazing according to wor type (verb, noun, ..):
lemmatizer = WordNetLemmatizer()
def lemmatizing(text):
    lemmatizedList=[lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i,j in pos_tag(word_tokenize(text))]
    return " ".join(lemmatizedList)

## Functions to remove punctuation from sentence:
def removePunctuation(text):
    return re.sub(r'[^\w\s]', '', text)


## all together applied to one pandas column: order is very important
def fullDataCleaning(pdDataFrame, textColumnName, corrSpelling=True, repContractions=True, remPunctuation=True, lemmatize=True, delStopWords=True):
    if corrSpelling:
        pdDataFrame[textColumnName], pdDataFrame["containsSpellingMistake"] =zip(*pdDataFrame[textColumnName].map(correctSpellingInSentence))
    if repContractions:
        pdDataFrame[textColumnName] = pdDataFrame[textColumnName].apply(lambda x: replaceContractions(x))
    if remPunctuation:
        pdDataFrame[textColumnName] = pdDataFrame[textColumnName].apply(lambda x: removePunctuation(x))
    if lemmatize:
        pdDataFrame[textColumnName] = pdDataFrame[textColumnName].apply(lambda x: lemmatizing(x))
    if delStopWords:
        pdDataFrame[textColumnName] = pdDataFrame[textColumnName].apply(lambda x: deleteStopwords(x))

    return pdDataFrame


## Testing functions:
if __name__ == '__main__':

    text="idris wasn't well content, with this resolved of mine."
    print("orginal sentence:")
    print(text)
    print("\n")

    print("Sentence without contractions replaced:")
    textNoContractions=replaceContractions(text)
    print(textNoContractions)
    print("\n")

    print("Sentence without contractions and punctuation removed:")
    textNoPunct = removePunctuation(textNoContractions)
    print(textNoPunct)
    print("\n")

    print("Sentence without contractions and punctuation removed and stemmed:")
    textStem = stemming(textNoPunct)
    print(textStem)
    print("\n")

    print("Sentence without contractions and punctuation removed and lemmatized:")
    textLemmatized = lemmatizing(textNoPunct)
    print(textLemmatized)
    print("\n")

    print("Sentence without contractions and punctuation removed and lemmatized and stopwords deleted:")
    textNoStopwords = deleteStopwords(textLemmatized)
    print(textNoStopwords)
    print("\n")

    mistakeSentence="Arter I got the book off Eb I uster look at it a lot, especial when I'd heerd Passon Clark rant o' Sundays in his big wig."
    print("Original sentence with spelling mistakes")
    print(mistakeSentence)
    print("\n")

    print("Sentence with corrected spelling")
    print(correctSpellingInSentence(mistakeSentence))
    print("\n")


    print("Testing to apply all cleaning on a column of a dataframe with function: fullDataCleaning")

    testData = pd.DataFrame(["Idris was well content with this resolve of mine.",
                  "I was faint, even fainter than the hateful modernity of that accursed city had made me."],
                 columns=["text"])

    print("Original data:")
    print(testData)
    print("\n")

    testData = fullDataCleaning(pdDataFrame=testData, textColumnName="text", corrSpelling=True, repContractions=True, remPunctuation=True, lemmatize=True, delStopWords=True)

    print("Data after cleaning:")
    print(testData)




