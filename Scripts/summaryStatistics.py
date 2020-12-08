import os
currentDirPath = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, os.path.dirname(currentDirPath)+"/Utils")
import dataReading
import dataCleaning
import calculateFeatures
import configparser
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt


summaryStatPlotsPath = os.path.dirname(currentDirPath) + '/Plots/Summary Statistics Plots'


# Step 1: Reading data:
config = configparser.ConfigParser()
config.read(os.path.dirname(currentDirPath)+'/config.ini')
inputFilesData=dataReading.readTextFiles(folderPath=config['DEFAULT']['trainingFilesPath'], lower=True)

print("finished step 1")

# Step 2: Pre-cleaning feature calculation: some feautures have to be calculated before cleaning (ex: number of special characters, named entities, ...):
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

print("finished step 2")


# Step 3: Cleaning: now that pre-cleaning features are calculated, cleaning can be applied:
inputFilesData=dataCleaning.fullDataCleaning(pdDataFrame=inputFilesData,
                                                         textColumnName="text",
                                                         corrSpelling=False, # sadly enough too time consuming for my computer..
                                                         repContractions=True,
                                                         remPunctuation=True,
                                                         lemmatize=True,
                                                         delStopWords=True)

print("finished step 3")




## Graph 1: counts by group:
ax = inputFilesData['author'].value_counts().sort_values().plot(kind='barh', title="Number of documents by author", figsize=(7, 6), rot=0, color=['red', 'green', 'blue'])
ax.set_xlabel('Number of documents')
ax.set_ylabel('Author')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for i, v in enumerate(inputFilesData['author'].value_counts().sort_values()):
    ax.text(v+10, i, str(v), fontweight='bold')

fig = plt.gcf()
fig.savefig(summaryStatPlotsPath + '/AuthorDistribution.png', dpi=600)
plt.close()




## Graph 2: Average text length by author:
ax = inputFilesData.groupby('author')['textLength'].mean().sort_values().plot(kind='barh', title="Average text length by author", figsize=(7, 6), rot=0, color=['red', 'green', 'blue'])
ax.set_xlabel('Average text length')
ax.set_ylabel('Author')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for i, v in enumerate(inputFilesData.groupby('author')['textLength'].mean().sort_values().round().astype(int)):
    ax.text(v+1, i, str(v), fontweight='bold')

fig = plt.gcf()
fig.savefig(summaryStatPlotsPath + '/AverageTextLength.png', dpi=600)
plt.close()





## Graph 3: Average sentiment by author:
ax = inputFilesData.groupby('author')['sentiment'].mean().sort_values().plot(kind='barh', title="Average sentiment score by author", figsize=(7, 6), rot=0, color=['red', 'green', 'blue'])
ax.set_xlabel('Average sentiment score')
ax.set_ylabel('Author')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for i, v in enumerate(inputFilesData.groupby('author')['sentiment'].mean().sort_values().round(3)):
    ax.text(v+0.001, i, str(v), fontweight='bold')

fig = plt.gcf()
fig.savefig(summaryStatPlotsPath + '/AverageSentiment.png', dpi=600)
plt.close()






## Graph 4: Average number of special characters by author:
ax = inputFilesData.groupby('author')['numberOfSpecialCharacters'].mean().sort_values().plot(kind='barh', title="Average number of special characters by document by author", figsize=(7, 6), rot=0, color=['red', 'green', 'blue'])
ax.set_xlabel('Average number of special characters')
ax.set_ylabel('Author')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for i, v in enumerate(inputFilesData.groupby('author')['numberOfSpecialCharacters'].mean().sort_values().round(2)):
    ax.text(v+0.02, i, str(v), fontweight='bold')

fig = plt.gcf()
fig.savefig(summaryStatPlotsPath + '/NumberOfSpecialCharacters.png', dpi=600)
plt.close()






## Graph 5: Average number of numerics: No numerics used here => not saving graph
ax = inputFilesData.groupby('author')['numberOfNumerics'].mean().sort_values().plot(kind='barh', title="Average number of special numerics by author", figsize=(7, 6), rot=0, color=['red', 'green', 'blue'])
ax.set_xlabel('Average number of numerics')
ax.set_ylabel('Author')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for i, v in enumerate(inputFilesData.groupby('author')['numberOfNumerics'].mean().sort_values().round(2)):
    ax.text(v+0.02, i, str(v), fontweight='bold')
plt.close()






## Graph 6: Average number of  specific special characters:
ax = inputFilesData[["author", "numberOfQuestionMarks", "numberOfQuotes", "numberOfDots"]].groupby('author').mean().plot(kind='bar', title="Average number of specific special characters by author", figsize=(7, 6), rot=0)
ax.set_xlabel('Author')
ax.set_ylabel('Punctuation types averages')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig = plt.gcf()
fig.savefig(summaryStatPlotsPath + '/AverageBySpecificPunctuation.png', dpi=600)
plt.close()






## Graph 7: Average number of named entities used by entity type by author:
ax = inputFilesData[["author", 'DATE', 'GPE', 'LOC', 'ORG', 'PERSON']].groupby('author').mean().plot(kind='bar', title="Average number of named entities used by entity type by author", figsize=(7, 6), rot=0)
ax.set_xlabel('Named entities type')
ax.set_ylabel('Average count by sentence')
fig = plt.gcf()
fig.savefig(summaryStatPlotsPath + '/UseOfNamedEntities.png', dpi=600)
plt.close()






## Graph 8: Average word length by author:
ax = inputFilesData.groupby('author')['avgWordLength'].mean().sort_values().plot(kind='barh', title="Average word length by author", figsize=(7, 6), rot=0, color=['red', 'green', 'blue'])
ax.set_xlabel('Average word length')
ax.set_ylabel('Author')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for i, v in enumerate(inputFilesData.groupby('author')['avgWordLength'].mean().sort_values().round(2)):
    ax.text(v+0.02, i, str(v), fontweight='bold')

fig = plt.gcf()
fig.savefig(summaryStatPlotsPath + '/AverageWordLength.png', dpi=600)
plt.close()






## Graph 9: TOP 10 most used words per author:
def top10Words(pandasColumn):
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(pandasColumn)
    total_word_counts = pd.DataFrame(word_count_vector.toarray(), columns=cv.get_feature_names()).sum(axis=0)
    top_10 = pd.DataFrame(total_word_counts, columns=["Total Count"]).sort_values(by="Total Count", ascending=False).head(10).sort_values(by="Total Count")

    return top_10


inputFilesDataMWS=inputFilesData[inputFilesData["author"]=="MWS"]
inputFilesDataEAP=inputFilesData[inputFilesData["author"]=="EAP"]
inputFilesDataHPL=inputFilesData[inputFilesData["author"]=="HPL"]

top10MWS=top10Words(inputFilesDataMWS["text"])
top10EAP=top10Words(inputFilesDataEAP["text"])
top10HPL=top10Words(inputFilesDataHPL["text"])


ax=top10MWS.plot(kind='barh', title="Top 10 words used by MWS (after stopword cleaning)", figsize=(7, 6), rot=0)
ax.set_xlabel('Number of times used')
ax.set_ylabel('Word')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for i, v in enumerate(top10MWS["Total Count"]):
    ax.text(v+1, i, str(v), fontweight='bold')

fig = plt.gcf()
fig.savefig(summaryStatPlotsPath + '/Top10WordsMWS.png', dpi=600)
plt.close()

ax=top10EAP.plot(kind='barh', title="Top 10 words used by EAP (after stopword cleaning)", figsize=(7, 6), rot=0)
ax.set_xlabel('Number of times used')
ax.set_ylabel('Word')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for i, v in enumerate(top10EAP["Total Count"]):
    ax.text(v+1, i, str(v), fontweight='bold')

fig = plt.gcf()
fig.savefig(summaryStatPlotsPath + '/Top10WordsEAP.png', dpi=600)
plt.close()

ax=top10HPL.plot(kind='barh', title="Top 10 words used by HPL (after stopword cleaning)", figsize=(7, 6), rot=0)
ax.set_xlabel('Number of times used')
ax.set_ylabel('Word')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for i, v in enumerate(top10HPL["Total Count"]):
    ax.text(v+1, i, str(v), fontweight='bold')

fig = plt.gcf()
fig.savefig(summaryStatPlotsPath + '/Top10WordsHPL.png', dpi=600)
plt.close()






# Graph 10: Average numbers of stopwords per sentence by author:
ax = inputFilesData.groupby('author')['numberOfStopWords'].mean().sort_values().plot(kind='barh', title="Average numbers of stopwords per sentence by author", figsize=(7, 6), rot=0, color=['red', 'green', 'blue'])
ax.set_xlabel('Average number of stopwords by sentence')
ax.set_ylabel('Author')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for i, v in enumerate(inputFilesData.groupby('author')['numberOfStopWords'].mean().sort_values().round().astype(int)):
    ax.text(v+0.002, i, str(v), fontweight='bold')

fig = plt.gcf()
fig.savefig(summaryStatPlotsPath + '/AverageNumberOfStopwords.png', dpi=600)
plt.close()