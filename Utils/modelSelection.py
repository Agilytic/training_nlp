from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import pickle
from pandas.plotting import table
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import h2o
from h2o.automl import H2OAutoML
from sklearn.feature_selection import SelectFromModel

currentDirPath = os.path.dirname(os.path.realpath(__file__))


def buildROC(y_test,y_pred, title, pngname, plotSavePath=os.path.dirname(currentDirPath) + '/Plots/Model Plots'):
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.gcf().savefig(plotSavePath + "/" +pngname, dpi=600)



def buildConfusionMatrix(y_test, y_pred, title, pngname, plotSavePath=os.path.dirname(currentDirPath) + '/Plots/Model Plots'):
    cm = confusion_matrix(y_test, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(set(y_test))
    ax.yaxis.set_ticklabels(set(y_test))

    fig = plt.gcf()
    fig.savefig(plotSavePath+ "/" + pngname, dpi=600)
    plt.close()




def buildClassificationReport(y_test, y_pred, title, pngname, plotSavePath=os.path.dirname(currentDirPath) + '/Plots/Model Plots'):
    classificationReport = classification_report(y_test, y_pred, target_names=set(y_test), output_dict=True)
    df = pd.DataFrame(classificationReport).transpose()
    df=df.round(3)
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title(title)

    table(ax, df, loc='upper center')

    plt.savefig(plotSavePath+ "/" + pngname,  bbox_inches = "tight")
    plt.close()




def xgBoost (X_train, X_test, y_train, y_test, savePlots=True, random_state=1, featureSelecetion=True, plotSavePath=os.path.dirname(currentDirPath) + '/Plots/Model Plots'):
    model = XGBClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    if featureSelecetion:
        selection = SelectFromModel(model, prefit=True)
        select_X_train = selection.transform(X_train)
        # retraining model on subset:
        model = XGBClassifier(random_state=random_state)
        model.fit(select_X_train, y_train)
        X_test=selection.transform(X_test)

        # keep features of interest:
        cols = selection.get_support(indices=True)
        selected_features = list(X_train.iloc[:, cols].columns)
        model.feature_names = selected_features
    else:
        model.feature_names = list(X_train.columns.values)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if savePlots:
        confusionTitle='Confusion Matrix for XGBoost Model' + " (Accuracy: "+ str(round(accuracy, 2))+")"
        buildConfusionMatrix(y_test, y_pred, title=confusionTitle, pngname='XGBoostConfusionMatrix.png', plotSavePath=plotSavePath)
        #buildROC(y_test, y_pred, title='ROC Curve for XGBoost Model', pngname='XGBoostROCCurve.png') # not for multiclass
        buildClassificationReport(y_test, y_pred, title='Classification Report for XGBoost Model', pngname='XGBoostClassificationReport.png', plotSavePath=plotSavePath)

    return model, accuracy





def randomForest(X_train, X_test, y_train, y_test, savePlots=True, random_state=1, featureSelecetion=True, plotSavePath=os.path.dirname(currentDirPath) + '/Plots/Model Plots'):
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    if featureSelecetion:
        selection = SelectFromModel(model, prefit=True)
        select_X_train = selection.transform(X_train)
        # retraining model on subset:
        model = RandomForestClassifier(random_state=random_state)
        model.fit(select_X_train, y_train)
        X_test = selection.transform(X_test)

        # keep features of interest:
        cols = selection.get_support(indices=True)
        selected_features = list(X_train.iloc[:, cols].columns)
        model.feature_names = selected_features
    else:
        model.feature_names = list(X_train.columns.values)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if savePlots:
        confusionTitle='Confusion Matrix for Random Forest Model'+ " (Accuracy: "+ str(round(accuracy, 2))+")"
        buildConfusionMatrix(y_test, y_pred, title=confusionTitle, pngname='RandomForestConfusionMatrix.png', plotSavePath=plotSavePath)
        #buildROC(y_test, y_pred, title='ROC Curve for XGBoost Model', pngname='RandomForestROCCurve.png') # not for multiclass
        buildClassificationReport(y_test, y_pred, title='Classification Report for Random Forest Model', pngname='RandomForestClassificationReport.png', plotSavePath=plotSavePath)

    return model, accuracy





def gbm(X_train, X_test, y_train, y_test, savePlots=True, random_state=1, featureSelecetion=True, plotSavePath=os.path.dirname(currentDirPath) + '/Plots/Model Plots'):
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    if featureSelecetion:
        selection = SelectFromModel(model, prefit=True)
        select_X_train = selection.transform(X_train)
        # retraining model on subset:
        model = GradientBoostingClassifier(random_state=random_state)
        model.fit(select_X_train, y_train)
        X_test = selection.transform(X_test)

        # keep features of interest:
        cols = selection.get_support(indices=True)
        selected_features = list(X_train.iloc[:, cols].columns)
        model.feature_names = selected_features
    else:
        model.feature_names = list(X_train.columns.values)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if savePlots:
        confusionTitle='Confusion Matrix for GBM Model' + " (Accuracy: "+ str(round(accuracy, 2))+")"
        buildConfusionMatrix(y_test, y_pred, title=confusionTitle, pngname='GBMConfusionMatrix.png', plotSavePath=plotSavePath)
        #buildROC(y_test, y_pred, title='ROC Curve for GBM Model', pngname='GBMROCCurve.png') # not for multiclass
        buildClassificationReport(y_test, y_pred, title='Classification Report for GBM Model', pngname='GBMClassificationReport.png', plotSavePath=plotSavePath)

    return model, accuracy




# note: no random_state parameter: probably because it is a closed form formula like a classic linear regression
def naiveBayes(X_train, X_test, y_train, y_test, savePlots=True, featureSelecetion=True, plotSavePath=os.path.dirname(currentDirPath) + '/Plots/Model Plots'):
    model = MultinomialNB()
    model.fit(X_train, y_train)

    if featureSelecetion:
        selection = SelectFromModel(model, prefit=True)
        select_X_train = selection.transform(X_train)
        # retraining model on subset:
        model = MultinomialNB()
        model.fit(select_X_train, y_train)
        X_test = selection.transform(X_test)

        # keep features of interest:
        cols = selection.get_support(indices=True)
        selected_features = list(X_train.iloc[:, cols].columns)
        model.feature_names = selected_features
    else:
        model.feature_names = list(X_train.columns.values)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if savePlots:
        confusionTitle='Confusion Matrix for Naive Bayes Model' + " (Accuracy: "+ str(round(accuracy, 2))+")"
        buildConfusionMatrix(y_test, y_pred, title=confusionTitle, pngname='NaiveBayesConfusionMatrix.png', plotSavePath=plotSavePath)
        buildClassificationReport(y_test, y_pred, title='Classification Report for Naive Bayes Model', pngname='NaiveBayesClassificationReport.png', plotSavePath=plotSavePath)

    return model, accuracy





def glm(X_train, X_test, y_train, y_test, savePlots=True, random_state=1, featureSelecetion=True, plotSavePath=os.path.dirname(currentDirPath) + '/Plots/Model Plots'):
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)

    if featureSelecetion:
        selection = SelectFromModel(model, prefit=True)
        select_X_train = selection.transform(X_train)
        # retraining model on subset:
        model = LogisticRegression(random_state=random_state)
        model.fit(select_X_train, y_train)
        X_test = selection.transform(X_test)

        # keep features of interest:
        cols = selection.get_support(indices=True)
        selected_features = list(X_train.iloc[:, cols].columns)
        model.feature_names = selected_features
    else:
        model.feature_names = list(X_train.columns.values)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if savePlots:
        confusionTitle='Confusion Matrix for GLM Model' + " (Accuracy: "+ str(round(accuracy, 2))+")"
        buildConfusionMatrix(y_test, y_pred, title=confusionTitle, pngname='GLMConfusionMatrix.png', plotSavePath=plotSavePath)
        buildClassificationReport(y_test, y_pred, title='Classification Report for GLM Model', pngname='GLMClassificationReport.png', plotSavePath=plotSavePath)

    return model, accuracy




# automatic feature selection in case of H2O
def h2oAutoMl(X_train, X_test, y_train, y_test, max_runtime_secs= 3600, savePlots=True, random_state=1, plotSavePath=os.path.dirname(currentDirPath) + '/Plots/Model Plots'):
    h2o.init(max_mem_size='16G')
    train=pd.concat([y_train,X_train], axis=1)
    model = H2OAutoML(seed=random_state, max_runtime_secs=max_runtime_secs)
    model.train(x=list(X_train.columns), y=pd.DataFrame(y_train).columns[0], training_frame=h2o.H2OFrame(train))
    y_pred_h2o = model.predict(h2o.H2OFrame(X_test))
    y_pred=y_pred_h2o[0].as_data_frame()

    accuracy = accuracy_score(y_test, y_pred)

    if savePlots:
        confusionTitle='Confusion Matrix for H2o Model' + " (Accuracy: "+ str(round(accuracy, 2))+")"
        buildConfusionMatrix(y_test, y_pred, title=confusionTitle, pngname='H2oConfusionMatrix.png', plotSavePath=plotSavePath)
        buildClassificationReport(y_test, y_pred, title='Classification Report for H2o Model', pngname='H2oClassificationReport.png', plotSavePath=plotSavePath)

    return model.leader, accuracy






def selectModel(X,Y,testSize=0.2, applyXgBoost=True, applyRandomForest=True, applyGbm=True, applyNaiveBayes=True, applyGlm=True, savePlots=True, applyH2oAutoMl=True, H2o_max_runtime_secs=3600, featureSelecetion=True, random_state=1, plotSavePath=os.path.dirname(currentDirPath) + '/Plots/Model Plots'):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testSize, random_state=random_state)

    modelResultsList=[]

    if applyXgBoost:
        xgBoostModel, xgBoostAccuracy = xgBoost(X_train, X_test, y_train, y_test, savePlots=savePlots, featureSelecetion=featureSelecetion, random_state=random_state, plotSavePath=plotSavePath)
    else:
        xgBoostModel, xgBoostAccuracy = 0, 0

    modelResultsList.append((xgBoostModel, xgBoostAccuracy))

    if applyRandomForest:
        randomForestModel, randomForestAccuracy = randomForest(X_train, X_test, y_train, y_test, savePlots=savePlots, featureSelecetion=featureSelecetion, random_state=random_state, plotSavePath=plotSavePath)
    else:
        randomForestModel, randomForestAccuracy = 0, 0

    modelResultsList.append((randomForestModel, randomForestAccuracy))

    if applyGbm:
        gbmModel, gbmAccuracy = gbm(X_train, X_test, y_train, y_test, savePlots=savePlots, featureSelecetion=featureSelecetion, random_state=random_state, plotSavePath=plotSavePath)
    else:
        gbmModel, gbmAccuracy = 0, 0

    modelResultsList.append((gbmModel, gbmAccuracy))

    if applyNaiveBayes:
        naiveModel, naiveAccuracy = naiveBayes(X_train, X_test, y_train, y_test, savePlots=savePlots, featureSelecetion=featureSelecetion, plotSavePath=plotSavePath)
    else:
        naiveModel, naiveAccuracy = 0, 0

    modelResultsList.append((naiveModel, naiveAccuracy))

    if applyGlm:
        glmModel, glmAccuracy = glm(X_train, X_test, y_train, y_test, savePlots=savePlots, featureSelecetion=featureSelecetion, random_state=random_state, plotSavePath=plotSavePath)
    else:
        glmModel, glmAccuracy = 0, 0

    modelResultsList.append((glmModel, glmAccuracy))


    if applyH2oAutoMl:
        h2oModel, h2oAccuracy = h2oAutoMl(X_train, X_test, y_train, y_test, max_runtime_secs=H2o_max_runtime_secs, savePlots=savePlots, random_state=random_state, plotSavePath=plotSavePath)
    else:
        h2oModel, h2oAccuracy = 0, 0

    modelResultsList.append((h2oModel, h2oAccuracy))

    # Take the model with the best accuracy:
    bestModel=max(modelResultsList,key=lambda item:item[1])

    print("\n")
    print("The best model is: " + str(type(bestModel[0])) + " with a accuracy of: " + str(bestModel[1]))
    print("\n")

    # return model object and accuracy
    return bestModel[0], bestModel[1]





# Depending on which model performs better, loading method is different:
def loadBestModel(path):
    bestModelFileName=os.listdir(path)[0]

    if ".pkl" in bestModelFileName:
        with open(path + '/' + bestModelFileName , 'rb') as file:
            predictionModel = pickle.load(file)
    else:
        h2o.init(max_mem_size='16G')
        predictionModel = h2o.load_model(path=path + '/' + bestModelFileName)

    return predictionModel





def predictWithBestModel(model, data):
    if "h2o" in str(type(model)):
        res=model.predict(h2o.H2OFrame(data)).as_data_frame()
        predictions = list(res[res.columns[0]])
    else:
        predictions=model.predict(data[model.feature_names]).tolist()

    return predictions

    # Remove older models placed in directory:
def removeFilesFromDirectory(path):
    files = glob.glob(path + "/*")
    for f in files:
        os.remove(f)




    # Save best model:
def saveModel(model, path=os.path.dirname(currentDirPath) + "/Models/bestPerformingModel"):
    removeFilesFromDirectory(path)
    if "h2o" in str(type(model)):
        h2o.save_model(model=model, path=path,
                       force=True)
    else:
        pickle.dump(model,
                    open(path+"/bestPerformingModel.pkl", 'wb'))





if __name__ == '__main__':
    dummyData=pd.DataFrame(
                        [["A",77,54],
                        ["B",1,1],
                        ["C",50,32],
                        ["A",27,20],
                        ["B",63,73],
                        ["C",31,40],
                        ["A",64,41],
                        ["B",28,98],
                        ["C",38,78],
                        ["A",97,27],
                        ["B",65,57],
                        ["C",99,83],
                        ["A",24,47],
                        ["B",38,32],
                        ["C",90,56],
                        ["A",52,59],
                        ["B",9,13],
                        ["C",42,59],
                        ["A",35,65],
                        ["B",100,64],
                        ["C",31,54],
                        ["A",6,9],
                        ["B",4,19],
                        ["C",96,16],
                        ["A",29,57],
                        ["B",64,12],
                        ["C",59,38],
                        ["A",70,73],
                        ["B",42,93],
                        ["C",37,96],
                        ["A",46,23],
                        ["B",38,59],
                        ["C",38,13],
                        ["A",98,4],
                        ["B",56,17],
                        ["C",29,51],
                        ["A",34,89],
                        ["B",98,40],
                        ["C",92,60],
                        ["A",42,43],
                        ["B",72,37],
                        ["C",35,45],
                        ["A",17,89],
                        ["B",70,56],
                        ["C",87,55],
                        ["A",66,86],
                        ["B",42,49],
                        ["C",29,38],
                        ["A",14,16],
                        ["B",57,6],
                        ["C",65,70],
                        ["A",16,28],
                        ["B",13,59],
                        ["C",7,84],
                        ["A",24,11],
                        ["B",30,92],
                        ["C",24,15],
                        ["A",64,35],
                        ["B",70,45],
                        ["C",24,97],
                        ["A",82,65],
                        ["B",97,22],
                        ["C",92,88],
                        ["A",35,78],
                        ["B",2,93],
                        ["C",67,61],
                        ["A",20,28],
                        ["B",58,83],
                        ["C",86,78],
                        ["A",35,77],
                        ["B",12,17],
                        ["C",94,22],
                        ["A",21,43],
                        ["B",17,11],
                        ["C",79,20],
                        ["A",11,58],
                        ["B",57,8],
                        ["C",19,52],
                        ["A",35,94],
                        ["B",39,93],
                        ["C",11,95],
                        ["A",70,1],
                        ["B",34,32],
                        ["C",30,76],
                        ["A",97,49],
                        ["B",55,57],
                        ["C",19,23],
                        ["A",29,76],
                        ["B",79,23],
                        ["C",62,55],
                        ["A",46,97],
                        ["B",53,7],
                        ["C",87,93],
                        ["A",69,69],
                        ["B",97,39],
                        ["C",44,81],
                        ["A",23,7],
                        ["B",24,15],
                        ["C",26,63],
                        ["A",55,72]
                        ], columns=["Y", "X1", "X2"])

    testModel, testAccuracy = selectModel(X=dummyData[["X1", "X2"]], Y=dummyData["Y"], testSize=0.2, applyXgBoost=True, applyRandomForest=True, applyGbm=True, applyNaiveBayes=True, savePlots=True)