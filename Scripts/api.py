#example usage with single file: http://localhost:5000/predictauthor?filepath=C:/Users/Diego/Documents/NLP%20training/Data/inputFilesExamples/doc_id00001MWS.txt
#example usage with single file: http://localhost:5000/predictauthor?folderpath=C:/Users/Diego/Documents/NLP%20training/Data/inputFilesExamples/

 # Important note: if best model is an h2o model, predictions are very slow.... If speed is an important aspect of your project, set applyH2oAutoMl to False in the training script (train.py).

from flask import Flask, request
import predictFunction
import json
import modelSelection
import os

currentDirPath = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
predictionModel = modelSelection.loadBestModel(os.path.dirname(currentDirPath) + "/Models/bestPerformingModel")


@app.route("/predictauthor")
def returnPrediction():
    filePath = request.args.get('filepath')
    folderPath = request.args.get('folderpath')

    return json.loads(predictFunction.predictAuthor(predictionModel=predictionModel, filePath=filePath, folderPath=folderPath))

if __name__ == "__main__":
    app.run(port=5000)