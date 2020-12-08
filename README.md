# training_nlp
# A) QUICKSTART: HOW TO RUN PREDICTIONS
# **1) Open the project in an IDE
# **2) As project interpreter point to poject_path/venv/Scripts/python.exe (required packages are already installed + it avoids conflicts with your own packages)
# **3) Run poject_path/Scripts/api.py
# **4) Go to any browser and visit one of those two urls:

* ** http://localhost:5000/predictauthor?filepath=<path to single txt file>,    Example: http://localhost:5000/predictauthor?filepath=C:/Users/Diego/Documents/NLP training/Data/inputFilesExamples/doc_id00001MWS.txt
* **  http://localhost:5000/predictauthor?folderpath=<path to folder with multiple txt files>,    Example: http://localhost:5000/predictauthor?folderpath=C:/Users/Diego/Documents/NLP training/Data/inputFilesExamples/

# B) PROJECT STRUCTURE:
# **1) Scripts: This is the heart of the project where you will find the train script, 2 predictions scripts (via api and via python function) and the script for descriptive statistics.
#**2) Utils: Utils functions used in the dufferents scripts for reading, claeaning and training data.
#**3) Models: Place where the best performing predictive model as well as TDIDF model is stored for later prediction.
#**4) Data: This doesn't contain the entire training set (wouldn't be best practice) but just some dummy data for unit tests. It also contains list of words used during cleaning (stopwords, contractions, ..).
#**5) Plots: Descriptive plots as well as plots related to model performance


# C) HOW TO RETRAIN THE MODEL WITH YOUR OWN DOCUMENTS:
# **1) Go to the config.ini file at the root of the project
# **2) Change the path to the folder where your own documents are located
# **3) Run poject_path/Scripts/train.py. The resulting model will be saved in poject_path/Models/bestPerformingModel
# **4) Make new predictions by running: poject_path/Scripts/api.py or with poject_path/Scripts/predictFunction.py

# D) A WORD ABOUT SPEED:
If best model is an h2o model, predictions are very slow.... This is because h2O doesn't work with pandas dataframe but with H2O frames. The conversion of data to H2O frame is slow.
If speed is an important aspect of your project, set applyH2oAutoMl to False in the training script (train.py).


