import pandas as pd
import glob
import ntpath

def readTextFiles(folderPath=None, filePath=None, lower=True, parseIDAuthorName=True):
    if (folderPath) and (filePath):
        raise ValueError('folderPath and filePath can not be both filled. Put one of the two to None!')

    if folderPath:
        allFiles = glob.glob(folderPath + "/*.txt")
        li=[]
        for fileName in allFiles:
            text = open(fileName, "r", errors="ignore").read()
            df=pd.DataFrame([(text, fileName)], columns=["text", "path"])
            li.append(df)
        if len(li)==0:
            raise ValueError('No text files found in folder: '+folderPath+'. Sure it is the right folder?')
        data = pd.concat(li)

    if filePath:
        text = open(filePath, "r").read()
        data=pd.DataFrame([(text, filePath)], columns=["text", "path"])

    if parseIDAuthorName:
        data["fileName"] = data["path"].apply(lambda x: ntpath.basename(x))
        data["id"] = data["fileName"].str[6:11]
        data["author"] = data["fileName"].str[11:].str.replace(".txt", "")

    if lower:
        data["text"]=data["text"].str.lower()

    return data.reset_index(drop=True)

if __name__ == '__main__':
    import os
    currentDirPath = os.path.dirname(os.path.realpath(__file__))
    folderPath=os.path.dirname(currentDirPath)+"/Data/inputFilesExamples"

    print("Opening all files  of one folder:")
    print(readTextFiles(folderPath=folderPath, lower=True))
    print("\n")

    print("Opening one single file:")
    filePath = folderPath+"/doc_id00003testMultiLine.txt"
    print(readTextFiles(filePath=filePath, lower=True))