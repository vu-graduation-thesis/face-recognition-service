import os
from os import listdir
from os.path import isdir, join
import face_recognition
import numpy
from tabulate import tabulate


trainingPath = "./training-data"
datasetPath = "./dataset"

knownLabel = []
knownDescriptor = []

unknownLabel = []
unknownDescriptor = []

# function to get folder
def getFolderDataset(datasetPath):
    return [f for f in listdir(datasetPath) if isdir(join(datasetPath, f))]


# init knownLabel and knownDescriptor
def training():
    global knownLabel
    global knownDescriptor
    knownLabel = []
    knownDescriptor = []
    for folder in getFolderDataset(trainingPath):
        for file in listdir(join(trainingPath, folder)):
            image = face_recognition.load_image_file(
                join(trainingPath, folder, file))
            faceEncoding = face_recognition.face_encodings(image)[0]
            knownLabel.append(folder)
            knownDescriptor.append(faceEncoding)


# function to get unknown image and assign label
def getUnknownImage():
    global unknownLabel
    global unknownDescriptor
    unknownLabel = []
    unknownDescriptor = []
    for folder in getFolderDataset(datasetPath):
        for file in listdir(join(datasetPath, folder)):
            image = face_recognition.load_image_file(
                join(datasetPath, folder, file))
            faceEncodings = face_recognition.face_encodings(image)
            if len(faceEncodings) > 0:
                faceEncoding = faceEncodings[0]
            else:
                faceEncoding = numpy.zeros(128)
            unknownLabel.append(folder)
            unknownDescriptor.append(faceEncoding)


# function to statitistic with threshold
def statistic():
    global knownLabel
    global knownDescriptor
    global unknownLabel
    global unknownDescriptor

    tableData = []
    tableHeader = ["Threshold", "Correct prediction", "Wrong prediction", "Unknown detect", "Number of predictions"]
    
    print("\n\nNumber of faces recognized: {number}".format(number = len(knownLabel)))
    print("Number of unrecognized faces: {number}\n\n".format(number = len(unknownLabel)))

    for i in range(20, 80, 5):
        threshold = i/100

        correctDict = {} # key: knownLabel, value: number of correct(with threshold & match knownLabel)
        mistakeDict = {} # key: knownLabel, value: number of mistake(with threshold & not match knownLabel)
        unknownDetectCount = 0
        totalImages = 0


        for kIndex, knownFaceEncoding in enumerate(knownDescriptor):
            knownFaceLabel = knownLabel[kIndex]
            correctDict[knownFaceLabel] = 0
            mistakeDict[knownFaceLabel] = 0
            for uIndex, unknownFaceEncoding in enumerate(unknownDescriptor):
                unknownFaceLabel = unknownLabel[uIndex]
                result = (numpy.linalg.norm(knownFaceEncoding - unknownFaceEncoding) < threshold)
                result = bool(result)
                totalImages += 1

                if result:
                    if knownFaceLabel == unknownFaceLabel:
                        correctDict[knownFaceLabel] += 1
                    else:
                        mistakeDict[knownFaceLabel] += 1
                else:
                    unknownDetectCount += 1

        tableData.append(["{percent}%".format(percent = 100 - i), sum(correctDict.values()), sum(mistakeDict.values()), unknownDetectCount, totalImages])

        
    print(tabulate(tableData, tableHeader, tablefmt="grid"))        

# training model        
training()

getUnknownImage()

statistic()
