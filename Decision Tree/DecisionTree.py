from math import log
import operator
#preint something
def createDataSet():
    dataSet = [[0, 1, 1, 'yes'],
               [0, 1, 0, 'no'],
               [1, 0, 1, 'no'],
               [1, 1, 1, 'no'],
               [0, 1, 0, 'no'],
               [0, 0, 1, 'no'],
               [1, 0, 1, 'no'],
               [1, 1, 0, 'no']]
    labels = ['chetan', 'goa', 'friends']
    # change to discrete values
    return dataSet, labels
#print("A")
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)


        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        """
        print("feature : " + str(i))
        print("baseEntropy : "+str(baseEntropy))
        print("newEntropy : " + str(newEntropy))
        print("infoGain : " + str(infoGain))
        """
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    # extracting data
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    # use Information Gain
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    #build a tree recursively
    myTree = {bestFeatLabel: {}}
    #print("myTree : "+labels[bestFeat])
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    #print("featValues: "+str(featValues))
    uniqueVals = set(featValues)
    #print("uniqueVals: " + str(uniqueVals))
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        #print("subLabels"+str(subLabels))
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        #print("myTree : " + str(myTree))
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())
    print("fistStr : "+str(firstStr))
    secondDict = inputTree[firstStr[0]]
    print("secondDict : " + str(secondDict))
    featIndex = featLabels.index(firstStr[0])
    print("featIndex : " + str(featIndex))
    key = testVec[featIndex]
    print("key : " + str(key))
    valueOfFeat = secondDict[key]
    print("valueOfFeat : " + str(valueOfFeat))
    if isinstance(valueOfFeat, dict):
        #print("is instance: "+str(valueOfFeat))
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        #print("is Not instance: " + valueOfFeat)
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

# collect data
myDat, labels = createDataSet()

#build a tree
mytree = createTree(myDat, labels)
print(mytree)

print("Now I can recognize chetan's with his friends photos ")

#run test
#print("1")

answer = classify(mytree, ['chetan', 'goa', 'friends'], [0, 1, 1])
print("Hi, the answer is "+ answer + ", it is chetan's with friends photo")
#print("qerty")
# test 2
answer = classify(mytree, ['chetan', 'goa', 'friends'], [1, 1, 1])
print("Hi, the answer is "+ answer + ", it is not chetan's with friends photo")