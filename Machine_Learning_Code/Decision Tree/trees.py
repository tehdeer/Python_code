'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    """
    dataSet: 训练数据列表
    return: 香农熵
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: 
        currentLabel = featVec[-1] 
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:  # 遍历所有类别
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 根据给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    """
    dataSet: 待划分的数据集，列表
    axis: 给定划分特征的索引
    value: 给定特征的取值
    return: 返回dataset中索引axis位置特征取值为value的样本构成的列表，且列表中剔除掉axis位置的特征
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet



def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures): 
        featList = [example[i] for example in dataSet] 
        uniqueVals = set(featList) 
        newEntropy = 0.0
        for value in uniqueVals: 
            subDataSet = splitDataSet(dataSet, i, value) 
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy 
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain  
            bestFeature = i 
    return bestFeature  



def majorityCnt(classList):
    """
    classList: 划分后的节点只剩下标签列表
    return: 类别众数
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建树，递归函数
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0] 
    if len(dataSet[0]) == 1: 
        return majorityCnt(classList) 
    bestFeat = chooseBestFeatureToSplit(dataSet) 
    bestFeatLabel = labels[bestFeat] 
    myTree = {bestFeatLabel: {}} 
    del (labels[bestFeat]) 
    featValues = [example[bestFeat] for example in dataSet] 
    uniqueVals = set(featValues)
    for value in uniqueVals:  
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 使用决策树分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
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
