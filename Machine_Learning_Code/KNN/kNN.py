'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label
'''
from numpy import *
import operator
from os import listdir

classes = ['didntLike', 'smallDoses', 'largeDoses']


# k-近邻算法
def classify0(inX, dataSet, labels, k):
    """
    inX: ndarray:(3,) 测试样本
    dataSet: ndarray(n,3) 训练样本
    labels: list
    k: cluster个数
    return:返回k个最近邻样本中出现次数最多的类别编号
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2 
    sqDistances = sqDiffMat.sum(axis=1) 
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort() 
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
    # group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # labels = ['A', 'A', 'B', 'B']
    group = array([[1, 101], [5, 89], [12, 97], [108, 5], [112, 9], [115, 8]]) 
    labels = ['art', 'art', 'art', 'action', 'action', 'action']
    return group, labels


# 将文本转换为numpy数组
def file2matrix(filename, train=True):
    """
    filename: 数据文本路径
    train: 判断是否是训练样本
    return:
    """
    with open(filename, 'r') as fr:
        data = fr.readlines()
        numberOfLines = len(data)
        returnMat = zeros((numberOfLines, 3))
        classLabelVector = []
        index = 0
        if train:
            for line in data:
                line = line.strip()
                listFromLine = line.split('\t') 
                returnMat[index, :] = listFromLine[0:3]
                classLabelVector.append(int(listFromLine[-1]))
                index += 1
        else:
            for line in data:
                line = line.strip()
                listFromLine = line.split('\t')
                returnMat[index, :] = listFromLine[0:3]
                classLabelVector.append(classes.index(listFromLine[-1]))
                index += 1

        return returnMat, classLabelVector


# 归一化特征值
def autoNorm(dataSet):
    """
    dataSet:ndarray:(nSample, nFea)
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1)) 
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 分类器测试
def datingClassTest():
    hoRatio = 0.50 
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt', train=False) 
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0] 
    numTestVecs = int(m * hoRatio) 
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])) 
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0 
    print("the total error rate is: %f" % (errorCount / float(numTestVecs))) 
    print('测试样本共{}个，错误分类{}个'.format(numTestVecs,errorCount))


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0] 
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))
