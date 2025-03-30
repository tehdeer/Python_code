import kNN


# 约会对象分类
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')  # 读取数据
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)  # 对数据归一化
kNN.datingClassTest()  # 分类器测试


