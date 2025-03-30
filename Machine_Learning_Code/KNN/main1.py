import kNN
from numpy import *

# 电影题材分类
film = {'art': '文艺片', 'action': '动作片'}
trainData, labels = kNN.createDataSet()
testData = array([24, 67])
k = 3
pred_label = kNN.classify0(testData, trainData, labels, k)
print("该电影为：%s"%film[pred_label])


