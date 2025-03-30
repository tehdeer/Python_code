import trees
import treePlotter

# 计算熵
myDat, labels = trees.createDataSet()
shannonent = trees.calcShannonEnt(myDat)
print(shannonent)


# 对眼镜数据决策树分类
fr = open('lenses.txt')
lenses_data = []
lines = fr.readlines()
for line in lines:
    lenses_data.append(line.strip().split('\t'))
featLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses_data, featLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)
testVec = ['young', 'myope', 'no', 'reduced']
featLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
classLabel = trees.classify(lensesTree, featLabels, testVec)
print('预测类别为：%s' % classLabel)
