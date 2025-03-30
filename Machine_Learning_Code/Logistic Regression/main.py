from logRegres import *
from numpy import *


# ******自建二维特征数据集分析不同的优化方法回归效果***********
dataArr, labelMat = loadDataSet()

# ******************************梯度下降法**********************************
weights = gradAscent(dataArr, labelMat)
print("梯度下降法")
plotBestFit(weights.getA())

# ****************************随机梯度下降法**********************************
print("随机梯度下降法")
weights2 = stocGradAscent0(array(dataArr), labelMat)
plotBestFit(weights2)


# # ****************************改进的随机梯度下降法**********************************
print("一次随机梯度下降很可能是得不到最优结果的")
print("改进的随机梯度下降法")
weights3 = stocGradAscent1(array(dataArr), labelMat)
plotBestFit(weights3)


# ***************************从疝气病症预测马的存活**********************************
print("\n从疝气病症预测马的存活：")
multiTest()
