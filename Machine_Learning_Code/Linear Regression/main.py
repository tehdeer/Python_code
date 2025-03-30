import regression
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

# 任务1，一元线性回归并可视化
# xArr, yArr = regression.loadDataSet('ex0.txt')
# xMat = mat(xArr)  # 列表转numpy矩阵
# yMat = mat(yArr)  # 列表转numpy矩阵
# # 绘制训练样本散点图
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])  # xMat[:, 1]为n行1列，.A等同np.asarray()  训练样本散点图
# plt.show()
#
# ws = regression.standRegres(xArr, yArr)  # 利用线性回归方法求解最佳拟合直线参数
# yHat = xMat * ws  # 预测值
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])  # xMat[:, 1]为n行1列，.A等同np.asarray()  训练样本散点图
# xCopy = xMat.copy()
# xCopy.sort(0)  # 对xCopy每列按顺序排列
# yHat = xCopy * ws  # 特征值排序后样本的预测值
# ax.plot(xCopy[:, 1], yHat)  # 画回归直线
# plt.show()



# 任务2，预测鲍鱼年龄
# abX, abY = regression.loadDataSet('abalone.txt')
# ridgeWeights = regression.ridgeTest(abX,abY)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ridgeWeights)  # 对每个系数画折线图，正则化项系数从小到大的顺序
# plt.show()


# 任务3，预测空气质量指数
# aqi_data = pd.read_csv('aqi2.csv')
# fea_data = aqi_data.iloc[:,1:]
# fea_data.insert(0,'constant',1)
# y_data = aqi_data.iloc[:,0]
#
# xArr = fea_data.values
# yArr = y_data.values
#
# xMat = mat(xArr)
# yMat = mat(yArr)
# print('ok')
# ws = regression.standRegres(xArr, yArr)  # 利用线性回归方法求解最佳拟合直线参数
# print(ws)
# yHat = xMat * ws  # 预测值
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(yHat.flatten().A[0], yMat.T[:, 0].flatten().A[0])  # xMat[:, 1]为n行1列，.A等同np.asarray()  训练样本散点图
# min_val = yHat.flatten().A[0].min()
# max_val = yHat.flatten().A[0].max()
# val = arange(min_val, max_val)
# ax.plot(val,val,'r')  # 画回归直线
# plt.show()
