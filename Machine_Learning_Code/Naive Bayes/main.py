from numpy import *
import bayes

# postingList, listClasses = bayes.loadDataSet()  # 创建数据样本
# vocabList = bayes.createVocabList(postingList)  # 根据样本创建词汇表

# 社区网站留言分类
bayes.testingNB()

# 垃圾邮件分类
bayes.spamTest()