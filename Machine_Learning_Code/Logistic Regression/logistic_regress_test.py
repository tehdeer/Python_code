import pandas as pd
import matplotlib.pyplot as plt

samples = pd.read_csv('testSet.txt', header=None, sep='\t')
data = samples.values
plt.scatter(data[data[:, 2] == 0, 0], data[data[:, 2] == 0, 1], c='red')
plt.scatter(data[data[:, 2] == 1, 0], data[data[:, 2] == 1, 1], c='blue')
plt.show()


