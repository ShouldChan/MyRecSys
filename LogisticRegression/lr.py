# coding:utf-8
import random
import math
import numpy as np


def makeData():
    maxDataNum = 100

    dataList = []

    for i in range(0, maxDataNum / 2):
        x = random.uniform(-3, 3)
        y = random.uniform(5, 15)

        label = 0
        # print x, y, label
        dataList.append([x, y, label])

    for i in range(0, maxDataNum / 2):
        x = random.uniform(-2, 3)
        y = random.uniform(-5, 8)

        label = 1
        # print x, y, label
        dataList.append([x, y, label])

    return dataList


def loadDataset(dataList):
    dataMat = []
    labelMat = []

    for line in dataList:
        dataMat.append([1.0, float(line[0]), float(line[1])])
        labelMat.append(int(line[2]))

    return dataMat, labelMat


def sigmoid(x):
    m, n = x.shape
    sig_x = np.zeros((m, n))
    for i in range(m):
        sig_x[i] = 1.0 / (1 + math.exp(-x[i]))

    return sig_x


def gradAscent(dataMat, labelMat):
    dataMatrix = np.mat(dataMat)
    # transpose() 转置
    labelMatrix = np.mat(labelMat).transpose()

    m, n = np.shape(dataMatrix)
    print "dataMatrix shape:", dataMatrix.shape

    alpha = 0.0001
    maxIters = 1000

    weights = np.ones((n, 1))
    print "weight shape:", weights.shape

    for k in range(maxIters):
        h = sigmoid(dataMatrix * weights)
        error = (labelMatrix - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# 画出决策边界
def plotBestFit(dataMat, labelMat, weights):
    import matplotlib.pyplot as plt

    n = np.shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i] == 1):
            xcord1.append(dataMat[i][1])
            ycord1.append(dataMat[i][2])
        else:
            xcord2.append(dataMat[i][1])
            ycord2.append(dataMat[i][2])

    fig = plt.figure()

    ax = fig.add_subplot(111)  # 111定义画布大小
    ax.scatter(xcord1, ycord1, s=10, c="red", marker="s")
    ax.scatter(xcord2, ycord2, s=10, c="green")
    x = np.arange(-3.0, 3.0, 0.1)  # arange(start, end, step)
    # y = (-weights[0] - weights[1] * x) / weights[2]
    y = (-(float)(weights[0][0]) - (float)(weights[1][0]) * x) / (float)(weights[2][0])

    print weights.shape
    print weights[0][0]
    print weights[0][0].shape

    ax.plot(x, y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


if __name__ == '__main__':
    dataList = makeData()
    dataMat, labelMat = loadDataset(dataList)

    weights = gradAscent(dataMat, labelMat)
    print weights
    plotBestFit(dataMat, labelMat, weights)
