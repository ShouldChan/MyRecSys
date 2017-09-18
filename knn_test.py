# coding:utf-8

import numpy as np 
import operator

'''
    parameters:
    inX: 用于分类的输入向量
    dataset: 训练样本集
    labels：标签向量
    k：最近邻数目
    （标签向量的元素个数和矩阵dataset的行数相等）
'''
def knn(inX,dataset,labels, k):
    # get the size of dataset
    datasetSize = dataset.shape[0]
    print datasetSize
    # make the difference matrix
    diffMat = np.tile(inX, (datasetSize,1)) - dataset
    print diffMat
    # make the difference **2
    sqDiffMat = diffMat**2
    print sqDiffMat
    # cumulate the diffmat to calc the distances
    sqDistances = sqDiffMat.sum(axis=1)
    print sqDistances
    distances = sqDistances**0.5
    print distances
    # sort 
    sortedDistIndices = distances.argsort()
    print sortedDistIndices
    # 用来累计这k个点所在类别的出现频率（次数）
    # 返回前k个点出现的频率最高的类别作为当前点的预测分类
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        print sortedDistIndices[i]
        print voteIlabel
        # 获取频数字典中该标签出现的个数
        # dict.get(key,0)指定返回键值，如果在字典里没有则返回0
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
        print classCount
    # 对classcount进行从大到小排序
    sortedClassCount = sorted(classCount.iteritems(), \
        key=operator.itemgetter(1), reverse=True)
    print sortedClassCount[0] #{'B':2}
    print sortedClassCount[0][0]  #B
    return sortedClassCount[0][0]


def create_dataset():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

group, labels = create_dataset()
knn([0,0],group,labels,3)