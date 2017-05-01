# encoding=utf-8
from __future__ import division
from math import *
import numpy as np


def Huffman_coding(vec):
    code = 0
    for i in range(len(vec)):
        code += (2 ** (len(vec) - i - 1)) * int(vec[i])
    return code


# 计算地理空间的距离
def haversine(geo1, geo2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1 = geo1[0]
    lat1 = geo1[1]
    lon2 = geo2[0]
    lat2 = geo2[1]
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

# test haversine
# geo1=[52.51777136760647,13.433514373863828]
# geo2=[52.55495792389866,13.417484372316503]
# a=haversine(geo1,geo2)
# print a

def MillerCoor(lat, lon):
    X = 6381372 * pi * 2
    Y = X / 2
    mill = 2.3
    arcx = lat * pi / 180
    arcy = lon * pi / 180
    y = 1.25 * log(tan(0.25 * pi + 0.4 * arcy))
    x = (X / 2) + (X / (2 * pi)) * arcx
    y = (Y / 2) - (Y / (2 * mill)) * y
    return x, y


# calculate Euclidean distance
# def distance(vector1, vector2):
#     dist1 = np.sqrt(np.sum(np.square(vector1 - vector2)))   #计算欧式距离的第一种写法
#     # dist2 = np.linalg.norm(vector1 - vector2)   #计算欧式距离的第二种写法
#     return dist1
#     # return sqrt(sum(power(vector2 - vector1, 2)))  #作者源码的写法

# test distance ok...
# v1=[1,2]
# v2=[2,3]
# v1=np.array(v1)
# v2=np.array(v2)
# print distance(v1,v2)

def evaluation(lis1, lis2, error):
    precision = 0
    num = 0
    for i in range(len(lis2)):
        num += 1
        if abs(lis2[i] - lis1[i]) <= error:
            precision += 1
    pre = precision / num
    return pre


# print MillerCoor(12,34)
def mean_var(List):
    # print "mean_var"
    narray = np.array(List)
    # print len(narray)
    dimen = int(narray.size / len(narray))
    # print intimen)
    var_s = 0
    if dimen > 1:
        for l in range(dimen):
            lis = []
            # print "l",l
            for col in range(len(narray)):
                # print narray[col][l]
                lis.append(narray[col][l])
            mean, var, mid = mean_var_single(lis)
            # print var
            var_s += var
            # print var_s
        var_s = var_s / dimen
    else:
        mean, var, mid = mean_var_single(narray)
        var_s = var
    # print "inner",var_s
    return var_s


def mean_var_single(List):
    narray = np.array(List)
    sum1 = narray.sum()
    # print sum1
    narray2 = narray * narray
    sum2 = narray2.sum()
    mean = sum1 / len(List)
    # print mean
    try:
        var = sqrt(sum2 / len(List) - mean ** 2)
    except Exception, e:
        var = 0
    # print "sibf",var
    List = sorted(List)
    if len(List) % 2 == 1:
        mid = List[int(len(List) / 2)]
    else:
        mid = (List[int(len(List) / 2)] + List[int(len(List) / 2 - 1)]) / 2
    return mean, var, mid
