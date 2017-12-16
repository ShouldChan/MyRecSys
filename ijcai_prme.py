# coding: utf-8
'''
Personalized Ranking Metric Embedding (PRME)
====

This is the implementation of the paper:
Feng, S. et al. Personalized ranking metric embedding for next new POI recommendation. In Proc. IJCAI.

How to use the program
----------------------------
Step1: Create directories as follows:

Step2: Put your data in data_dir, formated as (See sample_data.txt for more details):

    user_id current_poi_id lat lon checkin_time next_poi lat lon checkin_time

Setp3: Specify following variables in the program:
    1. dataset.
    2. file_suffix.
    3. time_format.
    4. user_num & poi_num.
    5. other parameters.

Output:
Three matrices will be stored in model_dir: LP, LS and UP, representing X^P(L), X^S(L) and X^P(u) in the paper.
----------------------------
'''

import time
import random
import numpy as np
import operator

from util import Util

data_dir = "./data/"
result_dir = "./result/"
model_dir = "./model/"

file_suffix = ".txt"

time_format = "%Y-%m-%d %H:%M:%S"

K = 60            # latent dimensionality /  the number of dimensions
tau = 3600 * 12    # time difference threshold
gamma = 0.005      # learning rate
lamda = 0.03      # regularization / regularization term
alpha = 0.2       # linear combination  /   component weight
dis_coef = 0.25   # distance coefficient

max_iters = 30   # max iterations

# ┏┓   ┏┓
# ┏┛┻━━━┛┻┓
# ┃    ☃   ┃
# ┃ ┳┛ ┗┳ ┃
# ┃   ┻    ┃
# ┗━┓     ┏━┛
# ┃     ┗━━━┓
# ┃  神兽保佑  ┣┓
# ┃　永无BUG！ ┏┛
# ┗┓┓┏━┳┓┏┛
# ┃┫┫  ┃┫┫
# ┗┻┛  ┗┻┛

def read_userpoiNum():
    user_set = set()
    poi_set = set()
    with open(data_dir+"gowalla.txt",'r') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            user, poi=temp[0],temp[1]
            user_set.add(user)
            poi_set.add(poi)
    user_num, poi_num = len(user_set),len(poi_set)
    return user_num,poi_num

user_num, poi_num=read_userpoiNum()
print("user_num:%d\tpoi_num:%d"%(user_num,poi_num))

def make_trainingset():
    trainingset = []
    with open(data_dir+"gowalla_train.txt",'r') as fread:
        lines = fread.readlines()
        for i in range(len(lines)):
            temp = lines[i].strip().split('\t')
            user, poi, latlon, seconds = temp[0], temp[1], temp[2], int(temp[3])
            lat,lon = latlon.strip().split(',')[0],latlon.strip().split(',')[1]
            if i != len(lines)-1:
                temp_nx = lines[i].strip().split('\t')
                user_nx, poi_nx, latlon_nx, seconds_nx = temp_nx[0], temp_nx[1], temp_nx[2], int(temp_nx[3])
                lat_nx,lon_nx = latlon_nx.strip().split(',')[0],latlon_nx.strip().split(',')[1]
            if user == user_nx:
                interval = seconds_nx - seconds
                trainingset.append([user,poi,lat,lon,poi_nx,lat_nx,lon_nx,interval])
    print(len(trainingset))
    with open(data_dir+"nextpoi_gowalla_train.txt",'w') as fw:
        for each_data in trainingset:
            user,poi,lat,lon,poi_nx,lat_nx,lon_nx,interval = each_data
            fw.write(str(each_data))

make_trainingset()

def make_testset():
    testset = []
    with open(data_dir+"gowalla_test.txt",'r') as fread:
        lines = fread.readlines()
        for i in range(len(lines)):
            temp = lines[i].strip().split('\t')
            user, poi, latlon, seconds = temp[0], temp[1], temp[2], int(temp[3])
            lat,lon = latlon.strip().split(',')[0],latlon.strip().split(',')[1]
            if i != len(lines)-1:
                temp_nx = lines[i].strip().split('\t')
                user_nx, poi_nx, latlon_nx, seconds_nx = temp_nx[0], temp_nx[1], temp_nx[2], int(temp_nx[3])
                lat_nx,lon_nx = latlon_nx.strip().split(',')[0],latlon_nx.strip().split(',')[1]
            if user == user_nx:
                interval = seconds_nx - seconds
                testset.append([user,poi,lat,lon,poi_nx,lat_nx,lon_nx,interval])
    print(len(testset))
    with open(data_dir+"nextpoi_gowalla_test.txt",'w') as fw:
        for each_data in testset:
            fw.write(str(each_data))

make_testset()
