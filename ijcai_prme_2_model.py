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

dataset_name = "SIN"

data_dir = "./data/"+dataset_name+"/"
result_dir = "./result/"+dataset_name+"/"
model_dir = "./model/"+dataset_name+"/"


time_format = "%Y-%m-%d %H:%M:%S"

K = 60            # latent dimensionality /  the number of dimensions
tau = 3600 * 6    # time difference threshold
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

def get_userpoiNum():
    user_set = set()
    poi_set = set()
    with open(data_dir+dataset_name+".txt",'r') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            user, poi=temp[0],temp[1]
            user_set.add(user)
            poi_set.add(poi)
    user_num, poi_num = len(user_set),len(poi_set)
    return user_num,poi_num


def get_locations():
    locations = {}

    with open(data_dir+dataset_name+"_poiset.txt",'r') as fr:
        lines = fr.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            poi, lat, lon = int(temp[0]),float(temp[1]),float(temp[2])
            # print(poi,lat,lon)
            locations[poi] = [lat,lon]
    return locations

# get_locations()

def read_training_data():
    train_data = []
    visits = set()

    with open(data_dir+"nextpoi_"+dataset_name+"_train.txt",'r') as fr:
        lines = fr.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            u, lc, li = int(temp[0]), int(temp[1]), int(temp[4])
            time_irrelevance = int(temp[7]) > tau
            train_data.append([u,lc,li,time_irrelevance])
            visits.add((u,lc,li))
    return train_data, visits

def learning(train_data, visits, locations, cont):
    user_num, poi_num = get_userpoiNum()
    print("user_num:%d\tpoi_num:%d"%(user_num,poi_num))

    if not cont:
        UP = np.random.normal(0.0, 0.01, (user_num, K))
        LS = np.random.normal(0.0, 0.01, (poi_num, K))
        LP = np.random.normal(0.0, 0.01, (poi_num, K))
    else:
        UP = np.load(model_dir + "UP.npy")
        LS = np.load(model_dir + "LS.npy")
        LP = np.load(model_dir + "LP.npy")

    try:
        for iteration in range(max_iters):
            log_likelihood = 0.0
            #将train_data中的元素随机打乱
            random.shuffle(train_data)

            t = time.time()
            for each_data in train_data:
                try:
                    u, lc, li, time_irrelevance = each_data
                    #lj为用户u没访问过的poi unvisited
                    lj = li
                    while (u, lc, lj) in visits or locations.get(lj) == None:
                        # sample(seq, n) 从序列poi_ids中选择1个随机且独立的元素；
                        #lj = random.sample(poi_ids, 1)[0]
                        #lj = random.randint(0, poi_num)
                        #if locations.get(lj) == None:
                        lj = random.randint(0, poi_num)
                        #lj += 1

                    #print locations.get(lj), "---->"
                    if time_irrelevance:
                        #计算范数
                        Di = np.linalg.norm(UP[u] - LP[li]) ** 2
                        Dj = np.linalg.norm(UP[u] - LP[lj]) ** 2
                        z = Dj - Di

                        log_likelihood += np.log(util.sigmoid(z))

                        UP[u] += gamma * ((1 - util.sigmoid(z)) * 2 * (LP[li] - LP[lj]) - 2 * lamda * UP[u])
                        LP[li] += gamma * ((1 - util.sigmoid(z)) * 2 * (UP[u] - LP[li]) - 2 * lamda * LP[li])
                        LP[lj] += gamma * (- (1 - util.sigmoid(z)) * 2 * (UP[u] - LP[lj]) - 2 * lamda * LP[lj])

                    else:
                        wci = (1.0 + util.dist(locations[lc], locations[li])) ** dis_coef
                        wcj = (1.0 + util.dist(locations[lc], locations[lj])) ** dis_coef
                        Di = wci * (alpha * np.linalg.norm(UP[u] - LP[li]) ** 2 + (1 - alpha) * np.linalg.norm(LS[lc] - LS[li])**2)
                        Dj = wcj * (alpha * np.linalg.norm(UP[u] - LP[lj]) ** 2 + (1 - alpha) * np.linalg.norm(LS[lc] - LS[lj])**2)
                        z = Dj - Di

                        log_likelihood += np.log(util.sigmoid(z))

                        UP[u] += gamma * ((1 - util.sigmoid(z)) *
                                          2 * alpha * ((wcj - wci) * UP[u] + (wci * LP[li] - wcj * LP[lj])) -
                                          2 * lamda * UP[u])
                        LP[li] += gamma * ((1 - util.sigmoid(z)) * 2 * alpha * wci * (UP[u] - LP[li]) - 2 * lamda * LP[li])
                        LP[lj] += gamma * (- (1 - util.sigmoid(z)) * 2 * alpha * wcj * (UP[u] - LP[lj]) - 2 * lamda * LP[lj])
                        LS[lc] += gamma * ((1 - util.sigmoid(z)) *
                                           2 * (1 - alpha) * ((wcj - wci) * LS[lc] + (wci * LS[li] - wcj * LS[lj])) -
                                           2 * lamda * LS[lc])
                        LS[li] += gamma * ((1 - util.sigmoid(z)) * 2 * (1 - alpha) * wci * (LS[lc] - LS[li]) - 2 * lamda * LS[li])
                        LS[lj] += gamma * (- (1 - util.sigmoid(z)) * 2 * (1 - alpha) * wcj * (LS[lc] - LS[lj]) - 2 * lamda * LS[lj])

                except OverflowError:
                    pass
                    # print "Calculation failed. #%d" % error_cnt

            print("Iter: %d    likelihood: %f    elapsed: %fs" % (iteration, log_likelihood, time.time() - t))
    finally:
        np.save(model_dir + "UP", UP)
        np.save(model_dir + "LP", LP)
        np.save(model_dir + "LS", LS)
        print("Model saved...")

def read_matrix(train_data, visits, locations):
    user_num, poi_num = get_userpoiNum()
    print("user_num:%d\tpoi_num:%d"%(user_num,poi_num))

    UP = np.load(model_dir + "UP.npy")
    LS = np.load(model_dir + "LS.npy")
    LP = np.load(model_dir + "LP.npy")

    DG = []
    t = time.time()
    for i in range(len(train_data) - 1):
        u, lc, li, time_irrelevance = train_data[i]
        u2, lc2, li2, time_irrelevance2 = train_data[i + 1]
        print(i)
        if u != u2:
            lj = li
            for lj in range(poi_num):
                #print lj
                # if (u, lc, lj) not in visits and locations.get(lj) != None:
                w = (1.0 + util.dist(locations[lc], locations[lj])) ** dis_coef
                DP = np.linalg.norm(UP[u] - LP[lj]) ** 2
                DS = np.linalg.norm(LS[lc] - LS[lj]) ** 2
                distance = w * (alpha * DP + (1 - alpha) * DS)
                DG.append([u, lj, distance])
    DG.sort(key=operator.itemgetter(0, 2))
    with open(result_dir + "not_newpoi_dist_list_prme.txt", 'w') as n:
        for [u, lj, distance] in DG:
            n.write(str(u + 1) + "\t" + str(lj) + "\t" + str(distance) + "\n")
    print("read_matrix elapsed: %f" % (time.time() - t))

def main():

    t = time.time()
    train_data, visits = read_training_data()
    locations = get_locations()
    print("Data Loaded... Elapsed", time.time() - t)

    # learning(train_data, visits, locations, False)
    # print("learning over...", time.time() - t)

    read_matrix(train_data, visits, locations)
    print("make distance_list ok...Elapsed", time.time() - t)


if __name__ == '__main__':
    util = Util()
    main()