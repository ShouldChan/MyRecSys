# coding:utf-8
import numpy as np 
import pandas as pd 
import csv
import datetime
import random
import time
from util import Util
import operator

time_format = '%Y-%m-%d %H:%M:%S'
poi_num = 110528
user_num = 349694
K = 60            # latent dimensionality /  the number of dimensions
tau = 3600 * 6    # time difference threshold
gamma = 0.005      # learning rate
lamda = 0.03      # regularization / regularization term
alpha = 0.2       # linear combination  /   component weight
dis_coef = 0.25   # distance coefficient

max_iters = 30   # max iterations
model_dir = './'

def read_training_data():
    train_data = []
    user_dict = {}
    user_list = []
    loc_dict = {}
    visits = set()
    # 给地点进行编号
    with open('./latlon_v2.txt', 'rb') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            locid,hash_locid = str(temp[0]),str(temp[1])
            loc_dict[hash_locid] = locid
    # 给用户id进行编号
    with open('./userid.txt','rb') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            u, userid = str(temp[0]),str(temp[1])
            user_dict[userid] = u
            user_list.append(userid)
    # 读取训练数据集
    with open('./train_next.csv','rb') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split(',')
            userid, starttime, geohashed_start_loc,  \
            geohashed_end_loc = str(temp[1]), str(temp[4]), \
            str(temp[5]), str(temp[6])
            u,lc,li = user_dict[userid],loc_dict[geohashed_start_loc],loc_dict[geohashed_end_loc]
            time_irrelevance = False
            train_data.append([u, lc, li, time_irrelevance])
            visits.add((u, lc, li))
    print 'read train ok...'
    return train_data, visits, user_dict, loc_dict, user_list


def get_locations():
    locations = {}
    with open('./latlon_v2.txt','rb') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            locations[int(temp[0])] = [float(temp[2]), \
            float(temp[3])]
    return locations


def learning(train_data, visits, locations, cont):
    # load matrix from ./model/ to continue the training...

    if not cont:
        UP = np.random.normal(0.0, 0.01, (user_num, K))
        LS = np.random.normal(0.0, 0.01, (poi_num, K))
        LP = np.random.normal(0.0, 0.01, (poi_num, K))
    else:
        UP = np.load(model_dir + "UP.npy")
        LS = np.load(model_dir + "LS.npy")
        LP = np.load(model_dir + "LP.npy")

    error_cnt = 0
    # range函数 在0-poi_num（不包括poi_num这个值）中产生一个数字序列
    #poi_ids = range(poi_num)
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

            print "Iter: %d    likelihood: %f    elapsed: %fs" % (iteration, log_likelihood, time.time() - t)
    finally:
        np.save(model_dir + "UP", UP)
        np.save(model_dir + "LP", LP)
        np.save(model_dir + "LS", LS)
        print "Model saved..."

def read_matrix(train_data, visits, locations):
    UP = np.load("./UP.npy")
    LS = np.load("./LS.npy")
    LP = np.load("./LP.npy")

    DG = []
    t = time.time()
    count = 0
    count_in = 0
    for (u,lc,li) in visits:
        count += 1
        print count
        while count_in <= 100:
            lj = random.randint(0, poi_num)
            if locations.get(lj) != None:
                w = (1.0 + util.dist(locations[lc], locations[lj])) ** dis_coef
                DP = np.linalg.norm(UP[u] - LP[lj]) ** 2
                DS = np.linalg.norm(LS[lc] - LS[lj]) ** 2
                distance = w * (alpha * DP + (1 - alpha) * DS)
                DG.append([u, lj, distance])
            count_in += 1
        count_in = 0

    # for i in range(len(train_data) - 1):
    #     u, lc, li, time_irrelevance = train_data[i]
    #     u2, lc2, li2, time_irrelevance2 = train_data[i + 1]
    #     print i
    #     if u != u2:
    #         lj = li
    #         for lj in range(poi_num):
    #             #print lj
    #             # recommend poi (unvisited and visited)
    #             if locations.get(lj) != None:
    #                 w = (1.0 + util.dist(locations[lc], locations[lj])) ** dis_coef
    #                 DP = np.linalg.norm(UP[u] - LP[lj]) ** 2
    #                 DS = np.linalg.norm(LS[lc] - LS[lj]) ** 2
    #                 distance = w * (alpha * DP + (1 - alpha) * DS)
    #                 DG.append([u, lj, distance])
    DG.sort(key=operator.itemgetter(0, 2))
    
    # 只取每个用户距离得分最小的前三个poi点
    temp = []
    # 头个用户的top3
    temp.append(DG[0])
    temp.append(DG[1])
    temp.append(DG[2])

    # 从2-n放进temp数组
    for i in range(len(DG)):
        u, lj, distance = DG[i]
        if i != len(DG)-1:
            u_next, lj_next, distance_next = DG[i+1]
        if u != u_next:
            temp.append(DG[i+1])
            temp.append(DG[i+2])
            temp.append(DG[i+3])
        i = i+4

    with open('./dist_list_prme.txt', 'wb') as fwrite:
        for i in range(0,len(temp),3):
            u_1, lj_1, distance_1 = temp[i]
            u_2, lj_2, distance_2 = temp[i+1]
            u_3, lj_3, distance_3 = temp[i+2]
            fwrite.write(str(u_1)+'\t'+str(lj_1)+'\t'+str(lj_2)+'\t'+str(lj_3)+'\n')

        # for [u, lj, distance] in DG:
        #     n.write(str(u) + "\t" + str(lj) + "\t" + str(distance) + "\n")
    print "read_matrix elapsed: %f" % (time.time() - t)

def make_result(user_dict, loc_dict, user_list):
    fwrite = open('./result_0922.csv','wb')
    top_dict = {}
    # print user_dict[467987]
    print user_dict['467987']
    loc_dict_id2hash = dict((value,key) for key,value in loc_dict.iteritems())
    # user_dict_index2id = dict((value,key) for key,value in user_dict.iteritems())
    # print user_dict_index2id[31737]
    with open('./dist_list_prme.txt','rb') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            u_index, top1,top2,top3 = temp[0],temp[1],temp[2],temp[3]
            # print uid
            # userrid = user_dict_index2id[uid]
            # print userrid
            top_dict[u_index] = [top1,top2,top3]
    # print top_dict
    count = 0
    knn_dict = {}
    with open('./result_knn.csv','rb') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split(',')
            orderid,hash_top1,hash_top2,hash_top3= \
            temp[0],temp[1],temp[2],temp[3]
            knn_dict[orderid] = [hash_top1,hash_top2,hash_top3]
    print 'read_knn_coldstart over...'

    with open('./test.csv','rb') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split(',')
            orderid,userid = str(temp[0]),str(temp[1])
            if userid not in user_list:
                print count
                count += 1
                hash_top1,hash_top2,hash_top3=knn_dict[orderid]
                fwrite.write(str(orderid)+','+str(hash_top1)+','+str(hash_top2)+','+str(hash_top3)+'\n')
                continue
            u_index = str(user_dict[userid])
            top1,top2,top3 = top_dict[u_index]

            # print type(top1)
            # print type(loc_dict[top1])
            hash_top1,hash_top2,hash_top3 = loc_dict_id2hash[top1],loc_dict_id2hash[top2],loc_dict_id2hash[top3]
            fwrite.write(str(orderid)+','+str(hash_top1)+','+str(hash_top2)+','+str(hash_top3)+'\n')
    fwrite.close()

if __name__ == '__main__':
    t = time.time()
    util = Util()
    # step1------read dataset
    locations = get_locations()
    train_data, visits, user_dict, loc_dict, user_list = read_training_data()
    print "Data Loaded... Elapsed", time.time() - t

    # step2------learning prme-g
    # learning(train_data, visits, locations, False)
    # print "learning over...", time.time()-t

    # step3------calculate the distance
    # print len(visits)
    # read_matrix(train_data, visits, locations)
    # print "make distance_list ok...Elapsed", time.time() - t

    # step4-------make the result.csv
    make_result(user_dict, loc_dict, user_list)
    print 'maker result over..'