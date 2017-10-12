# coding: utf-8

import sys, time
import numpy as np
import scipy.sparse as sp
import pandas as pd
import math
from MFbpr import MFbpr

def read_dataset():
    # step1----------read dataset
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('./rating.txt', sep = '\t', names = header)

    users = df.user_id.unique()
    items = df.item_id.unique()

    n_users = users.shape[0]
    n_items = items.shape[0]

    # print type(users)
    # print users
    # print items
    print n_users, n_items
    return df, users, items, n_users, n_items

def map_id2index(users, items, n_users, n_items):
    # step2------------map userid/itemid to matrix_index
    uid_2_uindex = {}
    ucount = 0
    iid_2_iindex = {}
    icount = 0

    for i in range(n_users):
        uid_2_uindex[users[i]] = ucount
        ucount += 1

    for j in range(n_items):
        iid_2_iindex[items[j]] = icount
        icount += 1

    # print sorted(uid_2_uindex.iteritems(), key = lambda asd:asd[1], reverse = False)
    # print iid_2_iindex
    return uid_2_uindex, iid_2_iindex

def make_traintest(df, uid_2_uindex, iid_2_iindex):
    # step3----------make train & test dataset
    from sklearn import model_selection as cv
    train_data, test_data = cv.train_test_split(df, test_size = 0.25)

    train_data_matrix = np.zeros((n_users, n_items))
    # print type(train_data_matrix)
    for line in train_data.itertuples():
        # print line[1],line[2]
        # print uid_2_uindex[line[1]], iid_2_iindex[line[2]], line[3]
        train_data_matrix[uid_2_uindex[line[1]], iid_2_iindex[line[2]]] = float(line[3])


    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[uid_2_uindex[line[1]], iid_2_iindex[line[2]]] = float(line[3])

    return train_data_matrix, test_data_matrix 

def calc_sparsity(df, n_users, n_items):
    # step4=============sparsity
    sparsity = round(1.0 - len(df) / float(n_users * n_items), 3)
    spar_2_str = str(sparsity * 100) + '%'
    return spar_2_str

def load_data(ratingFile, testRatio=0.3):
    user_count = item_count = 0
    ratings = []
    for line in open(ratingFile):
        arr = line.strip().split()
        user_id = int(arr[0])
        item_id = int(arr[1])
        score = float(arr[2])
        timestamp = long(arr[3])
        ratings.append((user_id, item_id, score, timestamp))
        user_count = max(user_count, user_id)
        item_count = max(item_count, item_id)    
    user_count += 1
    item_count += 1
    print user_count,item_count

    ratings = sorted(ratings, key=lambda x: x[3])   # sort by timestamp
    
    test_count = int(len(ratings) * testRatio)
    count = 0
    trainMatrix = sp.lil_matrix((user_count, item_count))
    print trainMatrix.shape
    testRatings = []
    for rating in ratings:
        if count < len(ratings) - test_count:
            trainMatrix[rating[0], rating[1]] = 1
        else:
            testRatings.append(rating)
        count += 1
    
    newUsers = set([])
    newRatings = 0
    
    for u in xrange(user_count):
        if trainMatrix.getrowview(u).sum() == 0:
            newUsers.add(u)
    for rating in ratings:
        if rating[0] in newUsers:
            newRatings += 1
    
    sys.stderr.write("Data\t{}\n".format(ratingFile))
    sys.stderr.write("#Users\t{}, #newUser: {}\n".format(user_count, len(newUsers)))
    sys.stderr.write("#Items\t{}\n".format(item_count))
    sys.stderr.write(
        "#Ratings\t {} (train), {}(test), {}(#newTestRatings)\n".format(
            trainMatrix.sum(),  len(testRatings), newRatings))
    
    return trainMatrix, testRatings

def evaluate_model_online(model, name, interval):
    start = time.time()
    model.evaluateOnline(testRatings, interval)
    sys.stderr.write("{}\t <hr, ndcg, prec>:\t {}\t {}\t {} [{}]\n".format( 
                     name, np.mean(model.hits), np.mean(model.ndcgs), np.mean(model.precs),
                     time.time() - start))


def mean_average_precision(predictions,test,K):
    n_users,n_items = predictions.shape
    index_predictions = np.argsort(-predictions)
    ap = float(0.0)
    for i in range(n_users):
        sum = float(0.0)
        fenzi_count = int(0)
        fenmu_count = int(0)
        for j in range(K):
            fenmu_count += 1
            topk_intest_index = index_predictions[i][j]
            if test[i][topk_intest_index] != 0:
                fenzi_count += 1
                sum += float(fenzi_count/fenmu_count)
        ap += float(sum/(np.count_nonzero(test[i])))
    MAP = float(ap/n_users)
    return MAP

def pre_rec_f1(predictions,test,K):
    n_users,n_items = predictions.shape
    sum = 0.0
    rec = 0.0
    pre = 0.0
    index_predictions = np.argsort(-predictions)
    for i in range(n_users):
        for j in range(K):
            topk_intest_index = index_predictions[i][j]
            if test[i][topk_intest_index] != 0:
                sum += 1.0
        rec += sum/(np.count_nonzero(test[i]))
        pre += float(sum/K)
        sum = 0.0
    precision = pre / n_users
    recall = rec / n_users
    f_score = (2.0*precision*recall) / (precision+recall)
    return precision, recall, f_score

def getDCG(rels):
    dcg = rels[0]
    i = 2
    for rel in rels[1:]:
        dcg = dcg + pow(2,rel) / math.log(i,2)
        i += 1
    return dcg

def getIDCG(rels):
    rels.sort()
    rels.reverse()
    return getDCG(rels)

def getNDCG(predictions,test,K):
    # followed the loop, len(topdata[i]) is [1,5,10,15,20]
    # objective: to mark each pid of topData[i](compared with corresponding testData) in three levels
    # to perform the ranking criteria better
    n_users,n_items = predictions.shape
    index_predictions = np.argsort(-predictions)
    scores = []
    ndcg = 0.0
    dcg = 0.0
    idcg = 0.0
    for i in range(n_users):
        for j in range(K):
            topk_intest_index = index_predictions[i][j]
            if test[i][topk_intest_index] != 0:
                scores.append(1)
            else:
                scores.append(0)
        # print scores
        dcg += getDCG(scores)
        idcg += getIDCG(scores)
        scores = []
    ndcg = dcg/idcg
    return ndcg

from sklearn.metrics import mean_squared_error
def get_rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return math.sqrt(mean_squared_error(pred, actual))

if __name__ == "__main__":
    # data
    trainMatrix, testRatings = load_data('reno_ratings.txt')

    # settings
    # topK = 5
    factors = 64
    maxIter = 10
    maxIterOnline = 1
    lr = 0.01
    adaptive = False
    init_mean = 0.0
    init_stdev = 0.1
    reg = 0.01
    showProgress = False
    showLoss = True

    # print type(testRatings) #list
    # print testRatings[1][1]

    # fwrite = open('./bpr_criteria.txt','a+')
    # topk_list = [5]
    # for topK in topk_list:
    #     bpr = MFbpr(trainMatrix, testRatings, topK, factors, maxIter, lr, adaptive, reg, init_mean, init_stdev, showProgress, showLoss)
    #     bpr.buildModel()
    #     u_vec = bpr.U
    #     v_vec = bpr.V
    #     print u_vec.shape
    #     print v_vec.shape

    #     predictions = u_vec.dot((v_vec.T))
    #     print predictions.shape
    #     np.save('./predictions.npy',predictions)

    #     # fwrite.write('top'+str(topK)+'\tpre: '+ \
    #     #     str(precision)+'\trec: '+str(recall)+ \
    #     #     '\tf1: '+str(f_score)+'\tndcg: '+ \
    #     #     str(ndcg)+'\tmap: '+str(MAP)+'\n')
    
    #     bpr.maxIterOnline = maxIterOnline;
    #     evaluate_model_online(bpr, "BPR", 1000);
    # fwrite.close()

    # step1-------read dataset
    t = time.time()
    df, users, items, n_users, n_items = read_dataset()
    print 'step1--read dataset\telapse:', time.time() - t

    # step2----------map userid/itemid to matrix_index
    uid_2_uindex, iid_2_iindex = map_id2index(users, items, n_users, n_items)
    print 'step2--map userid/itemid to matrix index\telapse:', time.time() - t

    # step3----------make train and test dataset
    train_data_matrix, test_data_matrix  = make_traintest(df, uid_2_uindex, iid_2_iindex)
    print 'step3--make train and test dataset\telapse:', time.time() - t
    # print type(train_data_matrix)

    # step4---------calculate sparsity
    print calc_sparsity(df, n_users, n_items)
    print 'step4--calculate sparsity\telapse:', time.time() - t
    

    predictions = np.load('./predictions.npy')
    print predictions.shape
    print type(predictions)
    print type(trainMatrix)
    print type(testRatings)
    print type(trainMatrix.toarray())
    # test = np.array(testRatings)
    # train = trainMatrix.toarray()
    
    test = test_data_matrix
    print test.shape

    fwrite = open('./bpr_criteria.txt','a+')
        # print 'top %d\t'%K
    MAP = mean_average_precision(predictions,test,5)
    print MAP
    precision, recall, f_score = pre_rec_f1(predictions,test,5)
    print precision, recall, f_score
    ndcg = getNDCG(predictions,test,5)
    print ndcg
    fwrite.write('top'+str(5)+'\tpre: '+ \
        str(precision)+'\trec: '+str(recall)+ \
        '\tf1: '+str(f_score)+'\tndcg: '+ \
        str(ndcg)+'\tmap: '+str(MAP)+'\n')
    fwrite.close()


