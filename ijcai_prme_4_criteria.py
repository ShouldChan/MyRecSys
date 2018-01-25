import time
import math

dataset_name = "brightkite"

n_users = int(1850)

def calculate(topk):
    # read list
    with open("./result/"+dataset_name+"/test_list_prme.txt", "r") as f:
        lines = f.readlines()
        testdata = []
        for line in lines:
            data = line.split("\t")
            value = []
            for d in data:
                if len(d) != 0:
                    value.append(d.replace("\r\n", ""))
            testdata.append(value)

    fw = open("./result/"+dataset_name+"/not_newpoi_criteria.txt","a+")
    for K in topk:
        K = int(K)
        print 'top %d\t'%K
        with open('./result/'+dataset_name+'/not_newpoi_top'+str(K)+'_prme.txt', "r") as f:
            lines = f.readlines()
            topdata = []
            for line in lines:
                data = line.split("\t")
                value = []
                for d in data:
                    if len(d) != 0:
                        value.append(d.replace("\n", ""))
                topdata.append(value)
        newTestData = []
        for test in testdata:
            test = map(lambda x: int(x), test)
            newTestData.append(test)
            # print newTestData

        newTopData = []
        for t in topdata:
            top = map(lambda x: int(x), t)
            newTopData.append(top)
        # print newTopData
        # print len(newTopData), len(newTestData)
        # return newTestData, newTopData

        precision, recall = precision_recall(newTestData, newTopData, K)


        print "precision:   %f" % precision
        print "recall:  %f" % recall

        accuracy = next_acc(newTestData, newTopData)
        print "accuracy:  %f" % accuracy

        MAP = mean_average_precision(newTestData,newTopData)
        print 'MAP:  %f' % MAP

        # with open('./result/acc_map_ndcg.txt', 'a+') as fwrite:
        #     fwrite.write('top'+str(K)+'\taccuracy: '+str(accuracy) \
        #         +'\tMAP: '+str(MAP)+'\n')
        ndcg = getNDCG(newTestData,newTopData)
        print 'ndcg: %f' % ndcg
        fw.write('top'+str(K)+'\taccuracy: '+str(accuracy) \
                +'\tMAP: '+str(MAP)+'\n')
    fw.write("------------------"+str(dataset_name)+"-------------------\n")
    fw.close()

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

def getNDCG(testData, topData):
    # followed the loop, len(topdata[i]) is [1,5,10,15,20]
    # objective: to mark each pid of topData[i](compared with corresponding testData) in three levels
    # to perform the ranking criteria better
    scores = []
    ndcg = 0.0
    dcg = 0.0
    idcg = 0.0
    for i in range(len(topData)):
        for j in topData[i]:
            if j in testData[i]:
                scores.append(1)
            else:
                scores.append(0)
        # print scores
        dcg += getDCG(scores)
        idcg += getIDCG(scores)
        scores = []
    ndcg = dcg/idcg
    return ndcg

def getHitRatio(testData, topData):
    hr = []
    for each_test, each_top in zip(testData, topData):
        for i in each_test:
            if i in each_top:
                hr.append(1)
                break
    return float(len(hr)) / float(n_users)

def precision_recall(testData, topData, K):
    precision, recall = 0.0, 0.0
    for each_test, each_top in zip(testData, topData):
        intsec = list(set(each_test).intersection(set(each_top)))
        precision += (1.0 * len(intsec)) / (1.0 * K)
        recall += (1.0 * len(intsec)) / (1.0 * len(each_test))
        # print len(intsec)
    return precision / n_users, recall / n_users

def next_acc(testData, topData):
    acc = 0.0
    sum = 0.0
    for each_test, each_top in zip(testData, topData):
        for i in each_test:
            if i in each_top:
                sum += 1.0
        acc += sum / (1.0 * len(each_test))
        sum = 0.0
    return acc / n_users

# AP = (Summation_j=1^ni * P(j) * y_(i,j)) / (Summation_j=1^ni y_(i,j))
# mAP = Summation AP / m
# P(j) = (Summation hitnumber) / j's ranking position
def mean_average_precision(testData, topData):
    print(len(testData))
    print(len(topData))
    sum_ap = float(0.0)
    for i in range(0,len(testData)):
        # print "Round\t%d"%i
        ap = float(0.0)
        p_j = float(0.0)
        sum_p_j_y_ij = float(0.0)
        sum_y_ij = float(0.0)
        hit_number = int(0)
        rank_position = int(0)
        count_relevant = int(0)
        flag_relevant = int(0)
        for j in topData[i]:
            rank_position += 1
            # print "rank_position\t%d"%rank_position
            if j in testData[i]:
                hit_number += 1
                flag_relevant = 1
                p_j = float(hit_number) / float(rank_position)
                # print "hit_number\t%d"%hit_number
                # print "p_j\t%f"%p_j
                count_relevant += 1
            else:
                p_j = 0.0
                flag_relevant = 0
                # print "hit_number\t%d"%hit_number
                # print "p_j\t%f"%p_j
            sum_p_j_y_ij += (float(p_j) * flag_relevant)
            sum_y_ij = float(count_relevant)
            # print sum_p_j_y_ij
            # print sum_y_ij
        if sum_y_ij == 0.0:
            ap += 0.0
        else:
            ap += sum_p_j_y_ij / sum_y_ij
        sum_ap += ap
        # print "ap\t%f"%ap
    # print "m\t%d"%len(testlist)
    MAP = float(sum_ap/len(testData))
    # print "MAP\t%f"%MAP
    return MAP


def main():
    topk = [1,5,10,15,20]

    calculate(topk)


if __name__ == '__main__':
    main()
