import time

n_users = int(354)

def calculate(topk):
    # read list
    with open('./result/test_list_prme.txt', "r") as f:
        lines = f.readlines()
        testdata = []
        for line in lines:
            data = line.split("\t")
            value = []
            for d in data:
                if len(d) != 0:
                    value.append(d.replace("\r\n", ""))
            testdata.append(value)

    for K in topk:
        K = int(K)
        print 'top %d\t'%K
        with open('./result/top'+str(K)+'_prme.txt', "r") as f:
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

        with open('./result/next_acc.txt', 'a+') as fwrite:
            fwrite.write('top'+str(K)+'\taccuracy: '+str(accuracy) \
                + str(recall)+'\n')


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

# def mean_average_precision(testData, topData, K):
#     i = 1
#     total = len(topData)
#     p = float(0.0)
#     hit_num = int(0)
#     sorted_pred_rank = sorted(topData.items(), key=lambda x:x[1],reverse=True)
#     for v_i in sorted_pred_rank:
#         if i<K+1:
#             if v_i[0] in testData:
#                 hit_num += 1
#                 p += float(hit_num / i)
#             else:
#                 break
#             i += 1
#         if i!= 1:
#             return hit_num / (i-1),p/total
#         return 0,0

def main():
    topk = [1,5,10,15,20]

    calculate(topk)
    # print "read elapsed:   %f" % (time.time() - t)
    # precision, recall = precision_recall(testData, topData)
    # print "calc elapsed:   %f" % (time.time() - t)

    # print "precision:   %f" % precision
    # print "recall:  %f" % recall

    # accuracy = next_acc(testData, topData)
    # print "accuracy:  %f" % accuracy

    # with open('./result/next_acc.txt', 'a+') as fwrite:
    #     fwrite.write(str(K)+'\t'+str(accuracy)+'\n')


if __name__ == '__main__':
    main()
