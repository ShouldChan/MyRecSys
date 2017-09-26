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

        MAP = mean_average_precision(newTestData,newTopData)
        print 'MAP:  %f' % MAP
        
        # with open('./result/next_acc.txt', 'a+') as fwrite:
        #     fwrite.write('top'+str(K)+'\taccuracy: '+str(accuracy) \
        #         + str(recall)+'\n')
        
        


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

def mean_average_precision(testData, topData):
    print len(testData[1]), len(topData) # len(testData)==n_users
    ap = float(0.0)
    for i in range(len(topData)):
        sum = float(0.0)
        fenzi_count = int(0)
        fenmu_count = int(0)
        for j in topData[i]:
            fenmu_count += 1
            if j in testData[i]:
                fenzi_count += 1
                sum += float(fenzi_count/fenmu_count)
        ap += float(sum/len(testData[i]))
    MAP = float(ap/len(testData))
    # print map
    return MAP

def main():
    topk = [1,5,10,15,20]

    calculate(topk)


if __name__ == '__main__':
    main()
