import time


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
                +'\tprecision: '+str(precision)+ \
                '\trecall: '+str(recall)+'\n')


def precision_recall(testData, topData, K):
    precision, recall = 0.0, 0.0
    for each_test, each_top in zip(testData, topData):
        intsec = list(set(each_test).intersection(set(each_top)))
        precision += (1.0 * len(intsec)) / (1.0 * K)
        recall += (1.0 * len(intsec)) / (1.0 * len(each_test))
        # print len(intsec)
    return precision / 2363, recall / 2363

def next_acc(testData, topData):
    acc = 0.0
    sum = 0.0
    for each_test, each_top in zip(testData, topData):
        for i in each_test:
            if i in each_top:
                sum += 1.0
        acc += sum / (1.0 * len(each_test))
        sum = 0.0
    return acc / 2363


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
