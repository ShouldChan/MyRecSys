# encoding=utf-8
from __future__ import division
from pylab import *
import numpy as np
import math
import statics


def split_file(filename, wfilename):
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    for line in open(filename, 'r').readlines():
        arr1.append(line.strip().split('\t')[0])
        arr2.append(line.strip().split('\t')[2])
        arr3.append(line.strip().split('\t')[3])
        arr4.append(line.strip().split('\t')[13])
    wfile = open(wfilename, 'a')
    for i in range(len(arr3)):
        # wfile.write(arr1[i]+'\t'+arr3[i].decode("unicode_escape").encode('utf-8')+'\t'+arr4[i]+'\n')
        wfile.write(arr1[i] + '\t' + arr2[i] + '\t' + arr3[i] + '\t' + arr4[i] + '\n')


# split_file('../data/room_infoLos-Angeles--CA.csv','../data/room_infoLos_location.csv')
from nltk.corpus import wordnet as wn
import nltk
import re


def statics_voc(filename, filenamew):
    stopkeys = ["a", "an", "am", "b", "of", "the", "", "for", "with", "or", "are",
                "is", "to", "i", "as", "all", "if", "o", "It", "so", "do", "A",
                "you", "from", "your", "The", "I", "our", "we", "be", "if", "We",
                "my", "There", "there", "thi", "but", "by", "at", "2", "it", "on",
                "and", "in", "\u2022", "la", "null", "1", "4", "15"
                ]

    r = '[,''\.!\?/\"\n:*&\\-~=_\')(;]+'
    r1 = '(\\\\n)+'
    arr = {}
    for line in open(filename, 'r').readlines():
        try:
            li = line.strip().split('\t')[26]

            li = re.sub(r1, ' ', li)
            print li
            for l in li.split(' '):
                l = re.sub(r, '', l)
                # print l
                l = nltk.PorterStemmer().stem_word(l)
                l = l.lower()
                # try:
                # 	l=l.decode('unicode_escape').encode('utf-8')
                # except Exception,e:
                # 	pass
                # print l
                if l in arr and l not in stopkeys:
                    arr[l] += 1
                elif l not in arr and l not in stopkeys:
                    arr[l] = 1
        except Exception, e:
            pass

    arr = dict(sorted(arr.iteritems(), key=lambda asd: asd[1], reverse=True))
    filw = open(filenamew, 'a')
    for i in arr:
        if arr.get(i) > 200:
            filw.write(str(i) + '\t' + str(arr.get(i)) + '\n')
            # print i,arr.get(i)


# filename='../data/room_infoLos-Angeles--CA.csv'
# filenamew='../data/room_infoLos-Angele_reviews.csv'
# statics_voc(filename,filenamew)
def read_filename(location):
    file_loc = []
    for i in open(location, 'r').readlines():
        files = i.strip()
        file_loc.append(files)
    return file_loc

# 处理设施amenities 用向量表示
def facility_sorted(filename):
    facility = []
    for line in open(filename, 'r').readlines():
        price = line.strip().split('\t')[4]
        r = '[$]+'
        r2 = '(u\')+'
        r3 = '[{}\']'

        infrustration = line.strip().split('\t')[6]
        infrustrations = [0] * 30

        infrus_names = ["\u4e00\u6c27\u5316\u78b3\u63a2\u6d4b\u5668", '\u65e0\u969c\u788d\u8bbe\u65bd',
                        '\u6d17\u8863\u673a', '\u8702\u9e23\u5668/\u65e0\u7ebf\u5bf9\u8bb2\u673a', '\u7535\u89c6',
                        '\u5065\u8eab\u623f',
                        '\u5ba4\u5185\u58c1\u7089', '\u5927\u53a6\u5185\u7535\u68af', '\u751f\u6d3b\u5fc5\u9700\u54c1',
                        '\u706d\u706b\u5668', '\u6e38\u6cf3\u6c60', '\u5b89\u5168\u5361', '\u65e0\u7ebf\u7f51\u7edc',
                        '\u5141\u8bb8\u5438\u70df',
                        '\u53a8\u623f', '\u6709\u7ebf\u7535\u89c6', '\u6d17\u53d1\u6c34', '\u95e8\u536b',
                        '\u6696\u6c14', '\u70d8\u5e72\u673a',
                        '\u6b22\u8fce\u5bb6\u5ead/\u643a\u5e26\u5b69\u5b50\u5165\u4f4f',
                        '\u5185\u90e8\u63d0\u4f9b\u514d\u8d39\u505c\u8f66\u4f4d',
                        '\u65e9\u9910', '\u7f51\u7edc', '\u7a7a\u8c03', '\u5141\u8bb8\u643a\u5e26\u5ba0\u7269',
                        '\u70df\u96fe\u63a2\u6d4b\u5668',
                        '\u9002\u5408\u4e3e\u529e\u6d3b\u52a8', '\u70ed\u6c34\u6d74\u7f38', '\u6025\u6551\u5305'
                        ]

        for infrus in infrustration.strip().split(','):

            try:
                infrus_name = infrus.strip().split(':')[0]
                infrus_value = int(infrus.strip().split(':')[1])
                infrus_name = re.sub(r2, '', infrus_name)
                infrus_name = re.sub(r3, '', infrus_name)

                if infrus_name == "\u4e00\u6c27\u5316\u78b3\u63a2\u6d4b\u5668" and infrus_value == 1:
                    infrustrations[0] = 1
                elif infrus_name == '\u65e0\u969c\u788d\u8bbe\u65bd' and infrus_value == 1:
                    infrustrations[1] = 1
                elif infrus_name == '\u6d17\u8863\u673a' and infrus_value == 1:
                    infrustrations[2] = 1
                elif infrus_name == '\u8702\u9e23\u5668/\u65e0\u7ebf\u5bf9\u8bb2\u673a' and infrus_value == 1:
                    infrustrations[3] = 1
                elif infrus_name == '\u7535\u89c6' and infrus_value == 1:
                    infrustrations[4] = 1
                elif infrus_name == '\u5065\u8eab\u623f' and infrus_value == 1:
                    infrustrations[5] = 1
                elif infrus_name == '\u5ba4\u5185\u58c1\u7089' and infrus_value == 1:
                    infrustrations[6] = 1
                elif infrus_name == '\u5927\u53a6\u5185\u7535\u68af' and infrus_value == 1:
                    infrustrations[7] = 1
                elif infrus_name == '\u751f\u6d3b\u5fc5\u9700\u54c1' and infrus_value == 1:
                    infrustrations[8] = 1
                elif infrus_name == '\u706d\u706b\u5668' and infrus_value == 1:
                    infrustrations[9] = 1
                elif infrus_name == '\u6e38\u6cf3\u6c60' and infrus_value == 1:
                    infrustrations[10] = 1
                elif infrus_name == '\u5b89\u5168\u5361' and infrus_value == 1:
                    infrustrations[11] = 1
                elif infrus_name == '\u65e0\u7ebf\u7f51\u7edc' and infrus_value == 1:
                    infrustrations[12] = 1
                elif infrus_name == '\u5141\u8bb8\u5438\u70df' and infrus_value == 1:
                    infrustrations[13] = 1
                elif infrus_name == '\u53a8\u623f' and infrus_value == 1:
                    infrustrations[14] = 1
                elif infrus_name == '\u6709\u7ebf\u7535\u89c6' and infrus_value == 1:
                    infrustrations[15] = 1
                elif infrus_name == '\u6d17\u53d1\u6c34' and infrus_value == 1:
                    infrustrations[16] = 1
                elif infrus_name == '\u95e8\u536b' and infrus_value == 1:
                    infrustrations[17] = 1
                elif infrus_name == '\u6696\u6c14' and infrus_value == 1:
                    infrustrations[18] = 1
                elif infrus_name == '\u70d8\u5e72\u673a' and infrus_value == 1:
                    infrustrations[19] = 1
                elif infrus_name == '\u6b22\u8fce\u5bb6\u5ead/\u643a\u5e26\u5b69\u5b50\u5165\u4f4f' and infrus_value == 1:
                    infrustrations[20] = 1
                elif infrus_name == '\u5185\u90e8\u63d0\u4f9b\u514d\u8d39\u505c\u8f66\u4f4d' and infrus_value == 1:
                    infrustrations[21] = 1
                elif infrus_name == '\u65e9\u9910' and infrus_value == 1:
                    infrustrations[22] = 1
                elif infrus_name == '\u7f51\u7edc' and infrus_value == 1:
                    infrustrations[23] = 1
                elif infrus_name == '\u7a7a\u8c03' and infrus_value == 1:
                    infrustrations[24] = 1
                elif infrus_name == '\u5141\u8bb8\u643a\u5e26\u5ba0\u7269' and infrus_value == 1:
                    infrustrations[25] = 1
                elif infrus_name == '\u70df\u96fe\u63a2\u6d4b\u5668' and infrus_value == 1:
                    infrustrations[26] = 1
                elif infrus_name == '\u9002\u5408\u4e3e\u529e\u6d3b\u52a8' and infrus_value == 1:
                    infrustrations[27] = 1
                elif infrus_name == '\u70ed\u6c34\u6d74\u7f38' and infrus_value == 1:
                    infrustrations[28] = 1
                elif infrus_name == '\u6025\u6551\u5305' and infrus_value == 1:
                    infrustrations[29] = 1
            except Exception, e:
                pass
        facility.append(infrustrations)

    facility = np.array(facility)
    coverage = list(facility.sum(axis=0))
    sor_cov = sorted(coverage)

    new_facility = []
    for i in range(len(sor_cov)):
        new_facility.append(infrus_names[coverage.index(sor_cov[i])])

    return new_facility


def infrustration(filename):
    arr = {}
    inf_train = []
    inf_test = []
    prices = []
    price_test = []
    facility = facility_sorted(filename)
    for line in open(filename, 'r').readlines():
        try:
            price = line.strip().split('\t')[13]
        except Exception, e:
            price = 0
        r = '[$]+'
        r2 = '(u\')+'
        r3 = '[{}\']'
        price = re.sub(r, '', price)
        price_rate = line.strip().split('\t')[24]
        overview_rate = line.strip().split('\t')[19]
        communication_rate = line.strip().split('\t')[20]
        clean_rate = line.strip().split('\t')[21]
        location_rate = line.strip().split('\t')[22]
        livin_rate = line.strip().split('\t')[23]
        infrustration = line.strip().split('\t')[25]
        infrustrations = [0] * 30
        for infrus in infrustration.strip().split(','):
            try:
                infrus_name = infrus.strip().split(':')[0]
                infrus_value = int(infrus.strip().split(':')[1])
                infrus_name = re.sub(r2, '', infrus_name)
                infrus_name = re.sub(r3, '', infrus_name)

                if infrus_name == facility[0] and infrus_value == 1:
                    infrustrations[0] = 1
                elif infrus_name == facility[2] and infrus_value == 1:
                    infrustrations[1] = 1
                elif infrus_name == facility[4] and infrus_value == 1:
                    infrustrations[2] = 1
                elif infrus_name == facility[6] and infrus_value == 1:
                    infrustrations[3] = 1
                elif infrus_name == facility[8] and infrus_value == 1:
                    infrustrations[4] = 1
                elif infrus_name == facility[10] and infrus_value == 1:
                    infrustrations[5] = 1
                elif infrus_name == facility[12] and infrus_value == 1:
                    infrustrations[6] = 1
                elif infrus_name == facility[14] and infrus_value == 1:
                    infrustrations[7] = 1
                elif infrus_name == facility[16] and infrus_value == 1:
                    infrustrations[8] = 1
                elif infrus_name == facility[18] and infrus_value == 1:
                    infrustrations[9] = 1
                elif infrus_name == facility[20] and infrus_value == 1:
                    infrustrations[10] = 1
                elif infrus_name == facility[22] and infrus_value == 1:
                    infrustrations[11] = 1
                elif infrus_name == facility[24] and infrus_value == 1:
                    infrustrations[12] = 1
                elif infrus_name == facility[26] and infrus_value == 1:
                    infrustrations[13] = 1
                elif infrus_name == facility[18] and infrus_value == 1:
                    infrustrations[14] = 1
                elif infrus_name == facility[1] and infrus_value == 1:
                    infrustrations[15] = 1
                elif infrus_name == facility[3] and infrus_value == 1:
                    infrustrations[16] = 1
                elif infrus_name == facility[5] and infrus_value == 1:
                    infrustrations[17] = 1
                elif infrus_name == facility[7] and infrus_value == 1:
                    infrustrations[18] = 1
                elif infrus_name == facility[9] and infrus_value == 1:
                    infrustrations[19] = 1
                elif infrus_name == facility[11] and infrus_value == 1:
                    infrustrations[20] = 1
                elif infrus_name == facility[13] and infrus_value == 1:
                    infrustrations[21] = 1
                elif infrus_name == facility[15] and infrus_value == 1:
                    infrustrations[22] = 1
                elif infrus_name == facility[17] and infrus_value == 1:
                    infrustrations[23] = 1
                elif infrus_name == facility[19] and infrus_value == 1:
                    infrustrations[24] = 1
                elif infrus_name == facility[21] and infrus_value == 1:
                    infrustrations[25] = 1
                elif infrus_name == facility[23] and infrus_value == 1:
                    infrustrations[26] = 1
                elif infrus_name == facility[25] and infrus_value == 1:
                    infrustrations[27] = 1
                elif infrus_name == facility[27] and infrus_value == 1:
                    infrustrations[28] = 1
                elif infrus_name == facility[29] and infrus_value == 1:
                    infrustrations[29] = 1
            except Exception, e:
                pass

                # infrustrations.append(int(infrus.strip().split(':')[1]))
                # if overview_rate=='5.0':
        inf_train.append(infrustrations)
        prices.append(int(price))
        # print overview_rate,price_rate,clean_rate,location_rate,price,infrustrations
    # if overview_rate!='5.0'and overview_rate!='0':
    # 	inf_test.append(infrustrations)
    # 	price_test.append(int(price))
    return inf_train, prices


def price_infrustration(filename):
    arr = {}
    inf_train = []
    inf_test = []
    prices = []
    price_test = []
    facility = facility_sorted(filename)
    for line in open(filename, 'r').readlines():
        try:
            price = line.strip().split('\t')[13]
        except Exception, e:
            price = 0
        r = '[$]+'
        r2 = '(u\')+'
        r3 = '[{}\']'
        price = re.sub(r, '', price)
        price_rate = line.strip().split('\t')[24]
        overview_rate = line.strip().split('\t')[19]
        communication_rate = line.strip().split('\t')[20]
        clean_rate = line.strip().split('\t')[21]
        location_rate = line.strip().split('\t')[22]
        livin_rate = line.strip().split('\t')[23]
        infrustration = line.strip().split('\t')[25]
        infrustrations = [0] * 30
        for infrus in infrustration.strip().split(','):
            try:
                infrus_name = infrus.strip().split(':')[0]
                infrus_value = int(infrus.strip().split(':')[1])
                infrus_name = re.sub(r2, '', infrus_name)
                infrus_name = re.sub(r3, '', infrus_name)

                if infrus_name == facility[0] and infrus_value == 1:
                    infrustrations[0] = 1
                elif infrus_name == facility[2] and infrus_value == 1:
                    infrustrations[1] = 1
                elif infrus_name == facility[4] and infrus_value == 1:
                    infrustrations[2] = 1
                elif infrus_name == facility[6] and infrus_value == 1:
                    infrustrations[3] = 1
                elif infrus_name == facility[8] and infrus_value == 1:
                    infrustrations[4] = 1
                elif infrus_name == facility[10] and infrus_value == 1:
                    infrustrations[5] = 1
                elif infrus_name == facility[12] and infrus_value == 1:
                    infrustrations[6] = 1
                elif infrus_name == facility[14] and infrus_value == 1:
                    infrustrations[7] = 1
                elif infrus_name == facility[16] and infrus_value == 1:
                    infrustrations[8] = 1
                elif infrus_name == facility[18] and infrus_value == 1:
                    infrustrations[9] = 1
                elif infrus_name == facility[20] and infrus_value == 1:
                    infrustrations[10] = 1
                elif infrus_name == facility[22] and infrus_value == 1:
                    infrustrations[11] = 1
                elif infrus_name == facility[24] and infrus_value == 1:
                    infrustrations[12] = 1
                elif infrus_name == facility[26] and infrus_value == 1:
                    infrustrations[13] = 1
                elif infrus_name == facility[18] and infrus_value == 1:
                    infrustrations[14] = 1
                elif infrus_name == facility[1] and infrus_value == 1:
                    infrustrations[15] = 1
                elif infrus_name == facility[3] and infrus_value == 1:
                    infrustrations[16] = 1
                elif infrus_name == facility[5] and infrus_value == 1:
                    infrustrations[17] = 1
                elif infrus_name == facility[7] and infrus_value == 1:
                    infrustrations[18] = 1
                elif infrus_name == facility[9] and infrus_value == 1:
                    infrustrations[19] = 1
                elif infrus_name == facility[11] and infrus_value == 1:
                    infrustrations[20] = 1
                elif infrus_name == facility[13] and infrus_value == 1:
                    infrustrations[21] = 1
                elif infrus_name == facility[15] and infrus_value == 1:
                    infrustrations[22] = 1
                elif infrus_name == facility[17] and infrus_value == 1:
                    infrustrations[23] = 1
                elif infrus_name == facility[19] and infrus_value == 1:
                    infrustrations[24] = 1
                elif infrus_name == facility[21] and infrus_value == 1:
                    infrustrations[25] = 1
                elif infrus_name == facility[23] and infrus_value == 1:
                    infrustrations[26] = 1
                elif infrus_name == facility[25] and infrus_value == 1:
                    infrustrations[27] = 1
                elif infrus_name == facility[27] and infrus_value == 1:
                    infrustrations[28] = 1
                elif infrus_name == facility[29] and infrus_value == 1:
                    infrustrations[29] = 1
            except Exception, e:
                pass

                # infrustrations.append(int(infrus.strip().split(':')[1]))
        if overview_rate == '5.0' or overview_rate == '4.5':
            inf_train.append(infrustrations)
            prices.append(int(price))
            # print overview_rate,price_rate,clean_rate,location_rate,price,infrustrations
        if overview_rate != '5.0' and overview_rate != '0' and overview_rate != '4.5':
            inf_test.append(infrustrations)
            price_test.append(int(price))
    return inf_train, prices, inf_test, price_test


def mean_var(List):
    narray = np.array(List)
    sum1 = narray.sum()
    narray2 = narray * narray
    sum2 = narray2.sum()
    mean = sum1 / len(List)
    var = math.sqrt(sum2 / len(List) - mean ** 2)
    List = sorted(List)
    if len(List) % 2 == 1:
        mid = List[int(len(List) / 2)]
    else:
        mid = (List[int(len(List) / 2)] + List[int(len(List) / 2 - 1)]) / 2
    return mean, var, mid


def basic_inf(vec, label):
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []
    for i in range(len(vec)):
        if sum(vec[i]) >= 0 and sum(vec[i]) < 6:
            class1.append(label[i])
        elif sum(vec[i]) >= 6 and sum(vec[i]) < 12:
            class2.append(label[i])
        elif sum(vec[i]) >= 12 and sum(vec[i]) < 18:
            class3.append(label[i])
        elif sum(vec[i]) >= 18 and sum(vec[i]) < 24:
            class4.append(label[i])
        elif sum(vec[i]) >= 24 and sum(vec[i]) <= 30:
            class5.append(label[i])
    print class1, class2, class3, class4, class5

    plot(range(len(class1)), sorted(class1, reverse=True), color="blue", linewidth=2.5, linestyle="-", label="class_1")
    plot(range(len(class2)), sorted(class2, reverse=True), color="pink", linewidth=2.5, linestyle="-", label="class_2")
    plot(range(len(class3)), sorted(class3, reverse=True), color="green", linewidth=2.5, linestyle="-", label="class_3")
    plot(range(len(class4)), sorted(class4, reverse=True), color="black", linewidth=2.5, linestyle="-", label="class_4")
    plot(range(len(class5)), sorted(class5, reverse=True), color="red", linewidth=2.5, linestyle="-", label="class_5")
    legend(loc='upper right')
    show()
    mean1, var1, mid1 = mean_var(class1)
    mean2, var2, mid2 = mean_var(class2)
    mean3, var3, mid3 = mean_var(class3)
    mean4, var4, mid4 = mean_var(class4)
    mean5, var5, mid5 = mean_var(class5)
    print mean1, var1, mid1
    print mean2, var2, mid2
    print mean3, var3, mid3
    print mean4, var4, mid4
    print mean5, var5, mid5
    return class1, class2, class3, class4, class5


def price_facility():
    overview_rate = line.strip().split('\t')[19]


def sklearn_kmeans(inf_train, prices):
    from sklearn import cluster
    from sklearn.cluster import SpectralClustering
    from sklearn.cluster import MeanShift
    clf = SpectralClustering(n_clusters=5, eigen_solver=None, random_state=None, n_init=10, gamma=1.0,
                             affinity='nearest_neighbors', n_neighbors=5, eigen_tol=0.0, assign_labels='kmeans',
                             degree=3, coef0=1, kernel_params=None)
    # clf=cluster.KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=1800, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)

    clf.fit(inf_train, prices)
    label = clf.labels_

    print label
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    cluster5 = []

    if len(prices) == len(label):
        filw = open('../data/result_cluster.txt', 'a')

        for i in range(len(prices)):
            filw.write(
                str(label[i]) + '\t' + str(prices[i]) + '\t' + str(inf_train[i]) + '\t' + str(sum(inf_train[i])) + '\n')
            if label[i] == 0:
                cluster1.append(prices[i])
            elif label[i] == 1:
                cluster2.append(prices[i])
            elif label[i] == 2:
                cluster3.append(prices[i])
            elif label[i] == 3:
                cluster4.append(prices[i])
            elif label[i] == 4:
                cluster5.append(prices[i])
        plot(range(len(cluster1)), sorted(cluster1, reverse=True), color="blue", linewidth=2.5, linestyle="-",
             label="class_1")
        plot(range(len(cluster2)), sorted(cluster2, reverse=True), color="green", linewidth=2.5, linestyle="-",
             label="class_2")
        plot(range(len(cluster3)), sorted(cluster3, reverse=True), color="pink", linewidth=2.5, linestyle="-",
             label="class_3")
        plot(range(len(cluster4)), sorted(cluster4, reverse=True), color="black", linewidth=2.5, linestyle="-",
             label="class_4")
        plot(range(len(cluster5)), sorted(cluster5, reverse=True), color="red", linewidth=2.5, linestyle="-",
             label="class_5")
        legend(loc='upper right')
        show()
        mean1, var1 = mean_var(cluster1)
        mean2, var2 = mean_var(cluster2)
        mean3, var3 = mean_var(cluster3)
        mean4, var4 = mean_var(cluster4)
        mean5, var5 = mean_var(cluster5)
        print mean1, var1
        print mean2, var2
        print mean3, var3
        print mean4, var4
        print mean5, var5


def sklearn_regression(inf_train, prices):
    from sklearn import linear_model

    import numpy as np
    arr = [0] * 5
    arr = basic_inf(inf_train, prices)

    dat_train = inf_train[:int(len(prices) * 0.8)]
    dat_p_tr = prices[:int(len(prices) * 0.8)]
    dat_test = inf_train[int(len(prices) * 0.8):]
    dat_p_te = prices[int(len(prices) * 0.8):]
    # clf=linear_model.LinearRegression()
    # clf=linear_model.RidgeCV(alphas=[0.1,1.0,10.0,100.0,1000.0,0.001])
    # clf=linear_model.Ridge(alpha=10.0)
    # clf = linear_model.LassoLars(alpha=.1)
    # print clf.alpha_
    # print clf.coef_
    # right=0
    # total=0
    # for i in range(len(dat_p_te)):
    # 	total+=1
    # 	price_pre=sum(np.array(dat_test[i])*np.array(clf.coef_))
    # 	err=abs(price_pre-dat_p_te[i])
    # 	if err<=10:
    # 		right+=1

    # 	print price_pre,dat_p_te[i]
    # 	#print price,infrustration,price_rate
    # precision=right/total
    # print precision
    # filename='../data/room_infoRome--Italy.csv'
    # filenames=read_filename('../locations.txt')
    # for i in filenames:
    # 	filer='../data/room_info'+i+'.csv'
    # 	filew='../data/descri_'+i+'.csv'
    # 	statics_voc(filer,filew)
    # train,prices,test,test_price=price_infrustration(filename)
    # #sklearn_kmeans(train,prices)
    # basic_inf(train,prices)
