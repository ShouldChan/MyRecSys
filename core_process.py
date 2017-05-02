# coding:utf-8
from pylab import *

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

def sklearn_regression(inf_train,prices):
    arr=[0]*5
    arr=basic_inf()