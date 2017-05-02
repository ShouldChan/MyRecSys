# coding:utf-8
import matlab.engine
from statics import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

eng = matlab.engine.start_matlab()

def distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dis = -sum((vec1 - vec2) * (vec1 - vec2)) #作者源码
    # dis = np.sqrt(np.sum(np.square(vec1 - vec2))) #我的改动
    return dis

# def draw_land_ap(fname, num, fig):
#     # filename='../data/landmark-'+fname+'.txt'
#     font = FontProperties(family='Times New Roman', size=14, weight='normal', style='normal')
#     filename_landmark = './landmarks/landmark-Los-Angeles--CA.txt'
#     filename = './houses/listings.csv'
#     landmark_lat = []
#     landmark_lon = []
#     house_lat = []
#     house_lon = []
#     landmarks = []
#     landmarks_name = []
#     landmarks_value = []
#     houses = []
#     dis_all = []
#     for j in open(filename, 'r').readlines():
#         h_lat = float(j.strip().split('\t')[2])
#         h_lon = float(j.strip().split('\t')[3])
#         house_lat.append(h_lat)
#         house_lon.append(h_lon)
#         houses.append([h_lat, h_lon])
#
#     range_w_1 = max(house_lat) + 0.05
#     range_w_2 = min(house_lat) - 0.05
#     range_l_1 = max(house_lon) + 0.05
#     range_l_2 = min(house_lon) - 0.05
#     for i in open(filename_landmark, 'r').readlines():
#         lat = float(i.strip().split('\t')[1])
#         lon = float(i.strip().split('\t')[2])
#         name = i.strip().split('\t')[0]
#         value = i.strip().split('\t')[3]
#         if lat >= range_w_2 and lat <= range_w_1 and lon >= range_l_2 and lon <= range_l_1:
#             landmark_lat.append(lat)
#             landmark_lon.append(lon)
#             landmarks.append([lat, lon])
#             landmarks_name.append(name)
#             landmarks_value.append(value)
#     k = 0
#     n = len(landmarks)
#     m = n * n - n
#     s = eng.zeros(m, 3)
#     for i in range(len(landmarks)):
#         for j in range(len(landmarks)):
#             if j != i:
#                 dis = distance(landmarks[i], landmarks[j])
#                 dis_all.append(dis)
#                 s[k][0] = i + 1
#                 s[k][1] = j + 1
#                 s[k][2] = dis
#                 k = k + 1
#     p = sorted(dis_all)[len(dis_all) / 2]
#     p = matlab.double([p])
#     idx, netsim, dpsim, expref = eng.apclusterSparse(s, p, nargout=4)
#     idx = list(idx)
#
#     filwname = '../result_ap/landmark_' + fname + '.txt'
#     filw = open(filwname, 'a')
#     print len(idx), len(landmarks), fname
#     plt.subplot(3, 3, num)
#     font = FontProperties(family='Times New Roman', size=14, weight='normal', style='normal')
#     colors = ['#293047', '#473c8b', '#007947', '#8f4b38', '#FF0000', "#CD2626", "#9932CC", "#8B1A1A", "#0A0A0A",
#               "#0000AA", "#008B00", "#9Aff9a", "#CDAD00", "#FF8c00", "#ff1493", "#FFFF00", "#EE00EE", "#bf3eff"]
#     radius = [0] * len(landmarks)
#     for i in range(len(idx)):
#         col = colors[(int(idx[i][0]) - 1) % len(colors)]
#         # crll=crl[(int(idx[i][0])-1)%6]
#         # plt.plot(landmarks[i][0],landmarks[i][1],markerfacecolor=col,marker='*')
#         plt.plot(landmarks[int(idx[i][0]) - 1][0], landmarks[int(idx[i][0]) - 1][1], markerfacecolor=col, marker='*')
#         # plt.plot((landmarks[i][0],landmarks[int(idx[i][0])-1][0]),(landmarks[i][1],landmarks[int(idx[i][0])-1][1]),col)
#         plt.title(fname, fontproperties=font)
#         plt.xlabel("Latitude", fontproperties=font)
#         plt.ylabel("Longitude", fontproperties=font)
#         rad = math.sqrt(sum((landmarks[i][0] - landmarks[int(idx[i][0]) - 1][0]) ** 2 + (
#         landmarks[i][1] - landmarks[int(idx[i][0]) - 1][1]) ** 2))
#         if rad >= radius[int(idx[i][0]) - 1]:
#             radius[int(idx[i][0]) - 1] = rad
#
#
#         # filw.write(str(landmarks_name[i])+'\t')
#         # filw.write(str(landmarks_value[i])+'\t')
#         # filw.write(str(landmarks[i][0])+'\t')
#         # filw.write(str(landmarks[i][1])+'\t')
#         # filw.write(str(landmarks_name[int(idx[i][0])-1])+'\t')
#         # filw.write(str(landmarks_value[int(idx[i][0])-1])+'\t')
#         # filw.write(str(landmarks[int(idx[i][0])-1][0])+'\t')
#         # filw.write(str(landmarks[int(idx[i][0])-1][1])+'\n')
#     plt.plot(house_lat,house_lon,'g.')
#     for i_co in range(len(landmarks)):
#         co = colors[(int(idx[i_co][0]) - 1) % len(colors)]
#         plot_circle(landmarks[int(idx[i_co][0]) - 1][0], landmarks[int(idx[i_co][0]) - 1][1],
#                     radius[int(idx[i_co][0]) - 1], fig, co, 3, 3, num)
#     plt.xticks(fontproperties=font)
#     plt.yticks(fontproperties=font)
#
def plot_circle(xx, yy, r, fig, col, num_1, num_2, num):
    from matplotlib.patches import Circle  # ,Ellipse
    # fig = plt.figure()
    ax = fig.add_subplot(3, 3, num)
    # ax = fig.add_subplot(num_1, num_2, num)

    # ell1 = Ellipse(xy = (x, y), width = 4, height = 8, angle = 30.0, facecolor= 'yellow', alpha=0.3)
    cir1 = Circle(xy=(xx, yy), radius=r, alpha=0.05)
    # ax.add_patch(ell1)
    ax.add_patch(cir1)

    x, y = xx, yy
    ax.plot(x, y, markerfacecolor=col, marker='*')

    plt.axis('scaled')
    # ax.set_xlim(-4, 4)
    # ax.set_ylim(-4, 4)
    plt.axis('equal')  # changes limits of x or y axis so that equal increments of x and y have the same length
def plot_cluster(x1, y1, num, idx, landmarks, radius, cluster_popular, popular_all, fig, fname):
    plt.subplot(x1, y1, num)
    font = FontProperties(family='Times New Roman', size=14, weight='normal', style='normal')
    colors = ['#293047', '#473c8b', '#007947', '#8f4b38', '#FF0000', "#CD2626", "#9932CC", "#8B1A1A", "#0A0A0A",
              "#0000AA", "#008B00", "#9Aff9a", "#CDAD00", "#FF8c00", "#ff1493", "#FFFF00", "#EE00EE", "#bf3eff"]
    # radius=[0]*len(landmarks)
    for i in range(len(idx)):
        col = colors[(int(idx[i][0]) - 1) % len(colors)]
        # crll=crl[(int(idx[i][0])-1)%6]
        # plt.plot(landmarks[i][0],landmarks[i][1],markerfacecolor=col,marker='*')
        plt.plot(landmarks[int(idx[i][0]) - 1][0], landmarks[int(idx[i][0]) - 1][1], markerfacecolor=col, marker='*')
        # plt.plot((landmarks[i][0],landmarks[int(idx[i][0])-1][0]),(landmarks[i][1],landmarks[int(idx[i][0])-1][1]),col)
        plt.title(fname, fontproperties=font)
        plt.xlabel("Latitude", fontproperties=font)
        plt.ylabel("Longitude", fontproperties=font)
        rad = math.sqrt(sum((landmarks[i][0] - landmarks[int(idx[i][0]) - 1][0]) ** 2 + (
            landmarks[i][1] - landmarks[int(idx[i][0]) - 1][1]) ** 2))
        radius[int(idx[i][0]) - 1].append(rad)
    # if rad>=radius[int(idx[i][0])-1]:
    # 	radius[int(idx[i][0])-1]=rad

    for i_co in range(len(landmarks)):
        co = colors[(int(idx[i_co][0]) - 1) % len(colors)]
        # print "print the circles..."
        beta = (cluster_popular[i_co] / popular_all) * 10
        # print beta*10
        radd = sorted(radius[int(idx[i_co][0]) - 1])[len(radius[int(idx[i_co][0]) - 1]) - 1] * (1 + beta)
        plot_circle(landmarks[int(idx[i_co][0]) - 1][0], landmarks[int(idx[i_co][0]) - 1][1], radd, fig, co, x1, y1,
                    num)
    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)


def Hoffman_code_all(matrix):
    from utils import Huffman_coding
    ma_x = []
    ma_y = []
    ma = []
    for i in range(len(matrix)):
        x = Huffman_coding(matrix[i][0:15])
        y = Huffman_coding(matrix[i][15:30])
        ma_x.append(x)
        ma_y.append(y)
        ma.append([math.log(x + 1), math.log(y + 1)])

    return ma_x, ma_y, ma


def test(fname, fig, num_1, num_2, num):
    from statics import infrustration
    # filename='../data/landmark-'+fname+'.txt'
    font = FontProperties(family='Times New Roman', size=14, weight='normal', style='normal')
    filename_landmark = './landmarks/landmark-Los-Angeles--CA.txt'
    filename = './houses/houseinfo.txt'

    landmark_lat = []
    landmark_lon = []
    house_lat = []
    house_lon = []
    house_id = []
    # house_price=[]
    landmarks = []
    landmarks_name = []
    landmarks_value = []
    houses = []
    house_value = []
    dis_all = []
    for j in open(filename, 'r').readlines():
        h_id = j.strip().split('\t')[1]
        h_lat = float(j.strip().split('\t')[2])
        h_lon = float(j.strip().split('\t')[3])
        # h_price=j.strip().split('\t')[4]
        house_lat.append(h_lat)
        house_lon.append(h_lon)
        houses.append([h_lat, h_lon])
        house_id.append(h_id)
        house_value.append(j.strip().split('\t')[5])
    # house_price.append(h_price)

    range_w_1 = max(house_lat) + 0.05
    range_w_2 = min(house_lat) - 0.05
    range_l_1 = max(house_lon) + 0.05
    range_l_2 = min(house_lon) - 0.05
    for i in open(filename_landmark, 'r').readlines():
        lat = float(i.strip().split('\t')[1])
        lon = float(i.strip().split('\t')[2])
        name = i.strip().split('\t')[0]
        value = i.strip().split('\t')[3]
        if lat >= range_w_2 and lat <= range_w_1 and lon >= range_l_2 and lon <= range_l_1:
            landmark_lat.append(lat)
            landmark_lon.append(lon)
            landmarks.append([lat, lon])
            landmarks_name.append(name)
            landmarks_value.append(float(value))
    popular_all = sum(landmarks_value)

    ##Step one: cluster the landmarks into serveral price-zone

    k = 0
    n = len(landmarks)
    m = n * n - n
    s = eng.zeros(m, 3)
    for i in range(len(landmarks)):
        for j in range(len(landmarks)):
            if j != i:
                dis = haversine(landmarks[i], landmarks[j])
                dis_all.append(dis)
                s[k][0] = i + 1
                s[k][1] = j + 1
                s[k][2] = dis
                k = k + 1
    p = sorted(dis_all)[len(dis_all) / 2]
    p = matlab.double([p])
    idx, netsim, dpsim, expref = eng.apclusterSparse(s, p, nargout=4)
    idx = list(idx)

    ##store the each cluster member
    radius = []
    cluster = []
    for aa in range(len(idx)):
        cluster.append([])
        radius.append([])
    for kd in range(len(idx)):
        # if cluster[idx[kd]-1]==[]:
        # cluster[idx[kd]-1].append()
        cluster[int(idx[kd][0]) - 1].append(kd)
    # print int(idx[kd][0])-1

    # calculate the attractive of every cluster by using the landmarks_value
    cluster_popular = [0] * len(idx)
    for ki in range(len(idx)):
        if cluster[ki] != []:
            # print "cluster",cluster[ki]
            for kj in range(len(cluster[ki])):
                # print kj,cluster[ki][kj]
                cluster_popular[ki] += landmarks_value[cluster[ki][kj]]

    filwname = '../data/result_ap/landmark_' + fname + '.txt'
    filw = open(filwname, 'a')
    # print len(idx),len(landmarks),fname,netsim,dpsim,expref
    plot_cluster(num_1, num_2, num, idx, landmarks, radius, cluster_popular, popular_all, fig, fname)

    filw_house_cluster = open('../data/result_ap/cluster2_house_' + fname + '.txt', 'a')

    ##Step two: move each house into the nearest price-zone

    house_cluster_list = []
    house_cluster = []
    hosue_dis2lanM = []
    hosue_dis2lanM_Near = []
    hosue_dis2lanM_popu = []
    for h in range(len(houses)):
        house_cluster_list.append([])
        hosue_dis2lanM.append([])

    for hi in range(len(houses)):
        for hj in range(len(idx)):
            rela_dis = math.sqrt(distance(houses[hi], landmarks[int(idx[hj][0]) - 1]))
            # if rela_dis<=max(radius[int(idx[hj][0])-1]):
            house_cluster_list[hi].append(int(idx[hj][0]) - 1)
            hosue_dis2lanM[hi].append(rela_dis)
        # else:pass
        # filw_house_cluster.write(house_id[hi]+'\t')
        # filw_house_cluster.write(str(house_price[hi])+'\t')
        # filw_house_cluster.write(str(house_lat[hi])+'\t')
        # filw_house_cluster.write(str(house_lon[hi])+'\t')
        # #filw_house_cluster.write(str(len(hosue_dis2lanM[hi]))+'\t')#the number of cluster that contain the house
        # try:
        # 	filw_house_cluster.write(str(min(hosue_dis2lanM[hi]))+'\t')
        # 	filw_house_cluster.write(str(landmark_lat[int(idx[hosue_dis2lanM[hi].index(min(hosue_dis2lanM[hi]))][0])-1])+'\t')
        # 	filw_house_cluster.write(str(landmark_lon[int(idx[hosue_dis2lanM[hi].index(min(hosue_dis2lanM[hi]))][0])-1])+'\t')
        # 	filw_house_cluster.write(str(int(idx[hosue_dis2lanM[hi].index(min(hosue_dis2lanM[hi]))][0])-1)+'\t')
        # except Exception,e:
        # 	pass

        # filw_house_cluster.write('\n')
        # move every house to the nearest landmark price_zone
        # store the index of the landmark that is the closest to the house point
        house_cluster.append(int(idx[hosue_dis2lanM[hi].index(min(hosue_dis2lanM[hi]))][0]) - 1)
        hosue_dis2lanM_Near.append(min(hosue_dis2lanM[hi]))
        hosue_dis2lanM_popu.append(
            landmarks_value[int(idx[hosue_dis2lanM[hi].index(min(hosue_dis2lanM[hi]))][0]) - 1] / popular_all)
    ##Step three: In each price-zone, cluster the house by using the facilities.

    infrust, house_price = infrustration(filename)
    infrust_x, infrust_y, infrust_ma = Hoffman_code_all(infrust)
    # print len(infrust),len(house_cluster),len(house_cluster_list)
    house_cluster_np = np.array(house_cluster)
    house_price_np = np.array(house_price)
    house_value_np = np.array(house_value)
    hosue_dis2lanM_np = np.array(hosue_dis2lanM)
    infrust_x_np = np.array(infrust_x)
    infrust_y_np = np.array(infrust_y)
    infrust_ma_np = np.array(infrust_ma)
    hosue_dis2lanM_Near_np = np.array(hosue_dis2lanM_Near)
    hosue_dis2lanM_popu_np = np.array(hosue_dis2lanM_popu)
    houses_np = np.array(houses)
    houses_id_np = np.array(house_id)
    filw_house_facility_cluster = open('../data/result_ap/facility7_cluster_' + fname + '.csv', 'a')
    kkk = 0
    for infk in list(set(house_cluster)):
        infrust_x_np_cluster = infrust_x_np[house_cluster_np == infk]
        infrust_y_np_cluster = infrust_y_np[house_cluster_np == infk]
        house_price_np_cluster = house_price_np[house_cluster_np == infk]
        house_value_np_cluster = house_value_np[house_cluster_np == infk]
        infrust_ma_np_cluster = infrust_ma_np[house_cluster_np == infk]
        houses_np_cluster = houses_np[house_cluster_np == infk]
        houses_id_np_cluster = houses_id_np[house_cluster_np == infk]
        hosue_dis2lanM_Near_np_cluster = hosue_dis2lanM_Near_np[house_cluster_np == infk]
        hosue_dis2lanM_popu_np_cluster = hosue_dis2lanM_popu_np[house_cluster_np == infk]
        inf_n = len(infrust_x_np_cluster)
        inf_m = inf_n ** 2 - inf_n
        inf_k = 0
        inf_s = eng.zeros(inf_m, 3)
        inf_dis_all = []
        price_zone_name = landmarks_name[int(infk) - 1]
        try:
            for infi in range(len(infrust_x_np_cluster)):
                for infj in range(len(infrust_y_np_cluster)):
                    if infi != infj:
                        inf_dis = distance(infrust_ma_np_cluster[infi], infrust_ma_np_cluster[infj])
                        inf_dis_all.append(inf_dis)

                        inf_s[inf_k][0] = infi + 1
                        inf_s[inf_k][1] = infj + 1
                        inf_s[inf_k][2] = inf_dis
                        inf_k = inf_k + 1

            inf_p = sorted(inf_dis_all)[len(inf_dis_all) / 2]

            inf_p = matlab.double([inf_p])
            inf_idx_raw, inf_netsim, inf_dpsim, inf_expref = eng.apclusterSparse(inf_s, inf_p, nargout=4)
            inf_idx = []
            for infii in range(len(inf_idx_raw)):
                inf_idx.append(inf_idx_raw[infii][0])

            # print houses_np_cluster,inf_idx
            # print "Drawing the ",infk,len(houses_np_cluster),len(inf_idx)
            colors = ['#293047', '#473c8b', '#007947', '#8f4b38', '#FF0000', "#CD2626", "#9932CC", "#8B1A1A", "#0A0A0A",
                      "#0000AA", "#008B00", "#9Aff9a", "#CDAD00", "#FF8c00", "#ff1493", "#FFFF00", "#EE00EE", "#bf3eff"]
            col = colors[kkk % len(colors)]
            # plot_infrust_cluster(inf_idx,infrust_ma_np_cluster,price_zone_name)
            kkk += 1
            # plot_infrust_cluster(inf_idx,houses_np_cluster,price_zone_name,col)#
            # plot_house_point(inf_idx,houses_np_cluster,price_zone_name)#dot the green dots
            # plt.plot(houses,markerfacecolor='g',marker='.')
            # plt.show()
            for inf_idx_facility_cluster in list(set(inf_idx)):
                inf_idx = np.array(inf_idx)
                # print inf_idx_facility_cluster,inf_idx==inf_idx_facility_cluster,inf_idx

                house_facility_cluster_ma = infrust_ma_np_cluster[inf_idx == inf_idx_facility_cluster]
                house_facility_cluster_price = house_price_np_cluster[inf_idx == inf_idx_facility_cluster]
                house_facility_cluster_value = house_value_np_cluster[inf_idx == inf_idx_facility_cluster]
                house_facliity_cluster_geo = houses_np_cluster[inf_idx == inf_idx_facility_cluster]
                house_facliity_cluster_id = houses_id_np_cluster[inf_idx == inf_idx_facility_cluster]
                house_facliity_cluster_dis = hosue_dis2lanM_Near_np_cluster[inf_idx == inf_idx_facility_cluster]
                house_facility_cluster_pop = hosue_dis2lanM_popu_np_cluster[inf_idx == inf_idx_facility_cluster]
                aa = 0
                for inf_idx_each_cluster_house_price in range(len(house_facility_cluster_price)):
                    aa += 1
                    # rev,rec=find_review_information(str(house_facliity_cluster_id[inf_idx_each_cluster_house_price]))
                    # print aa,rev,rec
                    filw_house_facility_cluster.write(
                        str(house_facliity_cluster_id[inf_idx_each_cluster_house_price]) + '\t')  # 0
                    # filw_house_facility_cluster.write(rev+'\t')#1
                    # filw_house_facility_cluster.write(rec+'\t')
                    filw_house_facility_cluster.write(price_zone_name + '\t')
                    filw_house_facility_cluster.write(str(inf_idx_facility_cluster) + '\t')  # 4
                    filw_house_facility_cluster.write(
                        str(house_facliity_cluster_geo[inf_idx_each_cluster_house_price][0]) + '\t')  # 5
                    filw_house_facility_cluster.write(
                        str(house_facliity_cluster_geo[inf_idx_each_cluster_house_price][1]) + '\t')  # 6
                    filw_house_facility_cluster.write(
                        str(house_facility_cluster_ma[inf_idx_each_cluster_house_price][0]) + '\t')  # 7
                    filw_house_facility_cluster.write(
                        str(house_facility_cluster_ma[inf_idx_each_cluster_house_price][1]) + '\t')  # 8
                    filw_house_facility_cluster.write(
                        str(house_facility_cluster_value[inf_idx_each_cluster_house_price]) + '\t')  # 9
                    filw_house_facility_cluster.write(
                        str(house_facliity_cluster_dis[inf_idx_each_cluster_house_price]) + '\t')  # 10
                    filw_house_facility_cluster.write(
                        str(house_facility_cluster_pop[inf_idx_each_cluster_house_price]) + '\t')  # 11
                    filw_house_facility_cluster.write(
                        str(house_facility_cluster_price[inf_idx_each_cluster_house_price]) + '\n')  # 12

        except Exception, e:
            pass


fig = plt.figure()
name = ['Los-Angeles--CA']
for i in range(len(name)):
    test(name[i], fig, 1, 3, i + 1)
