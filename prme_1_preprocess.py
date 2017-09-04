# coding:utf-8
import time

# step1.对用户和POI重新编号
user_dic = {}
user_set = set()
poi_dic = {}
poi_loc_dic = {}
poi_set = set()
# 加入poi的地理位置
poi_info = {}

# read lat lon
with open('./spot_location.txt') as fread:
	lines = fread.readlines()
	for line in lines:
		temp = line.strip().split(' ')
		poiid, lat, lon = temp[0], temp[1], temp[2]
		poi_info[poiid] = [float(lat), float(lon)]
	# print poi_info


with open('./data.txt','rb') as fread:
	lines = fread.readlines()
	for line in lines:
		temp = line.strip().split('\t')
		userid, poiid, timestamp = temp[0], temp[1], temp[2]
		user_set.add(userid)
		poi_set.add(poiid)
	user_list = list(set(user_set))
	poi_list = list(set(poi_set))
	# print user_list, poi_list
	# user_num: 606  poi_num: 2465
	# print len(user_list), len(poi_list)

user_count = 1
for i in user_list:
	user_dic[i] = user_count
	user_count += 1
# print user_dic

poi_count = 1
for j in poi_list:
	poi_dic[j] = poi_count
	poi_count += 1
# print poi_dic


# step2.对数据集按新的编号写入
from datetime import datetime, timedelta

dataset = []

with open('./data.txt','rb') as fread:
	lines = fread.readlines()
	for line in lines:
		temp = line.strip().split('\t')
		userid, poiid, timestamp = temp[0], temp[1], temp[2]
		userid = user_dic[userid]
		poiid = poi_dic[poiid]
		# ts = datetime.strptime(str(timestamp), '%Y-%m-%d %H:%M:%S')
		# print userid, poiid, timestamp
		dataset.append([userid, poiid, timestamp])
		# fwrite.write()

import operator
dataset.sort(key=operator.itemgetter(0,2))

# print dataset
poilatlon_set = set()
with open('./data_v1.txt', 'wb') as fwrite:
	for i in range(len(dataset)-1):
		temp_cu = dataset[i]
		temp_nx = dataset[i+1]
		userid_cu, poiid_cu, timestamp_cu = temp_cu[0], temp_cu[1], temp_cu[2]
		userid_nx, poiid_nx, timestamp_nx = temp_nx[0], temp_nx[1], temp_nx[2]
		poiid_ori_cu = list(poi_dic.keys())[list(poi_dic.values()).index(poiid_cu)]
		poiid_ori_nx = list(poi_dic.keys())[list(poi_dic.values()).index(poiid_nx)]
		# print poiid_ori_cu, poiid_ori_nx
		# print str(poi_info[poiid_ori_cu])
		# poilatlon_set.add([poiid_cu, poi_info[poiid_ori_cu]])
		# poilatlon_set.add([poiid_nx, poi_info[poiid_ori_nx]])
		if userid_cu == userid_nx:
			fwrite.write(str(userid_cu)+'\t'+ \
				str(poiid_cu)+'\t'+ \
				str(poi_info[poiid_ori_cu])+'\t'+ \
				str(timestamp_cu)+'\t'+ \
				str(poiid_nx)+'\t'+ \
				str(poi_info[poiid_ori_nx])+'\t'+ \
				str(timestamp_nx)+'\n')