# coding:utf-8
import time


# step1.对用户和POI重新编号
user_dic = {}
user_set = set()
poi_dic = {}
poi_set = set()

with open('./data.txt','rb') as fread:
	lines = fread.readlines()
	for line in lines:
		temp = line.strip().split('\t')
		userid, poiid, timestamp = temp[0], temp[1], temp[2]
		user_set.add(userid)
		poi_set.add(poiid)
	user_list = list(set(user_set))
	poi_list = list(set(poi_set))
	print user_list, poi_list
	# user_num: 606  poi_num: 2465
	print len(user_list), len(poi_list)

user_count = 1
for i in user_list:
	user_dic[user_count] = i
	user_count += 1
print user_dic

poi_count = 1
for j in poi_list:
	poi_dic[poi_count] = j
	poi_count += 1
print poi_dic