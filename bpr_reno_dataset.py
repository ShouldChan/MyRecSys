uid_set = set()
mid_set = set()
with open('./rating.txt','rb') as fread:
    lines = fread.readlines()
    for line in lines:
        temp = line.strip().split('\t')
        uid, mid, rate, timestamp = int(temp[0]), int(temp[1]),temp[2],temp[3]
        uid_set.add(uid)
        mid_set.add(mid)
print len(uid_set),len(mid_set)

uid_dict = {}
uid_count = 0
for uid in uid_set:
    uid_dict[uid_count] = uid
    uid_count += 1

mid_dict = {}
mid_count = 0
for mid in mid_set:
    mid_dict[mid_count] = mid
    mid_count += 1

uid_reve_dict = dict((value,key) for key,value in uid_dict.iteritems())
mid_reve_dict = dict((value,key) for key,value in mid_dict.iteritems())

reno_list=[]
with open('./rating.txt','rb') as fread:
    lines = fread.readlines()
    for line in lines:
        temp = line.strip().split('\t')
        uid, mid, rate, timestamp = int(temp[0]), int(temp[1]),temp[2],temp[3]
        uid = uid_reve_dict[uid]
        mid = mid_reve_dict[mid]
        reno_list.append([uid, mid, rate, timestamp])

import operator
reno_list.sort(key=operator.itemgetter(0,1))

fwrite = open('./reno_ratings.txt','wb')
for uid, mid, rate, timestamp in reno_list:
    fwrite.write(str(uid)+'\t'+str(mid)+'\t'+str(rate)+'\t'+str(timestamp)+'\n')
fwrite.close()