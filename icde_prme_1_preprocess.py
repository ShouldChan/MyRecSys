# Step1:   objective; to process in the form of next poi
from util import Util
import operator

util = Util()

time_format = "%Y-%m-%d %H:%M:%S"

checkin_list = []
user_set = set()
poi_set = set()
time_dict = {}

with open('./data/foursquare.txt','rb') as fread:
	lines = fread.readlines()
	for i in range(len(lines)):
		temp = lines[i].strip().split('\t')
		uid,pid,time,lat,lon=int(temp[0]),temp[1],temp[2],temp[3],temp[4]
		timestamp = util.date2time(time,time_format)
		# print type(timestamp)
		time_dict[time] = timestamp
		checkin_list.append([uid,pid,timestamp,lat,lon])
		user_set.add(uid)
		poi_set.add(pid)
		# print uid,pid,timestamp,lat,lon
	# sort by ASC [0]first [2]second
	checkin_list.sort(key=operator.itemgetter(0,2))
	# print checkin_list
	# print time_dict
time_reverse_dict = dict((value,key) for key,value in time_dict.iteritems())

# select next poi
# next_list = []
fwrite = open('./data/foursquare_next.txt','wb')
for i in range(len(checkin_list)):
	uid,pid,timestamp,lat,lon=checkin_list[i]
	if i != len(checkin_list)-1:
		uid_nx,pid_nx,timestamp_nx,lat_nx,lon_nx=checkin_list[i+1]
	else:
		break
	if uid == uid_nx:
		fwrite.write(str(uid)+'\t'+str(pid)+'\t'+str(lat)+ \
			'\t'+str(lon)+'\t'+str(time_reverse_dict[timestamp])+'\t'+str(pid_nx)+'\t'+ \
			str(lat_nx)+'\t'+str(lon_nx)+'\t'+str(time_reverse_dict[timestamp_nx])+'\n')
		# next_list.append([uid,pid,timestamp,lat,lon,pid_nx,timestamp_nx,lat_nx,lon_nx])
		# print uid,pid,timestamp,lat,lon,pid_nx,timestamp_nx,lat_nx,lon_nx
fwrite.close()

print 'user num:\t',len(user_set)  #
print 'poi numL\t',len(poi_set) #


# Step2 objective: make train and test
# mark the last 30% checkin of each user as test
# eachuser_count = {}
# count_eachuser = 0
# for i in range(len(next_list)):
# 	uid,pid,timestamp,lat,lon,pid_nx,timestamp_nx,lat_nx,lon_nx=next_list[i]
# 	if i != len(next_list)-1:
# 		nx_uid,nx_pid,nx_timestamp,nx_lat,nx_lon,nx_pid_nx,nx_timestamp_nx,nx_lat_nx,nx_lon_nx \
# 		=next_list[i+1]
# 	else:
# 		break
# 	if uid == nx_uid:
# 		count_eachuser += 1
# 	else:
# 		eachuser_count[uid] = count_eachuser
# 		count_eachuser = 0
