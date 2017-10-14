uid_set = set()
pid_set = set()

with open('./new_ca.txt','rb') as fread:
	lines = fread.readlines()
	
	for line in lines:
		temp = line.strip().split('\t')
		uid, pid, time, lat,lon=temp[0],temp[1],temp[2],temp[3],temp[4]
		uid_set.add(uid)
		pid_set.add(pid)

uid_dict = {}
pid_dict = {}
user_count = 0
poi_count = 0
for uid in uid_set:
	uid_dict[uid] = user_count
	user_count += 1
for pid in pid_set:
	pid_dict[pid] = poi_count
	poi_count += 1

fwrite = open('./new2_ca.txt','wb')
with open('./new_ca.txt','rb') as fread:
	lines = fread.readlines()
	
	for line in lines:
		temp = line.strip().split('\t')
		uid, pid, time, lat,lon=temp[0],temp[1],temp[2],temp[3],temp[4]
		uid = uid_dict[uid]
		pid = pid_dict[pid]
		fwrite.write(str(uid)+'\t'+str(pid)+'\t'+str(time)+ \
			'\t'+str(lat)+'\t'+str(lon)+'\n')
fwrite.close()

print len(uid_set) #2093
print len(pid_set) #3518