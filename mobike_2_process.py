# coding:utf-8

# test & get useful train data
with open('./train_next.csv','rb') as fread:
    lines = fread.readlines()
    for line in lines:
        temp = line.strip().split(',')
        userid, starttime, geohashed_start_loc,  \
        geohashed_end_loc = str(temp[1]), str(temp[4]), \
        str(temp[5]), str(temp[6])
        print userid, starttime, geohashed_start_loc

# make the mapped latlon txtfile
fwrite = open('./latlon_v2.txt','wb')
with open('./latlon.txt','rb') as fread:
    count = 0
    lines = fread.readlines()
    for line in lines:
        temp = line.strip().split('\t')
        locid, lat, lon = temp[0],temp[1],temp[2]
        fwrite.write(str(count)+'\t'+ \
            str(locid)+'\t'+str(lat)+'\t'+ \
            str(lon)+'\n')
        print count
        count += 1

# make unique user
user_set = set()
with open('./train_next.csv','rb') as fread:
    lines = fread.readlines()
    for line in lines:
        temp = line.strip().split(',')
        userid = str(temp[1])
        user_set.add(userid)

fwrite = open('./userid.txt','wb')
count = 0
for i in user_set:
    print count
    fwrite.write(str(count)+'\t'+str(i)+'\n')
    count += 1