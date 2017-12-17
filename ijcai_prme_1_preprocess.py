# coding: utf-8


# ┏┓   ┏┓
# ┏┛┻━━━┛┻┓
# ┃    ☃   ┃
# ┃ ┳┛ ┗┳ ┃
# ┃   ┻    ┃
# ┗━┓     ┏━┛
# ┃     ┗━━━┓
# ┃  神兽保佑  ┣┓
# ┃　永无BUG！ ┏┛
# ┗┓┓┏━┳┓┏┛
# ┃┫┫  ┃┫┫
# ┗┻┛  ┗┻┛

def read_userpoiNum():
    user_set = set()
    poi_set = set()
    with open(data_dir+"gowalla.txt",'r') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            user, poi=temp[0],temp[1]
            user_set.add(user)
            poi_set.add(poi)
    user_num, poi_num = len(user_set),len(poi_set)
    return user_num,poi_num

user_num, poi_num=read_userpoiNum()
print("user_num:%d\tpoi_num:%d"%(user_num,poi_num))

def make_trainingset():
    trainingset = []
    with open(data_dir+"gowalla_train.txt",'r') as fread:
        lines = fread.readlines()
        for i in range(len(lines)):
            temp = lines[i].strip().split('\t')
            user, poi, latlon, seconds = temp[0], temp[1], temp[2], int(temp[3])
            lat,lon = latlon.strip().split(',')[0],latlon.strip().split(',')[1]
            if i != len(lines)-1:
                temp_nx = lines[i+1].strip().split('\t')
                user_nx, poi_nx, latlon_nx, seconds_nx = temp_nx[0], temp_nx[1], temp_nx[2], int(temp_nx[3])
                lat_nx,lon_nx = latlon_nx.strip().split(',')[0],latlon_nx.strip().split(',')[1]
            if user == user_nx:
                interval = seconds_nx - seconds
                trainingset.append([user,poi,lat,lon,poi_nx,lat_nx,lon_nx,interval])
    print(len(trainingset))
    with open(data_dir+"nextpoi_gowalla_train.txt",'w') as fw:
        for each_data in trainingset:
            user,poi,lat,lon,poi_nx,lat_nx,lon_nx,interval = each_data
            fw.write(str(user)+'\t'+str(poi)+'\t'+str(lat)+'\t'+ \
                str(lon)+'\t'+str(poi_nx)+'\t'+str(lat_nx)+'\t'+ \
                str(lon_nx)+'\t'+str(interval)+'\n')

# step-1
# make_trainingset()

def make_testset():
    testset = []
    with open(data_dir+"gowalla_test.txt",'r') as fread:
        lines = fread.readlines()
        for i in range(len(lines)):
            temp = lines[i].strip().split('\t')
            user, poi, latlon, seconds = temp[0], temp[1], temp[2], int(temp[3])
            lat,lon = latlon.strip().split(',')[0],latlon.strip().split(',')[1]
            if i != len(lines)-1:
                temp_nx = lines[i+1].strip().split('\t')
                user_nx, poi_nx, latlon_nx, seconds_nx = temp_nx[0], temp_nx[1], temp_nx[2], int(temp_nx[3])
                lat_nx,lon_nx = latlon_nx.strip().split(',')[0],latlon_nx.strip().split(',')[1]
            if user == user_nx:
                interval = seconds_nx - seconds
                testset.append([user,poi,lat,lon,poi_nx,lat_nx,lon_nx,interval])
    print(len(testset))
    with open(data_dir+"nextpoi_gowalla_test.txt",'w') as fw:
        for each_data in testset:
            user,poi,lat,lon,poi_nx,lat_nx,lon_nx,interval = each_data
            fw.write(str(user)+'\t'+str(poi)+'\t'+str(lat)+'\t'+ \
                str(lon)+'\t'+str(poi_nx)+'\t'+str(lat_nx)+'\t'+ \
                str(lon_nx)+'\t'+str(interval)+'\n')

# step-2
# make_testset()

def make_poiset():
    poi_set = set()
    with open(data_dir+"gowalla.txt",'r') as fr:
        lines = fr.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            poi, latlon = temp[1],temp[2]
            lat, lon = latlon.strip().split(',')
            poi_set.add((poi,lat,lon))
    print(len(poi_set))
    with open(data_dir+"gowalla_poiset.txt",'w') as fw:
        for each_data in poi_set:
            poi, lat, lon = each_data
            fw.write(str(poi)+'\t'+str(lat)+'\t'+str(lon)+'\n')
# step-3
# make_poiset()

