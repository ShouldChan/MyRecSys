# coding: utf-8
import operator
from contextlib import nested

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

dataset_name = "SIN"
data_dir = "./data/"+dataset_name+"/"
# gowalla  n_users:5073   n_pois:  
# CA  n_users:2031 n_pois:  3112
# brightkite n_users:1850 N_POIS: 1672
# SIN n_users: 2321 n_pois: 5596

def read_userpoiNum():
    train_list = []
    test_list = []
    user_set = set()
    poi_set = set()
    fw = open(data_dir+dataset_name+".txt", 'w')
    with nested(open(data_dir+dataset_name+"_train.txt",'r'),open(data_dir+dataset_name+"_test.txt",'r')) as (fr_tr,fr_te):
        tr_lines = fr_tr.readlines()
        te_lines = fr_te.readlines()
        for line in tr_lines:
            temp = line.strip().split('\t')
            user, poi, latlon, timestamp = temp[0],temp[1],temp[2],int(temp[3])
            train_list.append([user, poi, latlon, timestamp])
        for line in te_lines:
            temp = line.strip().split('\t')
            user, poi, latlon, timestamp = temp[0],temp[1],temp[2],int(temp[3])
            test_list.append([user, poi, latlon, timestamp])
        dataset = train_list + test_list
        print(len(dataset))
        dataset.sort(key = operator.itemgetter(0, 3))

        for i in range(0, len(dataset)):
            user, poi = dataset[i][0], dataset[i][1]
            user_set.add(user)
            poi_set.add(poi)
            fw.write(str(dataset[i][0])+'\t'+str(dataset[i][1]) \
                +'\t'+str(dataset[i][2])+'\t'+str(dataset[i][3])+'\n')

    user_num, poi_num = len(user_set),len(poi_set)

    fw.close()
    return user_num,poi_num
# -----------------STEP 1-----------------
# ----------first runing need to fw.write()--------------
# ------after runing this, next time don't need to run again-------
user_num, poi_num = read_userpoiNum()
print("user_num:%d\tpoi_num:%d"%(user_num,poi_num))
# -------------------------------------------------------

def make_trainingset():
    trainingset = []
    with open(data_dir+dataset_name+"_train.txt",'r') as fread:
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
    with open(data_dir+"nextpoi_"+dataset_name+"_train.txt",'w') as fw:
        for each_data in trainingset:
            user,poi,lat,lon,poi_nx,lat_nx,lon_nx,interval = each_data
            fw.write(str(user)+'\t'+str(poi)+'\t'+str(lat)+'\t'+ \
                str(lon)+'\t'+str(poi_nx)+'\t'+str(lat_nx)+'\t'+ \
                str(lon_nx)+'\t'+str(interval)+'\n')
# --------------------STEP 2-------------------
make_trainingset()

def make_testset():
    testset = []
    with open(data_dir+dataset_name+"_test.txt",'r') as fread:
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
    with open(data_dir+"nextpoi_"+dataset_name+"_test.txt",'w') as fw:
        for each_data in testset:
            user,poi,lat,lon,poi_nx,lat_nx,lon_nx,interval = each_data
            fw.write(str(user)+'\t'+str(poi)+'\t'+str(lat)+'\t'+ \
                str(lon)+'\t'+str(poi_nx)+'\t'+str(lat_nx)+'\t'+ \
                str(lon_nx)+'\t'+str(interval)+'\n')

# --------------------STEP 2-------------------
make_testset()

def make_poiset():
    poi_set = set()
    with open(data_dir+dataset_name+".txt",'r') as fr:
        lines = fr.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            poi, latlon = temp[1],temp[2]
            lat, lon = latlon.strip().split(',')
            poi_set.add((poi,lat,lon))
    print(len(poi_set))
    with open(data_dir+dataset_name+"_poiset.txt",'w') as fw:
        for each_data in poi_set:
            poi, lat, lon = each_data
            fw.write(str(poi)+'\t'+str(lat)+'\t'+str(lon)+'\n')

# --------------------STEP 2-------------------
make_poiset()

