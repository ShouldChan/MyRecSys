# coding:utf-8
import numpy as np 
import pandas as pd 
import csv
import datetime

def read_training_data():
    train_data = []
    user_dict = {}
    loc_dict = {}
    # 给地点进行编号
    with open('./latlon_v2.txt', 'rb') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            locid,hash_locid = int(temp[0]),str(temp[1])
            loc_dict[hash_locid] = locid
    # 给用户id进行编号
    with open('./userid.txt','rb') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            u, userid = int(temp[0]),str(temp[1])
            user_dict[userid] = u
    # 读取训练数据集
    with open('./train_next.csv','rb') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split(',')
            userid, starttime, geohashed_start_loc,  \
            geohashed_end_loc = str(temp[1]), str(temp[4]), \
            str(temp[5]), str(temp[6])
            u,ls,le = user_dict[userid],loc_dict[geohashed_start_loc],loc_dict[geohashed_end_loc]
            train_data.append([u,ls,le])
    return train_data


def get_locations():
    locations = {}
    with open('./latlon_v2.txt','rb') as fread:
        lines = fread.readlines()
        for line in lines:
            temp = line.strip().split('\t')
            locations[str(temp[1])] = [float(temp[2]), \
            float(temp[3])]
    return locations

# locations=get_locations()
# print locations

# def learning(train_data, locations):



if __name__ == '__main__':
    # Step1------get_locations  geohash_string  lat  lon
    locations = get_locations()
    train_data = read_training_data()

