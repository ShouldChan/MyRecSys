# coding:utf-8
import geohash
import pandas as pd
from pandas import DataFrame, Series
import numpy as np 

train_start = pd.read_csv('./train.csv', usecols=[5])
train_end = pd.read_csv('./train.csv', usecols=[6])
train_start = train_start['geohashed_start_loc']
train_end = train_end['geohashed_end_loc']

test = pd.read_csv('./test.csv', usecols=[5])
test = test['geohashed_start_loc']

# append
train = train_start.append(train_end, ignore_index=True)
data = train.append(test, ignore_index=True)

# 去重 drop_duplicates
data = data.drop_duplicates()

# print data

# tranverse series and decode geohash
print len(data)
with open('./latlon.txt','wb') as fwrite:
	for index, value in data.iteritems():
		latlon_tuple = geohash.decode(value)
		lat = latlon_tuple[0]
		lon = latlon_tuple[1]
		print lat, lon
		fwrite.write(str(value)+'\t'+str(lat)+'\t'+str(lon)+'\n')

