import time
import re

def date_transfer(date):
	months = {'Jan':1,'Feb':2,'Mar':3,'Apr':4, \
	'May':5,'Jun':6,'Jul':7,'Aug':8, \
	'Sep':9,'Oct':10,'Nov':11,'Dec':12}
	results = date.strip().split(' ')
	y = results[-1]
	m = months[results[1]]
	d = results[2]
	t = results[3]
	new_date = str(y)+'-'+str(m)+'-'+str(d)+' '+str(t)
	# timeArray = time.strptime(new_date,"%Y-%m-%d %H:%M:%S")
	# timeStamp = int(time.mktime(timeArray))
	return new_date

# print date_transfer('Thu Jul 28 00:05:26 +0000 2011')
# userID	Time(GMT)	venueID	VenueName	VenueLocation	VenueCategory
fwrite = open('./new_ca.txt','wb')
with open('./ca.txt','rb') as fread:
	lines = fread.readlines()
	for line in lines:
		temp = line.strip().split('\t')
		uid, time, poiid,poiloc=temp[0],temp[1],temp[2],temp[4]
		# print time
		time = date_transfer(time)
		# print time
		# print poiloc
		poiloc = re.findall(r'\d+\.?\d*',poiloc)
		# print poiloc
		fwrite.write(str(uid)+'\t'+str(poiid)+'\t'+str(time)+'\t'+ \
			str(poiloc[0])+'\t'+str(poiloc[1])+'\n')

fwrite.close()