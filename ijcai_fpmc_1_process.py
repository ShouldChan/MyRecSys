# coding:utf-8

dataset_name = "SIN"
train_or_test = "test"
dataset_path = "./dataset/"+dataset_name+"/"+dataset_name+"_"+train_or_test+".txt"
save_path = "./"+dataset_name+"_"+train_or_test+".txt"

def splitDataset():
	checkin_list = []
	userid_set = set()
	with open(dataset_path, 'r') as fr:
		lines = fr.readlines()
		for i in range(0, len(lines)):
			temp = lines[i].strip().split('\t')
			userid, poiid, timestamp = temp[0],temp[1],int(temp[3])
			checkin_list.append([userid, poiid, timestamp])
			userid_set.add(userid)
		print("user nums:\t%d"%len(userid_set))
	
	fw = open(save_path, 'w')
	oldtime = checkin_list[0][2]
	olduserid = checkin_list[0][0]
	fw.write(olduserid+"\t")
	for i in range(0, len(checkin_list)):
		userid, poiid, timestamp = checkin_list[i]
		if userid == olduserid:
			if timestamp - oldtime <= 6 * 3600:
				fw.write(str(poiid)+"\t")
			else:
				fw.write("\n")
				fw.write(str(userid)+"\t")
				fw.write(str(poiid)+"\t")
				oldtime = timestamp
		else:
			olduserid = userid
			fw.write(str(olduserid)+"\t")
			fw.write(str(poiid)+"\t")
			oldtime = timestamp

	fw.close()

# splitDataset()


def appendPOI():
	fw = open("./"+dataset_name+"_"+train_or_test+"_20180124.txt", 'w')
	with open("./"+dataset_name+"_"+train_or_test+".txt", 'r') as fr:
		lines = fr.readlines()
		for i in range(0, len(lines)):
			temp = lines[i].strip().split('\t')
			str_list = list(lines[i])

			if i != len(lines)-1:
				temp_nx = lines[i+1].strip().split('\t')
				previous_poi = temp_nx[1]
				# print previous_poi
				str_list.insert(-1,str(previous_poi))
				str_result = "".join(str_list)
				print str_result
				fw.write(str_result)
	fw.close()

appendPOI()