# Step 1
# split train and test

f_train = open('./train.txt','wb')
f_test = open('./test.txt','wb')

with open('./rating.txt','rb') as fread:
	lines = fread.readlines()
	n_test = int(len(lines) * 0.45)
	import random
	resultList = random.sample(range(0,len(lines)),n_test)
	print resultList
	count = 0
	for i in range(len(lines)):
		if i in resultList:
			f_test.write(lines[i])
			count+=1
			print count
		else:
			f_train.write(lines[i])
		# temp = lines[i].strip().split('\t')
		# uid,mid,rate,timestamp = temp[0],temp[1],temp[2],temp[3]

f_train.close()
f_test.close()