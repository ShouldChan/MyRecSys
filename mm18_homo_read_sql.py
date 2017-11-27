# coding:utf-8
import MySQLdb

# first  u should use source *.file commmand to input in mysql database
conn=MySQLdb.Connection('localhost','root','','test')

cursor=conn.cursor()

sql='select * from youtubeuserinfo_tmm'
cursor.execute(sql)

# fetchall接收所有的返回结果行
data=cursor.fetchall()

data=list(data)

import pandas as pd 
mydata = pd.DataFrame(data,columns=['GooglePlusId','username','age', 'gender', \
    'company','aboutme','hobbies','Hometown','Location','Movies','subscriberCnt','viewcnt','TotalUploadViews'])

# print(mydata)
# mydata.to_csv('./googleplususerinfo.csv')
mydata.to_csv('./youtubeuserinfo_withheader.csv',index=False,header=True)
mydata.to_csv('./youtubeuserinfo_noindex_noheader.csv',index=False,header=False)

# with open('./googleplususerinfo_tmm.txt','w') as fw:
#     fw.write(str(data))