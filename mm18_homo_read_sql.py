# coding:utf-8
import MySQLdb

# first  u should use source *.file commmand to input in mysql database
conn=MySQLdb.Connection('localhost','haomx','haomx','test')

cursor=conn.cursor()

sql='select * from flickruserprofile'
cursor.execute(sql)

# fetchall接收所有的返回结果行
data=cursor.fetchall()

data=list(data)

import pandas as pd 
mydata = pd.DataFrame(data,columns=['googlePlusUserId','flickrUserId', 'location','firstDate', 'grouplist', \
    'groupnum','favoritelist','favoritenum','alltaglist','tagnum','photolist'.'photonum'])

# print(mydata)
mydata.to_csv('./flickruserprofile_withheader.csv',index=False,header=True)
mydata.to_csv('./flickruserprofile_noindex_noheader.csv',index=False,header=False)
