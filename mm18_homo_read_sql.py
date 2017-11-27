import MySQLdb

conn=MySQLdb.Connection('localhost','root','','homo')

cursor=conn.cursor()

sql='select * from googleplususerinfo_tmm'
cursor.execute(sql)

data=cursor.fetchall()

data=list(data)

import pandas as pd 
mydata = pd.DataFrame(data,columns=['userid','userName','gender', 'tagLine', \
    'aboutMe','Organization','birthday','placesLived'])

print(mydata)     

# with open('./googleplususerinfo_tmm.txt','w') as fw:
#     fw.write(str(data))