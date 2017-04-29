# coding:utf-8
import csv

# 1. extract the column we need
# data_dir = './houses/'
# fwrite = open(data_dir + 'roominfo.csv', 'wb')
# with open(data_dir + 'listings.csv', 'rb') as csvfile:
#     reader = csv.DictReader(csvfile)
#     column_1 = [row['id'] for row in reader]
# print type(column_1)
# fwrite.close()

# 2. count columns of listing.csv and give them a number
# data_dir='./houses/'
# fwrite =open(data_dir+'listing_title_No.txt','wb')
# with open(data_dir+'listing_title.txt','rb') as txtfile:
#     line=txtfile.readline()
#     for i in range(1,95):
#         stri=str(i)+'\n'
#         # print stri
#         line = line.replace(',',stri,1)
#         # print line
#     print line
#     fwrite.write(str(line))
# fwrite.close()

# 3. format .csv into .txt we need, select the properties we need
import csv

house_dir = './houses/'
txtwrite = open(house_dir + 'houseinfo.txt', 'wb')
with open(house_dir + 'listings.csv', 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id, h_id, h_lat, h_lon, h_price, h_value = row['id'], row['host_id'], row['latitude'], row['longitude'], row[
            'price'], row['review_scores_value']
        h_price = h_price.replace('$', '', 1)
        # print id, h_id, h_lat, h_lon, h_price, h_value
        if h_value != '':
            txtwrite.write(id + '\t' + h_id + '\t' + h_lat + '\t' + h_lon + '\t' + h_price + '\t' + h_value + '\n')
txtwrite.close()
