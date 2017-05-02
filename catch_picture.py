# coding:utf-8
import re
import string
import sys
import os
import urllib

url = "https://zh.airbnb.com/rooms/13078148"  # 这个是某贴吧地址
imgcontent = urllib.urlopen(url).read()  # 抓取网页内容
urllist = re.findall(r'src="(http.+?\.jpg)"', imgcontent, re.I)  # 提取图片链接
if not urllist:
    print 'not found...'
else:
    # 下载图片,保存在当前目录的pythonimg文件夹下
    filepath = os.getcwd() + '\picture'
    if os.path.exists(filepath) is False:
        os.mkdir(filepath)
    x = 1
    print '爬虫准备就绪...'
    for imgurl in urllist:
        temp = filepath + '\%s.jpg' % x
        print '正在下载第%s张图片' % x
        print imgurl
        urllib.urlretrieve(imgurl, temp)
        x += 1
    print '图片下载完毕，保存路径为' + filepath