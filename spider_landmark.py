# coding:utf-8
from bs4 import BeautifulSoup
import requests
import time

# 第一页网址
url = 'https://cn.tripadvisor.com/Attractions-g60763-Activities-New_York_City_New_York.html'
# 登录后保存清单所在页面的网址
url_saves = 'https://cn.tripadvisor.com/Saves#48450273'
# 找到33个页面的网址的规律，然后用解析式的方式来存储到变量中
urls = [
    'https://cn.tripadvisor.com/Attractions-g60763-Activities-oa{}-New_York_City_New_York.html#ATTRACTION_LIST'.format(
        str(i)) for i in range(30, 990, 30)]
# Request Headers中的信息
headers = {
    'User-Agent': '',
    'Cookie': ''
}


# 获取非登录网址中标题，图片，tags的信息
def get_attractions(url, data=None):
    wb_data = requests.get(url, headers=headers)
    time.sleep(2)
    soup = BeautifulSoup(wb_data.text, 'lxml')
    titles = soup.select('div.property_title > a[target="_blank"]')
    imgs = soup.select('img[width="160"]')
    cates = soup.select('div.p13n_reasoning_v2')

    if data == None:
        for title, img, cate in zip(titles, imgs, cates):
            data = {
                'title': title.get_text(),
                'img': img.get('src'),
                'meta': list(cate.stripped_strings)
            }
            print(data)
        print '123'

# 获取登录后保存清单中的信息
def get_favs(url, data=None):
    wb_data = requests.get(url, headers=headers)
    soup = BeautifulSoup(wb_data.text, 'lxml')
    titles = soup.select('a.location-name')
    imgs = soup.select('div.photo > div.sizedThumb > img.photo_image')
    metas = soup.select('span.format_address')
    if data == None:
        for title, img, meta in zip(titles, imgs, metas):
            data = {
                'title': title.get_text(),
                'img': img.get('src'),
                'meta': list(meta.stripped_strings)
            }
            print(data)


# 遍历32个网址，对每一个网址调用get_attractions函数
for single_url in urls:
    get_attractions(single_url)
