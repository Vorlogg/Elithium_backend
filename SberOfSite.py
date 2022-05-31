import requests
from bs4 import BeautifulSoup
import re
import json
from tqdm import tqdm
import pymorphy2
from datetime import datetime


class SberOfSite():
    def __init__(self):

        self.months = {"январь": "01", "февраль": "02", "март": "03", "апрель": "04", "май": "05", "июнь": "06",
                       "июль": "07",
                       "август": "08", "сентябрь": "09", "октябрь": "10", "ноябрь": "11", "декабрь": "12"}
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
            "Accept-Encoding": "*",
            "Connection": "keep-alive"
        }
        self.morph = pymorphy2.MorphAnalyzer()

        # datfrom = "2021-11-10 00:00:00"

    def date_format_news(self,datespl):
        moth = self.morph.parse(datespl[1])[0].normal_form
        year = datespl[2]
        if len(datespl[0]) != 2:
            day = "0" + datespl[0]
        else:
            day = datespl[0]
        dateend = "{0}-{1}-{2} {3}:{4}:{5}".format(year, self.months[moth], day, "03", "00", "00")
        return dateend



    def parse(self,date, max=100):
        jsonDate = []
        NewsUrls = []
        datefrom = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        flag = False
        for j in tqdm(range(max)):
            url = 'https://www.sberbank.ru/ru/press_center/all?page=' + str(j)
            r = requests.get(url, timeout=10, headers=self.headers)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                for tag in soup.find_all("a", class_="na-article"):
                    # print("url: https://www.sberbank.ru{0} , text:{1}, date:{2}".format(tag.get('href'), tag.text, tag.span))
                    NewsUrls.append("https://www.sberbank.ru" + tag.get('href'))
                for dat in soup.find_all("span", class_="na-article__date"):
                    temp: list = dat.text.split()
                    date_news = self.date_format_news(temp)
                    datenews = datetime.strptime(date_news, "%Y-%m-%d %H:%M:%S")
                    if datenews < datefrom:
                        flag = True
                        break
            if flag:
                break

        # print(len(NewsUrls))
        for urls in tqdm(NewsUrls):
            textnews = ""
            try:
                news = requests.get(urls, timeout=10, headers=self.headers)
                if news.status_code == 200:
                    soupnews = BeautifulSoup(news.text, 'html.parser')
                    title = soupnews.find("h1", class_="np-title").text
                    date = soupnews.find("div", class_="np-date").text
                    datespl = date.split()
                    dateend = self.date_format_news(datespl)
                    datenews = datetime.strptime(dateend, "%Y-%m-%d %H:%M:%S")
                    if datenews < datefrom:
                        break

                    for text in soupnews.find_all("p"):
                        textnews += text.text
                    textnews += soupnews.find("div", class_="np-body").text
                    textnews = re.sub(
                        "media@sberbank.ruРоссия, Москва, 117997, ул. Вавилова, 19© 1997—2021 ПАО Сбербанк.|" + title, '',
                        textnews)
                    # print(textnews)
                    jsonDate.append(
                        {"datetime": dateend, "title": title, "text": textnews, "source": "Сбербанк офф сайт",
                         "company_id": 4, "url": urls
                         })
            except:
                print("error")

        return jsonDate
        # with open('Sber_off_site.json', 'w', encoding='utf-8') as file:
        #     json.dump(jsonDate, file, ensure_ascii=False, indent=2)



