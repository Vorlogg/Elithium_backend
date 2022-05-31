import requests
from bs4 import BeautifulSoup
import re
import json
from tqdm import tqdm
import pymorphy2
from datetime import datetime

class SmartLab():
    def __init__(self):
        self.months = {"январь": "01", "февраль": "02", "март": "03", "апрель": "04", "май": "05", "июнь": "06", "июль": "07",
                  "август": "08", "сентябрь": "09", "октябрь": "10", "ноябрь": "11", "декабрь": "12"}

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
            "Accept-Encoding": "*",
            "Connection": "keep-alive"
        }
        self.morph = pymorphy2.MorphAnalyzer()

    def date_format_text_news(self,datespl):
        year = datespl[2]
        if len(datespl[0]) != 2:
            day = "0" + datespl[0]
        else:
            day = datespl[0]
        year = re.sub(",", "", year)
        time = re.sub(",", "", datespl[3])
        time = time.split(':')
        moth = self.morph.parse(datespl[1])[0].normal_form
        dateend = "{0}-{1}-{2} {3}:{4}:{5}".format(year, self.months[moth], day, time[0], time[1], "00")
        return dateend


    def parse(self,id, date, max=100):
        datefrom = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        jsonDate = []
        NewsUrls = []
        company_id = id
        company_id_list = {1: ["Аэрофлот", "AFLT"], 2: ["Газпром", "Газпром"], 3: ["ВТБ", "ВТБ"], 4: ["Сбер", "Sber"],
                           5: ["Лукойл", "LKOH"], 6: ["Aplle", "AAPL"]}
        for j in tqdm(range(1, max)):
            # print(j)
            url = "https://smart-lab.ru/forum/news/{0}/page{1}/".format(company_id_list[company_id][1], str(j))
            r = requests.get(url, timeout=10, headers=self.headers)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser').find("div", class_="temp_block")
                for tag in soup.find_all("a"):
                    strurls = re.sub("/blog/", "", tag.get("href"))
                    NewsUrls.append("https://smart-lab.ru/blog/news/" + strurls)

        # print(NewsUrls)

        for urls in tqdm(NewsUrls):
            try:
                textnews = ""
                news = requests.get(urls, timeout=10, headers=self.headers)
                if news.status_code == 200:
                    soupnews = BeautifulSoup(news.text, 'html.parser')
                    title = soupnews.find("h1").span.text
                    date = soupnews.find(class_="date").text
                    datespl = date.split()
                    dateend = self.date_format_text_news(datespl)
                    datenews = datetime.strptime(dateend, "%Y-%m-%d %H:%M:%S")
                    if datenews < datefrom:
                        break

                    textnews = soupnews.find_all("div", class_="content")[1].text
                    jsonDate.append(
                        {"datetime": dateend, "title": title, "text": textnews,
                         "source": "{0} smart-lab".format(company_id_list[company_id][0]),
                         "company_id": company_id, "url": urls
                         })

            except:
                print("Ошибка парсинга страницы")
        return jsonDate

