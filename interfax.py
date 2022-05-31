import requests
from bs4 import BeautifulSoup
import re
import json
from tqdm import tqdm
import pymorphy2
from datetime import datetime


class Interfax():
    def __init__(self):

        self.months = {"январь": "01", "февраль": "02", "март": "03", "апрель": "04", "май": "05", "июнь": "06", "июль": "07",
                  "август": "08", "сентябрь": "09", "октябрь": "10", "ноябрь": "11", "декабрь": "12"}

        self.morph = pymorphy2.MorphAnalyzer()

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
            "Accept-Encoding": "*",
            "Connection": "keep-alive"
        }
        self.datef = "2021-10-28 11:00:00"

    def date_format_text_news(self,datespl):
        year = datespl[3]
        if len(datespl[1]) != 2:
            day = "0" + datespl[1]
        else:
            day = datespl[1]
        time = re.sub(",", "", datespl[0])
        time = time.split(':')
        moth = self.morph.parse(datespl[2])[0].normal_form
        dateend = "{0}-{1}-{2} {3}:{4}:{5}".format(year, self.months[moth], day, time[0], time[1], "00")
        return dateend

    def date_format_news(self,datespl):
        year = datespl[2]
        if len(datespl[0]) != 2:
            day = "0" + datespl[0]
        else:
            day = datespl[0]
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
        company_id_list = {1: ["Аэрофлот", "%C0%FD%F0%EE%F4%EB%EE%F2"], 2: ["Газпром", "%C3%E0%E7%EF%F0%EE%EC"],
                           3: ["ВТБ", "%C2%D2%C1"], 4: ["Сбер", "%D1%E1%E5%F0%E1%E0%ED%EA"],
                           5: ["Лукойл", "%CB%D3%CA%CE%C9%CB"], 6: ["Aplle", "Apple"]}
        flag = False
        for j in tqdm(range(max)):
            url = 'https://www.interfax.ru/tags/{0}/page_{1}'.format(company_id_list[company_id][1], str(j))
            r = requests.get(url, timeout=10, headers=self.headers)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser').find('div', class_='sPageResult')
                for tag in soup.find_all("h3"):
                    NewsUrls.append("https://www.interfax.ru" + tag.a.get('href'))
                for dat in soup.find_all("time"):
                    temp: list = dat.text.split()
                    if len(temp) > 3:
                        temp.pop(3)
                    else:
                        temp.insert(2, "2021")
                    date_news = self.date_format_news(temp)
                    datenews = datetime.strptime(date_news, "%Y-%m-%d %H:%M:%S")
                    if datenews < datefrom:
                        flag = True
                        break
            if flag:
                break

        # print(NewsUrls)
        for urls in tqdm(NewsUrls):
            try:
                textnews = ""
                news = requests.get(urls, timeout=10, headers=self.headers)
                news.encoding = 'cp1251'
                if news.status_code == 200:
                    soupnews = BeautifulSoup(news.text, 'html.parser').find('div', class_="infinitblock")
                    title = soupnews.find("h1").text
                    date = soupnews.find("a", class_="time").text
                    datespl = date.split()
                    dateend = self.date_format_text_news(datespl)
                    datenews = datetime.strptime(dateend, "%Y-%m-%d %H:%M:%S")
                    if datenews < datefrom:
                        break
                    for text in soupnews.find_all("p"):
                        textnews += text.text

                    textnews = re.sub("^\n|\r", '', textnews)
                    textnews = re.sub(
                        "INTERFAX.RU - ", '',
                        textnews)
                    # print(textnews)
                    jsonDate.append(
                        {"datetime": dateend, "title": title, "text": textnews,
                         "source": "{0} interfax".format(company_id_list[company_id][0]),
                         "company_id": company_id, "url": urls
                         })

            except:
                print("Ошибка парсинга страницы")
        return jsonDate
        # with open('{0}_interfax.json'.format(company_id_list[company_id][0]), 'w', encoding='utf-8') as file:
        #     json.dump(jsonDate, file, ensure_ascii=False, indent=2)



