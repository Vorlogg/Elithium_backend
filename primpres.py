import requests
from bs4 import BeautifulSoup
import re
import json
from tqdm import tqdm
import pymorphy2
from datetime import datetime


class Primpress():
    def __init__(self):
        self.months = {"январь": "01", "февраль": "02", "март": "03", "апрель": "04", "май": "05", "июнь": "06",
                       "июль": "07",
                       "август": "08", "сентябрь": "09", "октябрь": "10", "ноябрь": "11", "декабрь": "12"}
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
            "Accept-Encoding": "*",
            "Connection": "keep-alive"}
        self.morph = pymorphy2.MorphAnalyzer()

    def date_format_text_news(self, datespl):
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

    def date_format_news(self, datespl):
        moth = datespl[1]
        day = datespl[0].lstrip()
        year_time = re.sub(",", "", datespl[2]).split()
        year = year_time[0]
        time = year_time[1].split(':')
        dateend = "{0}-{1}-{2} {3}:{4}:{5}".format(year, moth, day, time[0], time[1], "00")
        return dateend

    def parse(self, id, date="1990-01-01 00:00:00", max=100):
        datefrom = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        jsonDate = []
        NewsUrls = []
        company_id = id
        company_id_list = {1: ["Аэрофлот", "%D0%90%D1%8D%D1%80%D0%BE%D1%84%D0%BB%D0%BE%D1%82"],
                           2: ["Газпром", "%D0%93%D0%B0%D0%B7%D0%BF%D1%80%D0%BE%D0%BC"],
                           3: ["ВТБ", "%D0%92%D0%A2%D0%91"], 4: ["Сбер", "%D1%81%D0%B1%D0%B5%D1%80"],
                           5: ["Лукойл", "%D0%9B%D1%83%D0%BA%D0%BE%D0%B9%D0%BB"], 6: ["Aplle", "apple"]}
        flag = False

        for j in tqdm(range(1, max)):
            url = "https://primpress.ru/search?find={0}&page={1}".format(company_id_list[company_id][1], str(j))
            r = requests.get(url, timeout=10, headers=self.headers)
            if r.status_code == 200:
                # print(url)
                soup = BeautifulSoup(r.text, 'html.parser').find("div", id="sticky-content")
                # print(soup.find_all("a"))
                for tag in soup.find_all("a"):
                    # print(tag.get("href"))
                    strurls = tag.get("href")
                    if re.search("article", strurls):
                        NewsUrls.append(strurls)
                for dat in soup.find_all("div", class_="text-muted"):
                    temp: list = dat.text.split(".")
                    date_news = self.date_format_news(temp)
                    datenews = datetime.strptime(date_news, "%Y-%m-%d %H:%M:%S")
                    if datenews < datefrom:
                        flag = True
                        break
            if flag:
                break

        for urls in tqdm(NewsUrls):
            try:
                textnews = ""
                news = requests.get(urls, timeout=10, headers=self.headers)
                if news.status_code == 200:
                    soupnews = BeautifulSoup(news.text, 'html.parser').find("div", id="sticky-content")
                    title = soupnews.find("h1", class_="b-h-title b-h-title--home").text
                    date = soupnews.find(title="Дата публикации").text
                    datespl = date.split()
                    dateend = self.date_format_text_news(datespl)
                    datenews = datetime.strptime(dateend, "%Y-%m-%d %H:%M:%S")
                    if datenews < datefrom:
                        break

                    textnews = soupnews.find("div", class_="b-article-content clearfix mb-4").text
                    textnews = re.sub(
                        r'Читайте также:[\s\S]*', '',
                        textnews)
                    jsonDate.append(
                        {"datetime": dateend, "title": title, "text": textnews,
                         "source": "{0} smart-lab".format(company_id_list[company_id][0]),
                         "company_id": company_id, "url": urls
                         })

            except:
                print("Ошибка парсинга страницы")
        return jsonDate
    # print(len(jsonDate))
    # with open('{0}_primpres.json'.format(company_id_list[company_id][0]), 'w', encoding='utf-8') as file:
    #     json.dump(jsonDate, file, ensure_ascii=False, indent=2)
