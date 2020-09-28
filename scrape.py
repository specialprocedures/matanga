import os
import re
from tbselenium.tbdriver import TorBrowserDriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
import time
from random import randint
import pandas as pd
import datetime
from transliterate import translit, get_available_language_codes
import schedule


# Housekeeping

rgn = {"Tbilisi": "1633", "Batumi": "1634", "Kutaisi": "1896", "Samegrelo": "1819"}

# Main scraping function
def scrape(key, reg_code):
    df_list=[]
    url_stem2 = "https://matangaezngvjcud/?iFirstLevel={}&iPage={}"
    it = 0
    items = ['kickoff']

    while len(items) > 0:
        url = url_stem2.format(reg_code, it)
        with TorBrowserDriver("/opt/tor-browser_en-US/", pref_dict={'dom.webdriver.enabled': False, 'network.proxy.socks_remote_dns': True, 'javascript.enabled': False, 'useAutomationExtension': False}) as driver:

            time.sleep(randint(0, 30))
            driver.get(url)

            html = driver.page_source
            soup = BeautifulSoup(html, features='lxml')

            items = soup.find_all('div', {'class' : 'banner_item type2'}) + soup.find_all('div', {'class' : 'banner_item type1'})

            for item in items:
                out = {}

                out['time_stamp'] = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                out['region'] = key
                headings = item.find('span', {'class' : 'banner_title'})
                out['sell_code'] = headings.find_all('a')[0].text
                out['sell_name'] = headings.find_all('a')[1].text
                out['description'] = item.find('div', {'class' : 'banner_content_name'}).text

                locations = item.find('select', {'name': 'iCategoryID'}).find_all('option')
                for loc in locations:
                    out[loc.text.strip()] = True

                for li in item.find_all('li'):
                    sp_li = [i.strip() for i in li.text.split(':')]
                    out[sp_li[0]] = sp_li[1]

                for pr in item.find_all('div', {'class':'price1'}):
                    for p in pr.find_all('p'):
                        sp_pr = [i.strip() for i in p.text.split()]
                        out[sp_pr[1]] = sp_pr[0]

                curr = item.find('div', {'class':'price2'}).text.replace(" ", "")
                out[re.search("([^0-9]+)", curr)[0]] = re.search("([0-9]+)", curr)[0]

                df_list.append(out)
                # print(out)
        it += 1
    return df_list

#
def job(fail=0, sleep=3600):
    if fail <= 10:
        try:
            time.sleep(randint(0, sleep))
            df_list = []

            for key, reg_code in rgn.items():
                df_list.extend(scrape(key, reg_code))

            df = pd.DataFrame(df_list)
            df['desc_en'] = df['description'].apply(lambda x: translit(x, 'ru', reversed=True))
            ts = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
            df.to_csv(f"DATA\INPUT\SCRAPED\output_{ts}.csv")

            message = f"Scrape at {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')} successful with {len(df)} records"
            print(message)
            with open('DATA\INPUT\SCRAPED\logfile.txt', 'a') as f:
                f.write(message + '\n')
            return

        except (TimeoutException, WebDriverException) as e:
            message = f"Scrape at {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')} failed with {e}"
            print(message)
            with open('DATA\INPUT\SCRAPED\logfile.txt', 'a') as f:
                f.write(message + '\n')

            job(fail=fail+1, sleep=1200)
    else:
        message = f"Scrape at {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')} failed with max re-tries"
        print(message + '\n')
        with open('DATA\INPUT\SCRAPED\logfile.txt', 'a') as f:
            f.write(message + '\n')
        pass

job(sleep=1)
schedule.every(4).hours.do(job)
current_job = ""
while True:
    if schedule.jobs != current_job:
        print(schedule.jobs[-1])
        current_job = schedule.jobs
    schedule.run_pending()
    time.sleep(1)
