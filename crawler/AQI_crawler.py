from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd

import os
import datetime
import time

PROJECT_FOLDER = os.path.dirname(os.path.dirname(__file__))
AQI_PATH = os.path.join(PROJECT_FOLDER, 'data', 'AQI.csv')
MISSING_PATH = os.path.join(PROJECT_FOLDER, 'data', 'missing.csv')

def open_calender(driver: webdriver):

    calendar_element = driver.find_element(By.CSS_SELECTOR, 'img.oneday[fd="oneday"]')
    calendar_element.click()
    driver.implicitly_wait(1)
    return 

def choose_year(driver: webdriver, year: int):

    # 2016 <= year <= 2024(this year)
    if year > datetime.date.today().year or year <= 2016:
        raise ValueError(f"Target year({year}) is out of range.")

    element = driver.find_element(By.CSS_SELECTOR, "a[aria-live='assertive']")
    element.click()
    time.sleep(1)
    current_year = int(element.text)

    if current_year == year:    return
    
    prev_button = driver.find_element(By.XPATH, "//a[@role='button' and @class='k-link k-nav-prev']")
    next_button = driver.find_element(By.XPATH, "//a[@role='button' and @class='k-link k-nav-next']")

    pressed_button, step = None, None
    if current_year < year: pressed_button, step = next_button, range(current_year, year)
    else:                   pressed_button, step = prev_button, range(year, current_year)

    for _ in step:
        pressed_button.click()
        time.sleep(1)
    
    return

def choose_month(driver: webdriver, month: int):
    
    if month < 1 or month > 12:
        raise ValueError(f"Target month({month}) is out of range.")
    
    digits = [None, '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一', '十二']
    month_buttons = [None]
    for i in range(1, 13):
        button = driver.find_element(By.XPATH, f"//a[text()='{digits[i]}月']")
        month_buttons.append(button)

    month_buttons[month].click()
    time.sleep(1)

    return

def choose_day(driver: webdriver, date: datetime.datetime):

    day_button = driver.find_element(By.XPATH, f"//a[@data-value='{date.year}/{date.month - 1}/{date.day}']")
    print(f"Text = {day_button.text}, Title = {day_button.get_attribute('title')}")
    day_button.click()
    time.sleep(0.5)
    return

def crawler(driver: webdriver):

    if os.path.exists(AQI_PATH):
            result_df = pd.read_csv(AQI_PATH)
    else:   result_df = pd.DataFrame(columns = 'Date,Location,AQI'.split(','))

    if os.path.exists(MISSING_PATH):
            missing_df = pd.read_csv(MISSING_PATH)
    else:   missing_df = pd.DataFrame(columns = ['Date'])
    # start_date = datetime.date(year = 2017, month = 1, day = 1)
    # end_date   = datetime.date(year = 2024, month = 9, day = 30)
    start_date = datetime.date(year = 2024, month = 1, day = 2)
    end_date   = datetime.date(year = 2024, month = 1, day = 1)
    date = start_date
    result_idx = len(result_df)
    missing_idx = len(missing_df)

    time.sleep(2)
    while True:
        print(str(date))
        open_calender(driver)
        time.sleep(0.5)
        choose_year(driver, date.year)
        time.sleep(0.5)
        choose_month(driver, date.month)
        time.sleep(0.5)
        choose_day(driver, date)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        table = soup.find('table', class_ = 'tableAQI')
        table_date_strs = soup.find('td', colspan = '12').find('dd').text.split('-')
        table_date = datetime.date(year = int(table_date_strs[0]), month = int(table_date_strs[1]), day = int(table_date_strs[2]))

        if table_date == date:
            tags = table.find('tbody').find_all('td')
            for i in range(0, len(tags), 2):
                result_df.loc[result_idx, 'Date'] = str(date)
                result_df.loc[result_idx, 'Location']  = str(tags[i].text)
                result_df.loc[result_idx, 'AQI']       = str(tags[i + 1].text)
                result_idx += 1
        else:
            missing_df.loc[missing_idx, 'Date'] = str(date)
            missing_idx += 1
            missing_df.to_csv(MISSING_PATH, index = False)

        result_df.to_csv(AQI_PATH, index = False)
        time.sleep(0.8)

        if date == end_date:    break
        date += datetime.timedelta(days = -1)

    return

if __name__ == '__main__':

    url = 'https://statis.moenv.gov.tw/epanet/'
    driver = webdriver.Chrome()
    driver.get(url)
    crawler(driver)
    driver.quit()