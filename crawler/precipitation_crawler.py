import csv
import os
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-usb")
driver = webdriver.Chrome(options=options)


def main():
    url = "https://www.cwa.gov.tw/V8/C/D/DailyPrecipitation.html"

    StationID = "臺東"

    project_root = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(project_root, "data", "raw", "rainfall", "Taitung_Precipitation.csv")

    try:
        driver.get(url)

        with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Serial", "Precipitation(mm)"])

        for year in range(2017, 2025):
            print(f"Processing: {year}")

            try:
                year_select = Select(driver.find_element(By.NAME, "Year"))
                year_select.select_by_visible_text(str(year))

                station_select = Select(driver.find_element(By.NAME, "StationID"))
                station_select.select_by_visible_text(StationID)

                driver.execute_script("document.querySelector('[name=\"Year\"]').value = arguments[0];", str(year))
                driver.execute_script("document.querySelector('[name=\"Year\"]').dispatchEvent(new Event('change'))")
                driver.execute_script(
                    "document.querySelector('[name=\"StationID\"]').dispatchEvent(new Event('change'))"
                )

                WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "DayRain_MOD")))
                WebDriverWait(driver, 10).until(
                    lambda d: str(year) in d.find_element(By.ID, "DayRain_MOD").get_attribute("outerHTML")
                )

                html = driver.page_source

                soup = BeautifulSoup(html, "html.parser")
                tbody = soup.find("tbody", {"id": "DayRain_MOD"})

                result = []
                rows = tbody.find_all("tr")  # 找到所有行
                for tr in rows:
                    tds = tr.find_all("td")  # 找到所有 <td>
                    row_data = [td.text.strip() for td in tds]
                    if row_data:
                        result.append(row_data)

                with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)

                    for month in range(1, 13):
                        for day in range(1, len(result)):
                            serial = f"{year}{month:02}{day:02}"

                            try:
                                precipitation = result[day - 1][month - 1]

                                if precipitation == "":
                                    continue

                                if not precipitation.replace(".", "", 1).isdigit():
                                    precipitation = 0
                                else:
                                    precipitation = float(precipitation)

                                writer.writerow([serial, precipitation])
                            except IndexError:
                                continue

            except Exception as e:
                print(f"An error occurred while processing year {year}: {e}")
                continue

        print(f"Data saved to {csv_path}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
