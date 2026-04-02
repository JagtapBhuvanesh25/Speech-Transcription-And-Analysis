from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

def get_flipkart_reviews(url, pages=3):
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    driver.get(url)
    time.sleep(5)

    reviews_data = []

    for page in range(pages):
        print(f"Scraping page {page+1}...")

        reviews = driver.find_elements(By.XPATH, '//div[@class="_27M-vq"]')

        for review in reviews:
            try:
                name = review.find_element(By.XPATH, './/p[@class="_2sc7ZR _2V5EHH"]').text
            except:
                name = None

            try:
                rating = review.find_element(By.XPATH, './/div[@class="_3LWZlK _1BLPMq"]').text
            except:
                rating = None

            try:
                title = review.find_element(By.XPATH, './/p[@class="_2-N8zT"]').text
            except:
                title = None

            try:
                content = review.find_element(By.XPATH, './/div[@class="t-ZTKy"]/div/div').text
            except:
                content = None

            try:
                verified = review.find_element(By.XPATH, './/p[contains(text(),"Certified Buyer")]').text
            except:
                verified = "No"

            reviews_data.append({
                "Customer Name": name,
                "Rating": rating,
                "Title": title,
                "Review": content,
                "Verified": verified
            })

        # Next page
        try:
            next_button = driver.find_element(By.XPATH, '//a[@class="_1LKTO3"]')
            next_button.click()
            time.sleep(4)
        except:
            break

    driver.quit()
    return pd.DataFrame(reviews_data)


# 👉 Use product page URL (NOT homepage)
url = "https://www.flipkart.com/product/p/itmXXXXXXXX?pid=XXXX&lid=XXXX&marketplace=FLIPKART"

df = get_flipkart_reviews(url, pages=5)

df.to_csv("flipkart_reviews.csv", index=False)
print(df.head())