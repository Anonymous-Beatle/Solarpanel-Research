
#Scipt to scrape data from the Huawei solarpanel website / chart
#Robrecht Keijzer
#26-6-2023

#Standard Library
import time
import os

#Third party modules
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd


def open_data_screen():
    """Makes a driver, opened at the correct page to scrape the data."""
    driver = webdriver.Chrome()
    driver.get("https://eu5.fusionsolar.huawei.com/")
    assert "Login" in driver.title
    delay_factor = 1     #make higher if needed
    
    #change to iframe
    try:
        my_elem = WebDriverWait(driver, delay_factor*3).until(EC.frame_to_be_available_and_switch_to_it((By.XPATH, "//div[@id='loginAdv']/iframe[1]")))      #relative location, after id=loginAdv
    except TimeoutException:
        print("Loading took too much time! (iframe)")

    #click on login button
    try:
        my_elem = WebDriverWait(driver, delay_factor*5).until(EC.visibility_of_element_located((By.ID, "experiencePlant")))     #use 'visibility' when working inside an iframe.
    except TimeoutException:
        print("Loading took too much time! (demo_site)")
    demo_site = driver.find_element(By.ID, "experiencePlant")
    demo_site.click()

    #click on link to data page.
    try:
        my_elem = WebDriverWait(driver, delay_factor*10).until(EC.presence_of_element_located((By.XPATH, "//a[@class='nco-home-list-table-name nco-home-list-text-ellipsis']"))) 
    except TimeoutException:
        print("Loading took too much time! (data_page)")
    data_page = driver.find_element(By.XPATH, "//a[@class='nco-home-list-table-name nco-home-list-text-ellipsis']")
    data_page.click()
    
    try:
        my_elem = WebDriverWait(driver, delay_factor*3).until(EC.visibility_of_element_located((By.XPATH, "//button[@title='Month']"))) 
    except TimeoutException:
        print("Loading took too much time! (data_page)")
    month_button = driver.find_element(By.XPATH, "//button[@title='Month']")
    driver.execute_script("arguments[0].scrollIntoView();", month_button)      #Scroll
    
    month_button = driver.find_element(By.XPATH, "//button[@title='Month']")
    month_button.click()        #now click
    return driver




driver = open_data_screen()


#load the chart-cursor data by moving to the chart
try:
    my_elem = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.XPATH, "//div[@class='echarts-for-react ']"))) 
except TimeoutException:
    print("Loading took too much time! (data_page)")
data = driver.find_element(By.XPATH, "//div[@class='echarts-for-react ']")

time.sleep(1)       #moving the cursor seems to be the least reliable step in the script. I added a delay, and repeated this step 3 times.
for i in range(3):
    actions = ActionChains(driver)
    actions.move_to_element(data).perform()


actions = ActionChains(driver)
position = actions.move_by_offset(0, 0)

#this is where the yield data is found
try:
    my_elem = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, "//div[@class='echarts-for-react ']/div[2]/span[2]"))) 
except TimeoutException:
    print("Loading took too much time! (data_page)")



#SETTINGS:
#---------------------------------------------------------------------------------------------------------------------------------
chart_step = 1      #with each step, go 20 pixels to the right
shift_0 = -294        #cursor starts roughly in the middle. This is the required shift to move it to the left
no_of_steps = 500     #number of steps.
#---------------------------------------------------------------------------------------------------------------------------------


target_position = position.move_by_offset(shift_0, 0)
target_position.perform()
#time.sleep(0.1)

my_yield = []
my_month = []

#range(-range_factor, +range_factor+1, int(range_factor/5))
for x_shift in [chart_step]*no_of_steps:
    target_position = position.move_by_offset(x_shift, 0)
    target_position.perform()
    
    #read data from chart at place of the cursor
    my_data = driver.find_element(By.XPATH, "//div[@class='echarts-for-react ']/div[2]/span[2]")
    yield_raw = my_data.get_attribute('outerHTML')
    my_data2 = driver.find_element(By.XPATH, "//div[@class='echarts-for-react ']/div[2]").get_attribute('outerHTML')
    
    my_yield.append(yield_raw.split('>')[1][:-6])
    my_month.append(my_data2.split("pointer-events:")[1].split("<")[0][-2:])
    #time.sleep(0.1)


print(my_yield)
print(my_month)

driver.close()




a = input("Do you want to save the data? (type Yes, or anything else): ")
if a == "Yes":
    df = pd.DataFrame({"Month": my_month, "Yield": my_yield})
    file_path = os.path.join((os.path.dirname(__file__)),"demo_chart_raw.txt")
    df.to_csv(file_path, sep="\t", index=False)