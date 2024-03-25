
#Scipt to scrape data from the Huawei solarpanel website / chart
#Robrecht Keijzer
#27-6-2023

#Standard Library
import json
import os
from datetime import datetime
import time

#Third party modules
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd



file_path = os.path.join((os.path.dirname(__file__)),"login_info.json")
with open(file_path) as json_file:
    data = json.load(json_file)
    username = data['username']
    password = data['password']
    
#-----CHANGE DATE HERE-----
date = datetime(2023,6,13)
date_str = date.strftime("%Y-%m-%d")


def open_data_screen(date_str: str, username, password):
    """Makes a driver, opened at the correct page to scrape the data, and at the correct date."""
    driver = webdriver.Chrome()
    driver.get("https://eu5.fusionsolar.huawei.com/")
    assert "Login" in driver.title
    delay_factor = 1     #make higher if needed
    
    #"//input[@name='ssoCredentials.username']"
    
    #username text field
    try:
        my_elem = WebDriverWait(driver, delay_factor*3).until(EC.element_to_be_clickable((By.ID, "username")))      #relative location, after id=loginAdv
    except TimeoutException:
        print("Loading took too much time! (username)")

    username_field = driver.find_element(By.ID, "username")
    username_field.send_keys(username)
    
    password_field = driver.find_element(By.ID, "value")
    password_field.send_keys(password)
    password_field.send_keys(Keys.RETURN)

    try:
        my_elem = WebDriverWait(driver, delay_factor*10).until(EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='Select date']"))) 
    except TimeoutException:
        print("Loading took too much time! (data_page)")
    calendar = driver.find_element(By.XPATH, "//input[@placeholder='Select date']")
    driver.execute_script("arguments[0].scrollIntoView();", calendar)      #Scroll
    
    calendar.send_keys(Keys.CONTROL, "a")       #change to correct day
    calendar.send_keys(date_str)
    calendar.send_keys(Keys.RETURN)
    
    return driver





driver = open_data_screen(date_str, username, password)


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
chart_step = 10      #with each step, go 20 pixels to the right
shift_0 = -250        #cursor starts roughly in the middle. This is the required shift to move it to the left
no_of_steps = 48     #number of steps.
#---------------------------------------------------------------------------------------------------------------------------------


target_position = position.move_by_offset(shift_0, 0)
target_position.perform()
#time.sleep(0.1)


output = []
date_time = []

#range(-range_factor, +range_factor+1, int(range_factor/5))
for x_shift in [chart_step]*no_of_steps:
    target_position = position.move_by_offset(x_shift, 0)
    target_position.perform()
    
    #read data from chart at place of the cursor
    my_data = driver.find_element(By.XPATH, "//div[@class='echarts-for-react ']/div[2]/span[2]")
    output_raw = my_data.get_attribute('outerHTML')  
    my_data2 = driver.find_element(By.XPATH, "//div[@class='echarts-for-react ']/div[2]").get_attribute('outerHTML')
    
    output.append(output_raw.split(">")[1].split("<")[0])
    date_time.append(my_data2.split("pointer-events:")[1].split("<")[0].split(">")[1])
    #time.sleep(0.1)


print(output)
print(date_time)


driver.close()



a = input("Do you want to save the data? (type Yes, or anything else): ")
if a == "Yes":
    df = pd.DataFrame({"Time (Y-M-D)": date_time, "Solar Power (kW)": output})
    file_path = os.path.join((os.path.dirname(__file__)),f"solar_data_{date_str}_raw.txt")
    df.to_csv(file_path, sep="\t", index=False)