
#This script calculates a three year average (with error) of the loss due to cloudiness.
#Robrecht Keijzer
#30-06-2023



import os
import pandas as pd
from datetime import datetime, timedelta

#Goeie site voor de data: https://nsrdb.nrel.gov/data-viewer
#Ook DNI is interesting, dit is dus rekening houdend met bewolking etc.

#DHI = Diffuse Horizontal Irradiance        #This is the light that falls on the panel indirectly, through scattering etc
#DNI = Direct Normal irradiance             #Amount of radiation recieved for a surface perpendicular to the sun-rays. -> This is what we need
#GHI = Global Horizontal Irradiance         #Sum of the two, for a surface horizontal to the ground (so not interely what we need for our solar panels)


year_monthly_average = []

for year in [2017,2018,2019]:
    file_path = os.path.join("NSRDB data", f"464295_51.13_4.70_{year}.csv")

    df = pd.read_csv(file_path, skiprows=2)
    names = df.columns
    time_strings = [f"{year}-{month}-{day}-{hour}-{minute}" for year,month,day,hour,minute \
        in zip(list(df[names[0]]),list(df[names[1]]),list(df[names[2]]),list(df[names[3]]),list(df[names[4]]))]


    #---------------------
    day_one = datetime(year,1,1,0,0)
    fraction_sum = [0]*12
    fraction_number = [0]*12

    for day in range(365):
        day_date = day_one + timedelta(days=day)
        end_date = day_date + timedelta(days=1)-timedelta(minutes=15)

        start_index = time_strings.index(day_date.strftime('%Y-%m-%d-%H-%M').replace("-0", "-"))
        end_index = time_strings.index(end_date.strftime('%Y-%m-%d-%H-%M').replace("-0", "-"))
        one_day = time_strings[start_index:end_index]

        #convert to datetime
        one_day = [datetime.strptime(time, '%Y-%m-%d-%H-%M') for time in one_day]
        irradiance = list(df["Clearsky DNI"])[start_index:end_index]
        real_irr = list(df["DNI"])[start_index:end_index]

        indirect = list(df["Clearsky DHI"])[start_index:end_index]
        indirect_real = list(df["DHI"])[start_index:end_index]

        clearsky_power = sum(irradiance) + sum(indirect)
        cloudy_power = sum(real_irr) + sum(indirect_real)
        fraction = cloudy_power / clearsky_power

        fraction_sum[day_date.month-1] += fraction
        fraction_number[day_date.month-1] += 1



    remaining_power = [my_sum/number for my_sum, number in zip(fraction_sum, fraction_number)]
    
    year_monthly_average.append(remaining_power)

import numpy as np
np.save("year_monthly_av.npy", year_monthly_average)