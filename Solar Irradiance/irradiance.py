import os
import pandas as pd
from datetime import datetime, timedelta

#Goeie site voor de data: https://nsrdb.nrel.gov/data-viewer
#Ook DNI is interesting, dit is dus rekening houdend met bewolking etc.

#DHI = Diffuse Horizontal Irradiance        #This is the light that falls on the panel indirectly, through scattering etc
#DNI = Direct Normal irradiance             #Amount of radiation recieved for a surface perpendicular to the sun-rays. -> This is what we need
#GHI = Global Horizontal Irradiance         #Sum of the two, for a surface horizontal to the ground (so not interely what we need for our solar panels)

file_path = os.path.join("NSRDB data", "464295_51.13_4.70_2019.csv")

df = pd.read_csv(file_path, skiprows=2)
names = df.columns
time_strings = [f"{year}-{month}-{day}-{hour}-{minute}" for year,month,day,hour,minute \
    in zip(list(df[names[0]]),list(df[names[1]]),list(df[names[2]]),list(df[names[3]]),list(df[names[4]]))]


#---------------------
day_date = datetime(2019,7,21,0,0)
end_date = day_date + timedelta(days=1)

start_index = time_strings.index(day_date.strftime('%Y-%m-%d-%H-%M').replace("-0", "-"))
end_index = time_strings.index(end_date.strftime('%Y-%m-%d-%H-%M').replace("-0", "-"))

one_day = time_strings[start_index:end_index]

#convert to datetime
one_day = [datetime.strptime(time, '%Y-%m-%d-%H-%M') for time in one_day]
irradiance = list(df["Clearsky DNI"])[start_index:end_index]
real_irr = list(df["DNI"])[start_index:end_index]

indirect = list(df["Clearsky DHI"])[start_index:end_index]
indirect_real = list(df["DHI"])[start_index:end_index]

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.dates import (HOURLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange, HourLocator)
import pandas as pd
rcParams["font.family"] = "Serif"



fig = plt.figure(figsize=(11,6), layout="constrained")
fig.suptitle(f"Sun irradiance at {day_date.strftime('%Y-%m-%d')}", fontsize=15, fontweight= "bold")
ax1 = fig.add_subplot(1,1,1)


ax1.plot(one_day, irradiance, '-', color='firebrick', label="Clearsky")
ax1.plot(one_day, real_irr, '-', color='cadetblue', label="Measured (with clouds)")
ax1.plot(one_day, indirect, '-', color='green', label="Indirect Light")
ax1.plot(one_day, indirect_real, '-', color='purple', label="Measured Indirect")


clearsky_power = sum(irradiance) + sum(indirect)
cloudy_power = sum(real_irr) + sum(indirect_real)
fraction = cloudy_power / clearsky_power

ax1.text(one_day[0] + timedelta(minutes=30), 0.9*max(irradiance), f"Fraction: {fraction*100:.0f}%", fontsize=12, fontweight="bold")

date1 = day_date - timedelta(hours=1)
date2 = day_date + timedelta(hours=25)
delta = timedelta(minutes=60)
dates = drange(date1, date2, delta)
formatter = DateFormatter('%H:%M')

ax1.set_xlim(date1, date2)

ax1.set_xticks(dates)
ax1.xaxis.set_tick_params(rotation=45, labelsize=10)
ax1.xaxis.set_major_formatter(formatter)
#ax1.set_xlabel("Time", fontsize=15, fontweight= "bold")
ax1.set_ylabel("Irradiance", fontsize=15, fontweight= "bold")
ax1.grid(True)
ax1.legend()

plt.show()


import sys
#sys.exit()

from matplotlib.backends.backend_pdf import PdfPages

    
pdf_path = os.path.join("PDFs","2019_irridiances.pdf")

day_one = datetime(2019,1,1,0,0)
with PdfPages(pdf_path) as pdf:
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


        fig = plt.figure(figsize=(11,6), layout="constrained")
        fig.suptitle(f"Sun irradiance at {day_date.strftime('%Y-%m-%d')}", fontsize=15, fontweight= "bold")
        ax1 = fig.add_subplot(1,1,1)


        ax1.plot(one_day, irradiance, '-', color='firebrick', label="Clearsky")
        ax1.plot(one_day, real_irr, '-', color='cadetblue', label="Measured (with clouds)")
        ax1.plot(one_day, indirect, '-', color='green', label="Indirect Light")
        ax1.plot(one_day, indirect_real, '-', color='purple', label="Measured Indirect")



        clearsky_power = sum(irradiance) + sum(indirect)
        cloudy_power = sum(real_irr) + sum(indirect_real)
        fraction = cloudy_power / clearsky_power
        ax1.text(one_day[0] + timedelta(minutes=30), 0.9*max(irradiance), f"Fraction: {fraction*100:.0f}%", fontsize=12, fontweight="bold")


        date1 = day_date - timedelta(hours=1)
        date2 = day_date + timedelta(hours=25)
        delta = timedelta(minutes=60)
        dates = drange(date1, date2, delta)
        formatter = DateFormatter('%H:%M')

        ax1.set_xlim(date1, date2)

        ax1.set_xticks(dates)
        ax1.xaxis.set_tick_params(rotation=45, labelsize=10)
        ax1.xaxis.set_major_formatter(formatter)
        #ax1.set_xlabel("Time", fontsize=15, fontweight= "bold")
        ax1.set_ylabel("Irradiance", fontsize=15, fontweight= "bold")
        ax1.grid(True)
        ax1.legend()


        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()