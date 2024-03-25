
#Relation between total irradiance (Clearsky DNI + Clearsky DHI) and sun altitude.
#For a homogeneous atmosphere, the azimuth angle really shouldn't matter.
#Robrecht Keijzer
#30-03-2023

import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from suncalc import get_position
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.dates import (HOURLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange, HourLocator)
import pandas as pd
import time
rcParams["font.family"] = "Serif"

#-----PARAMETERS-----
file_path = os.path.join("NSRDB data", "464295_51.13_4.70_2019.csv")
day_date = datetime(2019,7,21,0,0)
lat = 51.113911     #Hooiweg 31, 2222 Itegem
long = 4.709500



df = pd.read_csv(file_path, skiprows=2)
names = df.columns
time_strings = [f"{year}-{month}-{day}-{hour}-{minute}" for year,month,day,hour,minute \
    in zip(list(df[names[0]]),list(df[names[1]]),list(df[names[2]]),list(df[names[3]]),list(df[names[4]]))]


#-----FIRST DAY-----
end_date = day_date + timedelta(days=1)

start_index = time_strings.index(day_date.strftime('%Y-%m-%d-%H-%M').replace("-0", "-"))
end_index = time_strings.index(end_date.strftime('%Y-%m-%d-%H-%M').replace("-0", "-"))

one_day = time_strings[start_index:end_index]

#convert to datetime
one_day = [datetime.strptime(time, '%Y-%m-%d-%H-%M') for time in one_day]
irradiance = list(df["Clearsky DNI"])[start_index:end_index]
indirect = list(df["Clearsky DHI"])[start_index:end_index]
total_irrad = [irr+indi for irr, indi in zip(irradiance, indirect)]



date1 = day_date
date2 = date1 + timedelta(days=1)
delta = timedelta(minutes=15)

sun_times = pd.date_range(start=date1, end=date2, freq=delta)


df2 = get_position(sun_times, long, lat)
azimuth = np.array(df2["azimuth"])
altitude = np.array(df2["altitude"])

sun_times = np.array(sun_times)
#convert plt_dates from numpt.datetime objects to datetime objects:
sun_times = np.array([datetime.strptime(str(date).split(".")[0], "%Y-%m-%dT%H:%M:%S") for date in sun_times])



#Trying to find the optimal atmosphere attenuation:
def length_trough_atmosphere(altitude):
    r_a = 6371      #radius of the earth
    h_atm = 90     #new hyperparameter: 'height' of a homogeneous atmosphere.
    
    u = 2*r_a*np.sin(altitude)
    v = r_a**2-(r_a+h_atm)**2
    
    c1 = 1/2*(-u + np.sqrt(u**2-4*v))
    return c1

def attenuation_factor(altitude, max_value):
    l0 = length_trough_atmosphere(max_value)
    l1 = length_trough_atmosphere(altitude)
    alpha = 0.0025
    return np.exp(-alpha*(l1-l0))


max_value = max(altitude)
irr_prediction = [attenuation_factor(alti, max_value) for alti in altitude]



fig = plt.figure(figsize=(14,5), layout="constrained")
fig.suptitle(f"Sun irradiance", fontsize=15, fontweight= "bold")
ax1 = fig.add_subplot(1,2,1)


ax1.plot(one_day, total_irrad, '-', color='firebrick', label="Clearsky")

scale_factor = max(total_irrad)/max(irr_prediction)
scaled_irr_prediction = [scale_factor*pred for pred in irr_prediction]
ax1.plot(sun_times, scaled_irr_prediction, '-', color='orange', label="Atmosphere model")


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






#-----SECOND DAY-----
day_date = datetime(2019,1,6,0,0)
end_date = day_date + timedelta(days=1)

start_index = time_strings.index(day_date.strftime('%Y-%m-%d-%H-%M').replace("-0", "-"))
end_index = time_strings.index(end_date.strftime('%Y-%m-%d-%H-%M').replace("-0", "-"))

one_day = time_strings[start_index:end_index]

#convert to datetime
one_day = [datetime.strptime(time, '%Y-%m-%d-%H-%M') for time in one_day]
irradiance = list(df["Clearsky DNI"])[start_index:end_index]
indirect = list(df["Clearsky DHI"])[start_index:end_index]
total_irrad = [irr+indi for irr, indi in zip(irradiance, indirect)]



date1 = day_date
date2 = date1 + timedelta(days=1)
delta = timedelta(minutes=15)

sun_times = pd.date_range(start=date1, end=date2, freq=delta)


df2 = get_position(sun_times, long, lat)
azimuth = np.array(df2["azimuth"])
altitude = np.array(df2["altitude"])

sun_times = np.array(sun_times)
#convert plt_dates from numpt.datetime objects to datetime objects:
sun_times = np.array([datetime.strptime(str(date).split(".")[0], "%Y-%m-%dT%H:%M:%S") for date in sun_times])





max_value = max(altitude)
irr_prediction = [attenuation_factor(alti, max_value) for alti in altitude]



#fig = plt.figure(figsize=(7,4.5), layout="constrained")
#fig.suptitle(f"Sun irradiance at {day_date.strftime('%Y-%m-%d')}", fontsize=15, fontweight= "bold")
ax2 = fig.add_subplot(1,2,2)


ax2.plot(one_day, total_irrad, '-', color='firebrick', label="Clearsky")

scale_factor = max(total_irrad)/max(irr_prediction)
scaled_irr_prediction = [scale_factor*pred for pred in irr_prediction]
ax2.plot(sun_times, scaled_irr_prediction, '-', color='orange', label="Atmosphere model")


date1 = day_date - timedelta(hours=1)
date2 = day_date + timedelta(hours=25)
delta = timedelta(minutes=60)
dates = drange(date1, date2, delta)
formatter = DateFormatter('%H:%M')

ax2.set_xlim(date1, date2)

ax2.set_xticks(dates)
ax2.xaxis.set_tick_params(rotation=45, labelsize=10)
ax2.xaxis.set_major_formatter(formatter)
#ax1.set_xlabel("Time", fontsize=15, fontweight= "bold")
ax2.set_ylabel("Irradiance", fontsize=15, fontweight= "bold")
ax2.grid(True)
ax2.legend()

plt.close()
#plt.show()


day_one = datetime(2019,1,1,0,0)

real_maxima = []
predicted_maxima = []

for day in range(365):
    
    day_date = day_one + timedelta(days=day)
    end_date = day_date + timedelta(days=1)-timedelta(minutes=15)
    start_index = time_strings.index(day_date.strftime('%Y-%m-%d-%H-%M').replace("-0", "-"))
    end_index = time_strings.index(end_date.strftime('%Y-%m-%d-%H-%M').replace("-0", "-"))
    one_day = time_strings[start_index:end_index]
    #--model predictions:--
    date1 = day_date
    date2 = date1 + timedelta(days=1)
    delta = timedelta(minutes=15)
    sun_times = pd.date_range(start=date1, end=date2, freq=delta)
    df2 = get_position(sun_times, long, lat)
    azimuth = np.array(df2["azimuth"])
    altitude = np.array(df2["altitude"])
    sun_times = np.array(sun_times)
    #convert plt_dates from numpt.datetime objects to datetime objects:
    sun_times = np.array([datetime.strptime(str(date).split(".")[0], "%Y-%m-%dT%H:%M:%S") for date in sun_times])
    #Trying to find the optimal atmosphere attenuation:
    def length_trough_atmosphere(altitude):
        r_a = 6371      #radius of the earth
        h_atm = 90     #new hyperparameter: 'height' of a homogeneous atmosphere.
        u = 2*r_a*np.sin(altitude)
        v = r_a**2-(r_a+h_atm)**2
        c1 = 1/2*(-u + np.sqrt(u**2-4*v))
        return c1
    def attenuation_factor(altitude, max_value):
        l0 = length_trough_atmosphere(max_value)
        b = 1           #1500 works good for january
        l1 = length_trough_atmosphere(altitude)
        alpha = 0.0025
        return b*np.exp(-alpha*(l1))
    max_value = max(altitude)
    irr_prediction = [attenuation_factor(alti, max_value) for alti in altitude]
    #convert to datetime
    one_day = [datetime.strptime(time, '%Y-%m-%d-%H-%M') for time in one_day]
    irradiance = list(df["Clearsky DNI"])[start_index:end_index]
    indirect = list(df["Clearsky DHI"])[start_index:end_index]
    total_irrad = [irr+indi for irr, indi in zip(irradiance, indirect)]
    
    real_maxima.append(max(total_irrad))
    predicted_maxima.append(max(irr_prediction))
    
    
    
scale_factor = [real/pred for real, pred in zip(real_maxima, predicted_maxima)]

n = 30
conv_scale_factor = np.convolve(scale_factor, np.ones(n)/n)


width = 0.8       

plt.figure(figsize = (12,6), layout="constrained")
# Plotting
plt.plot(conv_scale_factor[n:-n], '.-')

plt.xlabel('Month', size = 12, fontweight = "bold")
plt.ylabel('Scale Factor', size = 12, fontweight = "bold")
plt.title('Scale Factors troughout the year', size = 12, fontweight = "bold")

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations

#month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun","Jul","Aug","Sep","Okt","Nov","Dec"]

#plt.xticks(np.arange(1,13), month_names)
#y_ticks = [f"{i}%" for i in range(0,80,10)]
#plt.yticks(np.arange(0,0.8,0.1), y_ticks)


plt.text(160,1500, "Conclusion: When accounting for atmospheric attenuation, we can get the correct daily curve's shape.\n \
         For a very simple atmosphere, the irradiation would just be the irradiation for the sun at zenith * attenuation factor, or scale factor * attenuation.\n \
         However, attenuation alone is not enough, since this scale factor isn't constant. There are daily, monthly and even yearly variations.\n \
         This is probably to due to complex local atmospheric characteristics, such as humidity, composition, temperature windspeed etc. \
         ", 
         horizontalalignment='center', verticalalignment='center', size = 8, fontweight="bold")

plt.savefig("2019_scale_factors.png", dpi=500)
plt.show()

        





import sys
sys.exit()

from matplotlib.backends.backend_pdf import PdfPages

    
pdf_path = os.path.join("PDFs","2019_model_predictions2.pdf")

day_one = datetime(2019,1,1,0,0)
with PdfPages(pdf_path) as pdf:
    for day in range(365):
        
        day_date = day_one + timedelta(days=day)
        end_date = day_date + timedelta(days=1)-timedelta(minutes=15)

        start_index = time_strings.index(day_date.strftime('%Y-%m-%d-%H-%M').replace("-0", "-"))
        end_index = time_strings.index(end_date.strftime('%Y-%m-%d-%H-%M').replace("-0", "-"))
        one_day = time_strings[start_index:end_index]


        #--model predictions:--
        date1 = day_date
        date2 = date1 + timedelta(days=1)
        delta = timedelta(minutes=15)
        sun_times = pd.date_range(start=date1, end=date2, freq=delta)

        df2 = get_position(sun_times, long, lat)
        azimuth = np.array(df2["azimuth"])
        altitude = np.array(df2["altitude"])

        sun_times = np.array(sun_times)
        #convert plt_dates from numpt.datetime objects to datetime objects:
        sun_times = np.array([datetime.strptime(str(date).split(".")[0], "%Y-%m-%dT%H:%M:%S") for date in sun_times])
        #Trying to find the optimal atmosphere attenuation:
        def length_trough_atmosphere(altitude):
            r_a = 6371      #radius of the earth
            h_atm = 90     #new hyperparameter: 'height' of a homogeneous atmosphere.

            u = 2*r_a*np.sin(altitude)
            v = r_a**2-(r_a+h_atm)**2

            c1 = 1/2*(-u + np.sqrt(u**2-4*v))
            return c1

        def attenuation_factor(altitude, max_value):
            l0 = length_trough_atmosphere(max_value)
            b = 1500
            l1 = length_trough_atmosphere(altitude)
            alpha = 0.0025
            return b*np.exp(-alpha*(l1))

        max_value = max(altitude)
        irr_prediction = [attenuation_factor(alti, max_value) for alti in altitude]


        #convert to datetime
        one_day = [datetime.strptime(time, '%Y-%m-%d-%H-%M') for time in one_day]
        irradiance = list(df["Clearsky DNI"])[start_index:end_index]
        indirect = list(df["Clearsky DHI"])[start_index:end_index]
        total_irrad = [irr+indi for irr, indi in zip(irradiance, indirect)]

        fig = plt.figure(figsize=(11,6), layout="constrained")
        fig.suptitle(f"Sun irradiance at {day_date.strftime('%Y-%m-%d')}", fontsize=15, fontweight= "bold")
        ax1 = fig.add_subplot(1,1,1)


        ax1.plot(one_day, total_irrad, '-', color='firebrick', label="Clearsky")
        #scale_factor = max(total_irrad)/max(irr_prediction)
        #scaled_irr_prediction = [scale_factor*pred for pred in irr_prediction]
        ax1.plot(sun_times, irr_prediction, '-', color='orange', label="Atmosphere model")




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