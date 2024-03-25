
#Scipt to analyse data from the daily solarpanel curve
#Robrecht Keijzer
#29-6-2023


#Standard Library
import os
from datetime import datetime, timedelta
import time

#Third party modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.dates import (DateFormatter, drange)
rcParams["font.family"] = "Serif"
from suncalc import get_position


#-----PARAMETERS:-----
day_date = datetime(2023,6,24,0,0)
lat = 51.113911     #Hooiweg 31, 2222 Itegem
long = 4.709500
solar_panel_file_path = "solar_data_2023-06-24_raw.txt"         #dit nog aanpassen zodat de datum vanzelf wordt ingevuld
winteruur = False

#-----STEP 1: PREPROCESSING-----
data_df = pd.read_csv(solar_panel_file_path, sep="\t")
names = data_df.keys()
times = np.array(data_df[names[0]])
power = np.array(data_df[names[1]])

#Remove empty data points
to_be_kept = [i != "--" for i in power]
times = times[to_be_kept]
power = power[to_be_kept]

#remove duplicates
previous_data_str = "1"
new_times = []
new_power = []
for output_power, date_str in zip(power, times):
    if date_str != previous_data_str:
        new_times.append(date_str)
        new_power.append(output_power)
        previous_data_str = date_str
        
#convert times to datetime objects
times = [datetime.strptime(date, '%Y-%m-%d %H:%M') for date in new_times]
power = [float(output_power) for output_power in new_power]

print("End of Step 1")

#-----STEP 2: SUN DATA-----

def datetime_from_utc_to_local(utc_datetime):
    """Input = utc time, output = local time"""
    now_timestamp = time.time()
    offset = datetime.fromtimestamp(now_timestamp) - datetime.utcfromtimestamp(now_timestamp)
    if winteruur:
        offset = offset - timedelta(hours = 1)
    return utc_datetime + offset

#By including an offset, the sun position is recorded for a whole day in LOCAL time
now_timestamp = time.time()
offset = datetime.fromtimestamp(now_timestamp) - datetime.utcfromtimestamp(now_timestamp)
if winteruur:
    offset = offset - timedelta(hours = 1)
date1 = day_date - offset
date2 = date1 + timedelta(days=1)
delta = timedelta(minutes=5)

sun_times = pd.date_range(start=date1, end=date2, freq=delta)

#print(sun_times)

df = get_position(sun_times, long, lat)
azimuth = np.array(df["azimuth"])
altitude = np.array(df["altitude"])

sun_times = np.array(sun_times)
#convert plt_dates from numpt.datetime objects to datetime objects:
sun_times = np.array([datetime.strptime(str(date).split(".")[0], "%Y-%m-%dT%H:%M:%S") for date in sun_times])
#convert to local time
sun_times = [datetime_from_utc_to_local(time) for time in sun_times]

#convert sun position to spherical coordinates
theta = [(np.pi/2-alti) for alti in altitude]
phi = [(3*np.pi/2-azi) for azi in azimuth]


sun_x = np.sin(theta)*np.cos(phi)
sun_y = np.sin(theta)*np.sin(phi)
sun_z = np.cos(theta)

sun_vector = np.array([[x, y, z] for x,y,z in zip(sun_x, sun_y, sun_z)])

#print(sun_vector)
print("End of Step 2")

#-----STEP 3: CALCULATE SOLAR PANEL RELATIVE SURFACE AREA-----

def calculate_projection_area(sun_vector, roof_angle):
    """Returns the relative area, with the sun at position normal_vector and the roof at angle roof_angle."""
    sun_vector = sun_vector / np.linalg.norm(sun_vector)
    #MAKE BASIS
    colinear = True
    while colinear:
        test_vector = np.random.rand(3)
        if np.inner(sun_vector, test_vector) != np.linalg.norm(sun_vector) * np.linalg.norm(test_vector):      #this is only false if cos(theta) == 0, so they are colinear
            colinear = False
    ex = np.cross(sun_vector, test_vector)
    ex = ex / np.linalg.norm(ex)
    ey = np.cross(sun_vector, ex)
    #ex, ey and normal_vector form an orthogonal basis

    def project_point(point):
        new_x = np.inner(point, ex)
        new_y = np.inner(point, ey)
        return np.array([new_x, new_y])


    #DEFINE RECTANGLE
    #x-axis is the length axis of the roof, y and z coordinates of rectangle change with roof angle
    def make_rectangle(width_height_ratio = 1, roof_angle = 10): #the panels are wider than they are high
        roof_angle = roof_angle*np.pi/180
        a = 1
        b = width_height_ratio
        p1 = [b/2, -a/2*np.cos(roof_angle), -a/2*np.sin(roof_angle)]
        p2 = [b/2, a/2*np.cos(roof_angle), a/2*np.sin(roof_angle)]
        p3 = [-b/2, a/2*np.cos(roof_angle), a/2*np.sin(roof_angle)]
        p4 = [-b/2, -a/2*np.cos(roof_angle), -a/2*np.sin(roof_angle)]
        normal_direction = [0, -np.sin(roof_angle), np.cos(roof_angle)]
        return np.array([p1, p2, p3, p4, normal_direction])

    points = make_rectangle(roof_angle = roof_angle)
    projected_points = [project_point(point) for point in points]
    projected_points[4] = np.sign(np.inner(points[4], sun_vector))       #if the solar panel is lit from behind, insert minus sign.

    def area_of_polygon(ps):
        """Calculates the area of the projection."""
        x1, y1 = ps[0]
        x2, y2 = ps[1]
        x3, y3 = ps[2]
        x4, y4 = ps[3]
        area = 1/2*ps[4]*abs((x1*y2-x2*y1) + (x2*y3-x3*y2) + (x3*y4-x4*y3) + (x4*y1-x1*y4))
        return area

    return area_of_polygon(projected_points)


areas_at_angles = []
for roof_angle in range(0,100,10):
    areas = [calculate_projection_area(vector, roof_angle) for vector in sun_vector]
    areas_at_angles.append(areas)
    
    
print("End of Step 3")

#-----STEP 4: SCALE EACH AREA WITH THE ATTENUATION FACTOR-----

#Two Hyperparameters: h_atm and the attenuation factor alpha

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
attenuation_factors = [attenuation_factor(alti, max_value) for alti in altitude]

attenuated_areas = []
for area in areas_at_angles:
    attenuated_areas.append([factor*ar for factor, ar in zip(attenuation_factors, area)])

#attenuated_areas = areas_at_angles
print("End of Step 4")

#-----STEP 5: PLOT OF THE SURFACE AREAS AT DIFFERENT ANGLES-----

fig = plt.figure(figsize=(11,6), layout="constrained")
fig.suptitle(f"Attenuated Solar Panel Surface Area: {day_date.strftime('%d-%m-%Y')}", fontsize=15, fontweight= "bold")

ax1 = fig.add_subplot(1,1,1)

date1 = day_date - timedelta(hours=1)
date2 = date1 + timedelta(days=1) + timedelta(hours=1)
delta = timedelta(minutes=60)
dates = drange(date1, date2, delta)
formatter = DateFormatter('%H:%M')

ax1.set_xticks(dates)
ax1.xaxis.set_tick_params(rotation=45, labelsize=10)
ax1.xaxis.set_major_formatter(formatter)

ax1.set_xlabel("Time", fontsize=15, fontweight= "bold")
ax1.set_ylabel("Relative Area", fontsize=15, fontweight= "bold")
ax1.grid(True)
ax1.set_ylim([-0.2,1.1])

for i in range(0,10,1):
    ax1.plot(sun_times, attenuated_areas[i], '-', label=f"{10*i} Degrees")

ax1.plot(ax1.get_xlim(), [0,0], "k--")
ax1.legend()
#plt.savefig("Relative_Areas_24-06-2023.png")
plt.close()
#plt.show()


print("End of Step 5")

#-----STEP 6: PLOT OF THE SOLAR PANEL POWER WITH PREDICTION-----

fig = plt.figure(figsize=(11,6), layout="constrained")
fig.suptitle(f"Solar Panel Power at {day_date.strftime('%d-%m-%Y')} (sunny day)", fontsize=15, fontweight= "bold")
ax1 = fig.add_subplot(1,1,1)

ax1.plot(times, power, '-', color='firebrick', label="Power")

for i in [4,5,6]: #range(10):#[4,5,6]:
    a = attenuated_areas[i]
    scale_factor = max(power)/max(a)
    scaled_areas = [area * scale_factor for area in a]
    ax1.plot(sun_times, scaled_areas, '-', label=f"{i*10} degrees, with attenuation")
    
    b = areas_at_angles[i]
    scale_factor = max(power)/max(b)
    scaled_areas = [area * scale_factor for area in b]
    ax1.plot(sun_times, scaled_areas, '-', label=f"{i*10} degrees, without attenuation")

date1 = times[0] - timedelta(minutes=20)
date2 = times[-1] + timedelta(minutes=20)
delta = timedelta(minutes=60)
dates = drange(date1, date2, delta)
formatter = DateFormatter('%H:%M')

ax1.set_xlim(date1, date2)

ax1.set_xticks(dates)
ax1.xaxis.set_tick_params(rotation=45, labelsize=10)
ax1.xaxis.set_major_formatter(formatter)
ax1.set_xlabel("Time", fontsize=15, fontweight= "bold")
ax1.set_ylabel("Power (kW)", fontsize=15, fontweight= "bold")
ax1.grid(True)
ax1.plot(ax1.get_xlim(), [0,0], "k--")
ax1.legend()
ax1.set_ylim([-0.5,3.1])

#plt.savefig("Power_vs_Surface_Area_24-06-2023.png")
plt.show()

print("End of Step 6")