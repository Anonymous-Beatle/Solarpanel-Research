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
lat = 51.113911     #Hooiweg 31, 2222 Itegem
long = 4.709500



def calc_scale_factor(day_date, degrees, winteruur):
    solar_panel_file_path = f"solar_data_{day_date.strftime('%Y-%m-%d')}_raw.txt"
    
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
    for roof_angle in list(degrees):
        areas = [calculate_projection_area(vector, roof_angle) for vector in sun_vector]
        areas_at_angles.append(areas)



    #-----STEP 4: SCALE EACH AREA WITH THE ATTENUATION FACTOR-----

    #Two Hyperparameters: h_atm and the attenuation factor alpha

    def length_trough_atmosphere(altitude):
        r_a = 6371      #radius of the earth
        h_atm = 30     #new hyperparameter: 'height' of a homogeneous atmosphere.

        u = 2*r_a*np.sin(altitude)
        v = r_a**2-(r_a+h_atm)**2

        c1 = 1/2*(-u + np.sqrt(u**2-4*v))
        return c1

    def attenuation_factor(altitude, max_value):
        l0 = length_trough_atmosphere(max_value)
        l1 = length_trough_atmosphere(altitude)
        alpha = 0.004
        if altitude > 0:
            return np.exp(-alpha*(l1-l0))
        else:
            return 0


    max_value = max(altitude)
    attenuation_factors = [attenuation_factor(alti, max_value) for alti in altitude]

    attenuated_areas = []
    for area in areas_at_angles:
        attenuated_areas.append([factor*ar for factor, ar in zip(attenuation_factors, area)])

    pred_max = []
    for i in range(len(degrees)):
        pred_max.append(max(attenuated_areas[i]))    
    real_max = max(power)
    
    return attenuated_areas, pred_max, real_max







dates = [datetime(2023,2,13,0,0), datetime(2023,3,2,0,0), datetime(2023,4,3,0,0), datetime(2023,5,3,0,0), datetime(2023,6,24,0,0)]
degrees = [40,45,50,55]
winteruur = [True, True, False, False, False]





predictions_max = []
measured_max = []
attenuated_areas = []
for date, winteru in zip(dates, winteruur):
    attenuated_area, pred, real = calc_scale_factor(date, degrees, winteru)
    predictions_max.append(pred)
    measured_max.append(real)
    attenuated_areas.append(attenuated_area)
    
predictions_max = np.transpose(predictions_max)

print(predictions_max)
print(measured_max)


plt.figure(figsize=(12,6), layout="constrained")
for i, degr in enumerate(degrees):
    plt.plot(np.arange(0,len(dates)), predictions_max[i], 'o-', label=f"{degr} degrees")
plt.ylabel("Scale factors")
month_names = ["Feb", "Mar", "Apr", "May", "Jun"]
plt.xticks(np.arange(0,len(dates)), month_names)
plt.title("Predicted Power")

plt.legend()



scale_factors = []
for pred, real in zip(predictions_max, measured_max):
    angle_pred = []
    for little_pred in pred:
        angle_pred.append(real/little_pred)
    scale_factors.append(angle_pred)
    
#CHECK IF PREDICTIONS ARE THE SAME WITH OR WITHOUT ATTENUATION 
#Eigenlijk echt niet logisch, de attenuation factor heb ik laten afhangen van de maximale hoogte die de zon bereikt boven de horizon.
#In principe als de zon in de winter minder hoog komt, moet de attenuation factor ook verkleinen.

plt.figure(figsize=(12,6), layout="constrained")
for i, degr in enumerate(degrees):
    plt.plot(np.arange(0,len(dates)), scale_factors[i], 'o-', label=f"{degr} degrees")
plt.ylabel("Scale factors")
month_names = ["Feb", "Mar", "Apr", "May", "Jun"]
plt.xticks(np.arange(0,len(dates)), month_names)
plt.title("Month april was capped: Scale factor should be larger")

plt.legend()


width = 0.2

plt.figure(figsize = (12,6), layout="constrained")
# Plotting
for i, degr in enumerate(degrees):
    plt.bar(np.arange(0,len(dates)) + i*width, scale_factors[i] , width, label=f'{degr} degrees')

plt.xlabel('Month', size = 12, fontweight = "bold")
plt.ylabel('Scale Factor Needed', size = 12, fontweight = "bold")
plt.title('Scale factors', size = 12, fontweight = "bold")

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations

month_names = ["Feb", "Mar", "Apr", "May", "Jun"]

plt.xticks(np.arange(0,len(dates)) + width / 3, month_names)
#y_ticks = [f"{i}%" for i in range(0,80,10)]
#plt.yticks(np.arange(0,0.8,0.1), y_ticks)

plt.legend()
plt.show()


