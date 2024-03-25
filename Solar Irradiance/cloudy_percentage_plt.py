import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "serif"
import numpy as np


year_monthly_average = np.load("year_monthly_av.npy")

# Width of a bar 
width = 0.2       

plt.figure(figsize = (7,4.5), layout="constrained")
# Plotting
plt.bar(np.arange(1,13), year_monthly_average[0] , width, label='2017')
plt.bar(np.arange(1,13)+width, year_monthly_average[1] , width, label='2018')
plt.bar(np.arange(1,13)+2*width, year_monthly_average[2] , width, label='2019')

plt.xlabel('Month', size = 12, fontweight = "bold")
plt.ylabel('Remaining power with cloudiness', size = 12, fontweight = "bold")
plt.title('Remaining power in each month with cloudiness', size = 12, fontweight = "bold")

# xticks()
# First argument - A list of positions at which ticks should be placed
# Second argument -  A list of labels to place at the given locations

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun","Jul","Aug","Sep","Okt","Nov","Dec"]

plt.xticks(np.arange(1,13) + width / 3, month_names)
y_ticks = [f"{i}%" for i in range(0,80,10)]
plt.yticks(np.arange(0,0.8,0.1), y_ticks)

plt.legend()
#plt.show()



year_monthly_av = np.transpose(year_monthly_average)

standard_error = [np.std(month)/np.sqrt(3-1) for month in year_monthly_av]
mean = [np.mean(month) for month in year_monthly_av]

plt.savefig("Cloudiness_barplot.png", dpi=500)
#plt.close()



   

plt.figure(figsize = (7,4.5), layout="constrained")
# Plotting
plt.errorbar(np.arange(1,13), mean, yerr=standard_error, marker='.', color="firebrick", ls='none', capsize=10)

plt.xlabel('Month', size = 12, fontweight = "bold")
plt.ylabel('Remaining power with cloudiness', size = 12, fontweight = "bold")
plt.title('Remaining power in each month with cloudiness', size = 12, fontweight = "bold")

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun","Jul","Aug","Sep","Okt","Nov","Dec"]

plt.xticks(np.arange(1,13), month_names)

y_ticks = [f"{i}%" for i in range(0,80,10)]
plt.yticks(np.arange(0,0.8,0.1), y_ticks)

plt.text(6.5,0.05, "Conclusion: In summer less clouds than in winter", 
         horizontalalignment='center', verticalalignment='center', size = 12, fontweight="bold")
plt.savefig("Cloudiness_errorbar.png", dpi=500)
plt.show()