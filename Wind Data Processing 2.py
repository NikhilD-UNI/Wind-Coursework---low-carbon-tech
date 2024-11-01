# %%
import pandas as pd
import numpy as np 
from windrose import WindroseAxes
import matplotlib.pyplot as plt
from windrose import WindAxes
from scipy import stats


pa = 'C:/Users/nikhi/OneDrive - Imperial College London/SEF/Low Carbon Technologies/Wind/Coursework/Wind data/Wind data/9 Manston 2008.xls '# put in file directory path
df = pd.read_excel( pa, sheet_name='MANSTON', usecols=[0,1,2,3,4,5])
df_cleaned = df.dropna()
ws = df_cleaned['Wind - Mean Speed (knots)']*0.514
wd = df_cleaned['Wind - Mean Direction']

ax = WindroseAxes.from_ax()
ax.bar(wd, ws, normed=False, bins = [0,3,6,12,18,25], opening=0.8, edgecolor='white', nsector = 36)
ax.set_legend(title=r"$m \cdot s^{-1}$", loc="lower right")
y_ticks = range(0, 1000, 200)
ax.set_rgrids(y_ticks,y_ticks)

plt.show()

#plt.savefig('my_plot.png')  # This saves the image as a .png file

#print(df_cleaned)

# %%
## WEIBULL Distribution in PYTHON

wind_speed_data = ws
shape, loc, scale = stats.weibull_min.fit(wind_speed_data, floc=0, scale = 12)
x = np.linspace(min(wind_speed_data), max(wind_speed_data), 100)
weibull_pdf = stats.weibull_min.pdf(x, shape, loc, scale)

plt.figure(figsize=(8, 6))

# Histogram of wind speed data
plt.hist(wind_speed_data, bins=40, density=True, alpha=0.6, color='g', label="Wind Speed Data")

# Weibull distribution curve
plt.plot(x, weibull_pdf, 'r-', lw=2, label=f'Weibull Fit (shape={shape:.2f}, scale={scale:.2f})')

# Labels and legend
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Probability Density')
plt.title('Weibull Distribution Fit to Wind Speed Data')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# %%
