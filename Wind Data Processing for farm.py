
# %%
import pandas as pd
import numpy as np 
from windrose import WindroseAxes
import matplotlib.pyplot as plt
from windrose import WindAxes
from scipy.stats import weibull_min
from scipy import optimize
from scipy.special import gamma


pa = 'C:/Users/nikhi/OneDrive - Imperial College London/SEF/Low Carbon Technologies/Wind/Coursework/Wind data/Wind data/9 Manston 2008.xls '# put in file directory path

def calculate_turbulence_intensity(df):
    ti_values = {}
    grouped = df.groupby('Wind - Mean Direction')
    
    for direction, group in grouped:
        mean_speed = group['Wind - Mean Speed (m/s)'].mean()
        std_dev_speed = group['Wind - Mean Speed (m/s)'].std()
        ti = std_dev_speed / mean_speed  # Turbulence Intensity formula
        
        # Store TI value for each direction
        ti_values[direction] = ti
    
    return ti_values


# Load the data (assuming you have the correct path and sheet)
df = pd.read_excel(pa, sheet_name='MANSTON', usecols=[0, 1, 2, 3, 4, 5])
df = df.dropna()
df['Wind - Mean Speed (knots)'] = df['Wind - Mean Speed (knots)']*0.51444
df.rename(columns={'Wind - Mean Speed (knots)': 'Wind - Mean Speed (m/s)'}, inplace=True)


# Extract relevant columns (assuming wind speed is in column 4 and direction in column 5)
df2 = df.iloc[:, [4, 5]]

# Remove data with zero wind direction
df2 = df2[df2[df2.columns[1]] != 0]

df3 = df2
# Calculate TI for each wind direction
ti_values = calculate_turbulence_intensity(df3)
for direction, ti in ti_values.items():
    print(f"Direction {direction}°: TI = {ti:.2f}")

# Grouping the data by wind direction
grouped_df = df2.groupby(df2.columns[1])

# Prepare to store Weibull parameters (a, k), directions, and frequency
a_values = []
k_values = []
directions = []
frequencies = []

# Calculate the total number of entries to compute relative frequency
total_entries = len(df2)

# Function to solve for Weibull k and a using Method of Moments
def solve_weibull_params(mean, variance):
    # Numerical method to solve for k using sample mean and variance
    def equation_to_solve(k):
        gamma1 = gamma(1 + 1/k)
        gamma2 = gamma(1 + 2/k)
        return (variance / mean**2) - (gamma2 / gamma1**2 - 1)
    
    # Use an initial guess for k and solve numerically (simple search)
    k_guess = 1.5
    k_opt = optimize.newton(equation_to_solve, k_guess)
    
    # Calculate a based on the optimized k
    a = mean / gamma(1 + 1/k_opt)
    
    return k_opt, a

# Iterate through each wind direction group
for name, group in grouped_df:
    wind_speeds = group[group.columns[0]]

    # Calculate sample mean and variance
    mean = wind_speeds.mean()
    variance = wind_speeds.var()

    # Use Method of Moments to estimate Weibull k and a
    if variance > 0 and len(wind_speeds) > 1:  # Only if we have enough data points
        try:
            k_moments, a_moments = solve_weibull_params(mean, variance)
        except Exception as e:
            print(f"Error estimating Weibull parameters for direction {name}: {e}")
            k_moments, a_moments = np.nan, np.nan
    else:
        k_moments, a_moments = np.nan, np.nan  # Not enough variation in the data

    # Store results
    a_values.append(a_moments)
    k_values.append(k_moments)
    directions.append(name)

    # Frequency calculation
    frequency = len(group) / total_entries
    frequencies.append(frequency)

# Create a new DataFrame to store the results
weibull_df = pd.DataFrame({
    'Wind Direction': directions,
    'Weibull Scale (a)': a_values,
    'Weibull Shape (k)': k_values,
    'Frequency': frequencies
})


for name, group in grouped_df:
    print(f"Wind Direction: {name}, Count: {len(group)}")

# Display the final DataFrame
print(weibull_df)

#Plot Turbulence Intensity for each direction
plt.figure(figsize=(12, 6))
plt.plot(ti_values.keys(), ti_values.values(), 'r--x', linewidth=2)
plt.xlabel('Wind Direction (°)')
plt.ylabel('Turbulence Intensity (TI)')
plt.title('Turbulence Intensity by Wind Direction')
plt.grid(True)
plt.show()

# Plot Wind Speed vs Hours
plt.figure(figsize=(12, 6))
plt.plot(range(0,len(df3['Wind - Mean Speed (m/s)']),1), df3['Wind - Mean Speed (m/s)'], color='purple', linewidth=0.5)
plt.xlabel('Hours')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed vs. Hours')
plt.grid(True)
plt.show()

